import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve


def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def createEH(edges, halfedges):
    # Create dictionaries to map halfedges to their indices
    halfedges_dict = {(v1, v2): i for i, (v1, v2) in enumerate(halfedges)}
    # reversed_halfedges_dict = {(v2, v1): i for i, (v1, v2) in enumerate(halfedges)}

    EH = np.zeros((len(edges), 2), dtype=int)

    for i, (v1, v2) in enumerate(edges):
        # Check if the halfedge exists in the original order
        if (v1, v2) in halfedges_dict:
            EH[i, 0] = halfedges_dict[(v1, v2)]
        # Check if the halfedge exists in the reversed order
        if (v2, v1) in halfedges_dict:
            EH[i, 1] = halfedges_dict[(v2, v1)]

    return EH


def compute_edge_list(vertices, faces, sortBoundary=False):
    halfedges = np.empty((3 * faces.shape[0], 2))
    for face in range(faces.shape[0]):
        for j in range(3):
            halfedges[3 * face + j, :] = [faces[face, j], faces[face, (j + 1) % 3]]

    edges, firstOccurence, numOccurences = np.unique(np.sort(halfedges, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    edges = halfedges[np.sort(firstOccurence)]
    edges = edges.astype(int)
    halfedgeBoundaryMask = np.zeros(halfedges.shape[0])
    halfedgeBoundaryMask[firstOccurence] = 2 - numOccurences
    edgeBoundMask = halfedgeBoundaryMask[np.sort(firstOccurence)]

    boundEdges = edges[edgeBoundMask == 1, :]
    boundVertices = np.unique(boundEdges).flatten()

    # EH = [np.where(np.sort(halfedges, axis=1) == edge)[0] for edge in edges]
    # EF = []

    EH = createEH(edges, halfedges)
    EF = np.column_stack((EH[:, 0] // 3, (EH[:, 0] + 2) % 3, EH[:, 1] // 3, (EH[:, 1] + 2) % 3))

    if (sortBoundary):
        loop_order = []
        loopEdges = boundEdges.tolist()
        current_node = boundVertices[0]  # Start from any node
        visited = set()
        while True:
            loop_order.append(current_node)
            visited.add(current_node)
            next_nodes = [node for edge in loopEdges for node in edge if
                          current_node in edge and node != current_node and node not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            loopEdges = [edge for edge in loopEdges if
                         edge != (current_node, next_node) and edge != (next_node, current_node)]
            current_node = next_node
            current_node = next_node

        boundVertices = np.array(loop_order)

    # computing onerings
    oneRings = [[] for _ in range(vertices.shape[0])]

    # appending the middle vertex
    for v in range(vertices.shape[0]):
        oneRings[v].append(v)

    for edgeIndex in range(edges.shape[0]):
        oneRings[edges[edgeIndex, 0]].append(edges[edgeIndex, 1])
        oneRings[edges[edgeIndex, 1]].append(edges[edgeIndex, 0])  # Assuming undirected graph

    return halfedges, edges, edgeBoundMask, boundVertices, EH, EF, oneRings


def compute_operators(vertices, faces, edges, edgeBoundMask, EF):
    edgeInnerMask = np.logical_not(edgeBoundMask)

    vj = vertices[faces[EF[:, 0], EF[:, 1]], :]
    vk = vertices[faces[EF[:, 0], (EF[:, 1] + 1) % 3], :]
    vi = vertices[faces[EF[:, 0], (EF[:, 1] + 2) % 3], :]
    vl = vertices[faces[EF[edgeInnerMask, 2], EF[edgeInnerMask, 3]], :]

    # cotangent weights
    vecDotijk = np.sum((vi - vj) * (vk - vj), axis=1)
    vecCrossNormijk = np.linalg.norm(np.cross(vi - vj, vk - vj), axis=1)
    w = 0.5 * vecDotijk / vecCrossNormijk
    vecDotkli = np.sum((vi[edgeInnerMask, :] - vl) * (vk[edgeInnerMask, :] - vl), axis=1)
    vecCrossNormkli = np.linalg.norm(np.cross(vi[edgeInnerMask, :] - vl, vk[edgeInnerMask, :] - vl), axis=1)
    w[edgeInnerMask] += 0.5 * vecDotkli / vecCrossNormkli

    # d0 operator
    rows = np.arange(0, edges.shape[0])
    rows = np.tile(rows[:, np.newaxis], 2)
    values = np.full((edges.shape[0], 2), [-1, 1])

    # Creating a sparse matrix
    d0 = csc_matrix((values.flatten(), (rows.flatten(), edges.flatten())), shape=(edges.shape[0], vertices.shape[0]))

    # voronoi areas
    v01 = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    v02 = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    normals = np.cross(v01, v02)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    return d0, diags(w, 0)


def ARAP_deformation(origVertices, edges, d0, W, oneRings, constHandles, constPositions, numIterations):

    #starting from the original vertices and the identity local rotations
    currVertices = np.copy(origVertices)
    R = np.array([np.eye(3) for _ in range(origVertices.shape[0])])

    #the right-hand side of rotated edges for the global least-squares
    g = np.zeros((edges.shape[0], 3))

    #variables vs fixed points
    varIndices = np.setdiff1d(np.arange(0, origVertices.shape[0]), constHandles)
    d0Const = d0[:, constHandles]
    d0Var = d0[:, varIndices]

    for currIter in range(numIterations):

        # Global step: rotate each edge ij by (Ri+Rj)/2 for the previous or initial iteration's R (see Lecture notes) and try to integrate to it
        for edgeIndex in range(edges.shape[0]):
            mutualR = (R[edges[edgeIndex, 0]] + R[edges[edgeIndex, 1]]) / 2.0
            g[edgeIndex] = (origVertices[edges[edgeIndex, 0], :] - origVertices[edges[edgeIndex, 1], :]) @ mutualR

        rhs = d0Var.transpose() @ W @ (g + d0Const * constPositions)
        lhs = d0Var.transpose() @ W @ d0Var

        currVertices[varIndices, :] = spsolve(lhs, rhs)
        currVertices[constHandles, :] = constPositions

        # local step - for existing currVertices and original positions origVertices, find the best fitting local one-ring-based rotation matrices R
        for oneRingIndex in np.arange(len(oneRings)):
            P = origVertices[oneRings[oneRingIndex][1:], :] - origVertices[oneRings[oneRingIndex][0], :]
            Q = currVertices[oneRings[oneRingIndex][1:], :] - currVertices[oneRings[oneRingIndex][0], :]
            S = P.transpose() @ Q
            U, sig, Vt = np.linalg.svd(S)
            currR = Vt @ U
            if np.linalg.det(currR) < 0.0:
                currR = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            R[oneRingIndex] = currR

    return currVertices
