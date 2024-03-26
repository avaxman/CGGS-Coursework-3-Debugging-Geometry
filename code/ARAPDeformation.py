import os
import polyscope as ps
import numpy as np
from ARAPFunctions import load_off_file, compute_operators,  compute_edge_list, ARAP_deformation



if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('..', 'data', 'horsers.off'))

    ps_orig_mesh = ps.register_surface_mesh("Input Mesh", vertices, faces)
    ps_curr_mesh = ps.register_surface_mesh("Output Mesh", vertices, faces)

    _, edges, edgeBoundMask, boundVertices, EH, EF, oneRings = compute_edge_list(vertices, faces)

    d0, W = compute_operators(vertices, faces, edges, edgeBoundMask, EF)

    constHandles = [535, 780, 793, 827, 769]
    numIterations = 25

    constPositions = vertices[constHandles, :]
    constPositions[0, :] += [0.01, 0.06, 0.01]
    constPositions[1, :] += [0, 0.05, -0.05]
    constPositions[2, :] += [0, 0.05, -0.06]

    currVertices = ARAP_deformation(vertices, edges, d0, W, oneRings, constHandles, constPositions, numIterations)

    ps_curr_mesh.update_vertex_positions(currVertices)

    ps.show()
