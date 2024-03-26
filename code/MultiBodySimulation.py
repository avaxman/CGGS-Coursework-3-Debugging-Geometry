import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from MBSFunctions import run_timestep


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def callback():
    # Executed every frame
    global spherePoses, sphereVelocities, CRCoeff, timeStep, isAnimating, radius, gravityConstant

    # UI stuff
    psim.PushItemWidth(50)

    psim.TextUnformatted("Animation Parameters")
    psim.Separator()
    changed, isAnimating = psim.Checkbox("isAnimating", isAnimating)

    psim.PopItemWidth()

    # Actual animation
    if not isAnimating:
        return

    #this runs the current simulation loop step
    run_timestep(spherePoses, sphereVelocities, radius, gravityConstant, timeStep, CRCoeff)

    #Pure GUI stuff; no need to otouch
    currVertices = np.zeros((numSpheres * origVertices.shape[0], 3))
    for sphereIndex in np.arange(numSpheres):
        currVertices[(sphereIndex * origVertices.shape[0]):((sphereIndex + 1) * origVertices.shape[0]),:] = origVertices + spherePoses[sphereIndex, :].reshape((1, 3))

    ps_mesh.update_vertex_positions(currVertices)

if __name__ == '__main__':
    ps.init()

    origVertices, faces = load_off_file(os.path.join('..', 'data', 'spherers.off'))


    #randomizing a set of spheres of radius 0.1 above the plane
    numSpheres = 50
    spherePoses = np.random.uniform(-2.0, 2.0, size=(numSpheres, 3))
    spherePoses[:, 1] += 2.2


    isAnimating = False
    timeStep = 0.02
    CRCoeff = 0.0
    radius = 0.1
    origVertices *= 2.0*radius  #This is just for GUI; you don't need to touch it.
    gravityConstant = 0.1
    sphereVelocities = np.zeros_like(spherePoses)

    ###GUI stuff
    currVertices = np.zeros((numSpheres*origVertices.shape[0], 3))
    fullFaces = np.zeros((numSpheres*faces.shape[0], 3))
    for sphereIndex in np.arange(numSpheres):
        currVertices[(sphereIndex*origVertices.shape[0]):((sphereIndex+1)*origVertices.shape[0]), :] = origVertices+ spherePoses[sphereIndex, :].reshape((1, 3))
        fullFaces[(sphereIndex*faces.shape[0]):((sphereIndex+1)*faces.shape[0]), :] = faces + origVertices.shape[0]*sphereIndex
    ps_mesh = ps.register_surface_mesh("Balls", currVertices, fullFaces, smooth_shade=True)
    ps.set_length_scale(1.0)
    ps.set_bounding_box([-5.0, 0, -5.0], [5.0, 5.0, 5.0])
    ps.set_ground_plane_height_factor(0.0, True)
    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()
