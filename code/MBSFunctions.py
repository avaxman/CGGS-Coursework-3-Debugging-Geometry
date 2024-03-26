
import numpy as np

def run_timestep(spherePoses, sphereVelocities, radius, gravityConstant, timeStep, CRCoeff):

    # free integration

    # collecting all gravitational forces
    forces = np.tile([0.0, -9.8, 0.0], (spherePoses.shape[0], 1))  # floor's gravity
    sqrRadius = radius ** 2

    # pairwise gravitational forces
    for s1 in range(spherePoses.shape[0]):
        for s2 in range(s1 + 1, spherePoses.shape[0]):
            forceDirection = (spherePoses[s2, :] - spherePoses[s1, :])
            sqrDistance = np.sum(forceDirection ** 2)
            if sqrDistance > 4.1 * sqrRadius:  #otherwise the objects are too close and it will overshoot (tunnelling)
                forceDirection = forceDirection/np.sqrt(sqrDistance)
                forces[s1, :] -= forceDirection*gravityConstant / sqrDistance
                forces[s2, :] += forceDirection*gravityConstant / sqrDistance

    # semi-implicit integration
    sphereVelocities += timeStep * forces
    spherePoses += timeStep * sphereVelocities

    # doing collision detection and resolution
    for s1 in range(spherePoses.shape[0]):
        # between every two spheres
        for s2 in range(s1 + 1, spherePoses.shape[0]):
            sqrDistance = np.sum((spherePoses[s1, :] - spherePoses[s2, :]) ** 2)

            #explicitly finding out intersection between any two spheres
            if sqrDistance < 2 * sqrRadius:
                colNormal = spherePoses[s1, :] - spherePoses[s2, :]
                colNormal = colNormal / np.linalg.norm(colNormal)
                colDist = 2 * radius - np.sqrt(sqrDistance)
                #resolving interpenetration equally since masses and radii are equal
                spherePoses[s1, :] += colNormal * colDist / 2.0
                spherePoses[s2, :] -= colNormal * colDist / 2.0
                #hardcoding the linear velocity resolution (without inertia tensor) for mass = 1kg.
                velBefore = np.dot(sphereVelocities[s1, :] - sphereVelocities[s2, :], colNormal)
                jn = -(1+CRCoeff) * velBefore*colNormal
                sphereVelocities[s1, :] += jn
                sphereVelocities[s2, :] -= jn


        # Resolving collision between each sphere and the ground.
        if (spherePoses[s1, 1] < radius):
            spherePoses[s1, 1] = radius
            if (sphereVelocities[s1, 1] < 0.0):
                sphereVelocities[s1, 1] = - CRCoeff * sphereVelocities[s1, 1]
