import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def rand_positions(room_dim, n_pos):
    positions = []
    for i in range(n_pos):
        factor = np.random.sample(3)
        
        x_pos = room_dim[0]*factor[0]
        y_pos = room_dim[1]*factor[1]
        z_pos = room_dim[2]*factor[2]
        
        positions.append([x_pos, y_pos, z_pos])
    
    return positions

def plot_room(room, source, receivers, title):

    r_points = np.array(receivers)

    cube_definition = [(0,0,0),(0,room[1],0),(room[0],0,0),(0,0,room[2])] 

    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('X [m]', fontsize=18)
    ax.set_ylabel('Y [m]', fontsize=18)
    ax.set_zlabel('Z [m]', fontsize=18)

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.scatter(r_points[:,0], r_points[:,1], r_points[:,2],  s=50, c='b', label = 'Receptor')
    ax.scatter(source[0], source[1], source[2], s=30, c='r', label = 'Fuente')
    ax.legend(fontsize=18)
    ax.set_aspect('auto')
    return 

def norm_to_save(impulso):
    impulso_norm = impulso / np.max(abs(impulso))
    init = np.argmax(impulso_norm)
    impulso_crop = impulso_norm[init:]
    return impulso_crop
