import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SR3 = np.sqrt(3)
SR2 = np.sqrt(2)
Phi = (1 + np.sqrt(5)) / 2  # 1.61803
phi = Phi - 1
# octahedron
a = 1 / (2 * np.sqrt(2))
b = 1 / 2
# icosahedron
c = 1 / 2
d = 1 / (2 * Phi)
# dodecahedron
e = 1 / Phi
f = 2 - Phi

# https://www.mathsisfun.com/geometry/model-construction-tips.html
# http://www.maths.surrey.ac.uk/hosted-sites/R.Knott/Fibonacci/phi3DGeom.html#section3
# http://paulbourke.net/geometry/platonic/ - used for 3d_coords_dict
#todo only working on Cube values for now, other shapes will need implementing subsequently
platonics_dict = {
    4: {
        'name': 'Tetrahedron',
        'num_faces': 4,
        '2d_num_edges': 9,
        '2d_nodes': range(6),
        '2d_coords': torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
                                   [0.5, SR3 / 2], [1.5, SR3 / 2],
                                   [1.0, SR3]]),
        '2d_edge_index': torch.tensor([[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
                                       [1, 3, 0, 3, 4, 2, 1, 4, 0, 1, 4, 5, 1, 2, 3, 5, 3, 4]]),
        '3d_num_edges': 4,
        '3d_nodes': range(4),
        '3d_coords': torch.tensor([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], dtype=torch.float),
        # '3d_coords': torch.tensor([(0, 0, 0), (, , 0), (, , 0), (SR3/2, SR3/2, 0.5)], dtype=torch.float),
        '3d_edge_index': torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]], dtype=torch.long),
        '3d_coords_dict': {
            0: [[1, 1, 1], [-1, 1, -1], [1, -1, -1]],
            1: [[-1, 1, -1], [-1 - 1, 1], [1 - 1 - 1]],
            2: [[1, 1, 1], [1, -1 - 1], [-1, -1, 1]],
            3: [[1, 1, 1], [-1, -1, 1], [-1, 1, -1]]}
    },

    6: {
        'name': 'Cube',
        'num_faces': 6,
        '2d_num_edges': 19,
        '2d_nodes': range(14),
        '2d_coords': torch.tensor([[1.0, 0.0], [2.0, 0.0],
                                   [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
                                   [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
                                   [1.0, 3.0], [2.0, 3.0],
                                   [1.0, 4.0], [2.0, 4.0]]),
        '2d_edge_index': torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11,
             12, 12, 13, 13],
            [1, 3, 0, 4, 3, 6, 0, 2, 4, 7, 1, 3, 5, 8, 4, 9, 2, 7, 3, 6, 8, 10, 4, 7, 9, 11, 5, 8, 7, 11, 12, 8, 10, 13,
             10, 13, 11, 12]]),
        '3d_num_edges': 12,
        '3d_nodes': range(8),
        '3d_edge_index': torch.tensor([  # edges described by corner node indices
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
            [1, 3, 4, 0, 2, 5, 1, 3, 6, 0, 2, 7, 0, 5, 7, 1, 4, 6, 2, 5, 7, 3, 4, 6]], dtype=torch.long),

        '3d_face_vertex_indx': {
            # a list of VERTEX indices for each face using 'radial' right hand thumb rule, starting for the lowest vertex index
            0: [0, 1, 2, 3],
            1: [0, 1, 4, 5],
            2: [1, 2, 6, 5],
            3: [2, 3, 7, 6],
            4: [0, 3, 7, 4],
            5: [4, 5, 6, 7]
        },

        '3d_face_adjacencies': {
            # a list adjacent of FACE indices for each face using 'radial' right hand thumb rule, starting for the lowest face index
            0: [1, 4, 3, 2],
            1: [0, 2, 4, 5],
            2: [0, 3, 5, 1],
            3: [0, 4, 5, 2],
            4: [0, 1, 5, 3],
            5: [1, 2, 3, 4]
        },

        '3d_vertex_coords': {  # 3d coordinates with centre mass at (0, 0, 0)
            0: [-1, -1, -1],
            1: [1, -1, -1],
            2: [1, 1, -1],
            3: [-1, 1, -1],
            4: [-1, -1, 1],
            5: [1, -1, 1],
            6: [1, 1, 1],
            7: [-1, 1, 1]
        },

        # - Determine for each pair of overlapping charts, how directions are changed when going from one chart to the other. In other words, determine the transition map.
        '3d_face_frame_indx': {  # direction of x-axis and y-axis for each face use vertex indices
            0: [[3, 2], [3, 0]],
            1: [[0, 1], [0, 4]],
            2: [[1, 2], [1, 5]],
            3: [[2, 3], [2, 6]],
            4: [[3, 0], [3, 7]],
            5: [[4, 5], [4, 7]]
        },

        '3d_face_vertex_coords': {  # coordinates of each node defining a face
            0: [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]],
            1: [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]],
            2: [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            3: [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
            4: [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
            5: [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]]
        },

        'group_rotation': {
            # dictionary that describes the rotation needed to map frame on f1 to frame on f2, and also vertexes on f1 to points on f2
            'f1_idx:f2_idx': torch.tensor([[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 1, 0]], dtype=torch.float)
        }
    },

    8: {
        'name': 'Octahedron',
        'num_faces': 8,
        '2d_num_edges': 17,
        '2d_nodes': range(10),
        '2d_coords': torch.tensor([[SR3, 0.0],
                                   [SR3 / 2, SR3 / 2],
                                   [SR3, SR3],
                                   [SR3 / 2, 3 * SR3 / 2], [3 * SR3 / 2, 3 * SR3 / 2],
                                   [0, 2 * SR3], [SR3, 2 * SR3],
                                   [SR3 / 2, 5 * SR3 / 2],
                                   [SR3, 3 * SR3],
                                   [SR3 / 2, 7 * SR3 / 2]]),
        '2d_edge_index': torch.tensor([
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9],
            [1, 2, 0, 2, 3, 0, 1, 3, 4, 6, 1, 2, 5, 6, 7, 2, 6, 3, 7, 2, 3, 4, 7, 8, 3, 5, 6, 8, 9, 6, 7, 9, 7, 8]]),
        '3d_num_edges': 12,
        '3d_nodes': range(6),
        '3d_coords': torch.tensor([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
                                  dtype=torch.float),
        '3d_coords_dict': {
            0: [[-a, 0, a], [-a, 0, -a], [0, b, 0]],
            1: [[-a, 0, -a], [a, 0, -a], [0, b, 0]],
            2: [[a, 0, -a], [a, 0, a], [0, b, 0]],
            3: [[a, 0, a], [-a, 0, a], [0, b, 0]],
            4: [[a, 0, -a], [-a, 0, -a], [0, -b, 0]],
            5: [[-a, 0, -a], [-a, 0, a], [0, -b, 0]],
            6: [[a, 0, a], [a, 0, -a], [0, -b, 0]],
            7: [[-a, 0, a], [a, 0, a], [0, -b, 0]]}
    },

    # todo need Dodecahedron 2d stats
    # 12: {
    #     'name': 'Dodecahedron',
    #     'num_faces': 12,
    #     # '2d_num_edges': ,
    #     # '2d_nodes': range(),
    #     # '2d_coords': torch.tensor([
    #     #            ]),
    #     '3d_num_edges': 30,
    #     '3d_nodes': range(20),
    #     '3d_coords': torch.tensor([(0, phi, Phi), (0, phi, -Phi), (0, -phi, Phi), (0, -phi, -Phi),
    #                                 (Phi, 0, phi), (Phi, 0, -phi), (-Phi, 0, phi), (-Phi, 0, -phi),
    #                                 (phi, Phi, 0), (phi, -Phi, 0), (-phi, Phi, 0), (-phi, -Phi, 0),
    #                                 (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    #                                 (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)],
    #                               dtype=torch.float),
    #
    #     '2d_edge_index': torch.tensor([
    #         [],
    #         []])},

    #  f  0  1   -f  0  1   -e  e  e    0  1  f    e  e  e
    # -f  0  1    f  0  1    e -e  e    0 -1  f   -e -e  e
    #  f  0 -1   -f  0 -1   -e -e -e    0 -1 -f    e -e -e
    # -f  0 -1    f  0 -1    e  e -e    0  1 -f   -e  e -e
    #  0  1 -f    0  1  f    e  e  e    1  f  0    e  e -e
    #  0  1  f    0  1 -f   -e  e -e   -1  f  0   -e  e  e
    #  0 -1 -f    0 -1  f   -e -e  e   -1 -f  0   -e -e -e
    #  0 -1  f    0 -1 -f    e -e -e    1 -f  0    e -e  e
    #  1  f  0    1 -f  0    e -e  e    f  0  1    e  e  e
    #  1 -f  0    1  f  0    e  e -e    f  0 -1    e -e -e
    # -1  f  0   -1 -f  0   -e -e -e   -f  0 -1   -e  e -e
    # -1 -f  0   -1  f  0   -e  e  e   -f  0  1   -e -e  e

    20: {
        'name': 'Icosahedron',
        'num_faces': 20,
        '2d_num_edges': 41,
        '2d_nodes': range(22),
        '2d_coords': torch.tensor([[SR3, 0.0],
                                   [SR3 / 2, SR3 / 2], [3 * SR3 / 2, SR3 / 2],
                                   [0, SR3], [SR3, SR3],
                                   [SR3 / 2, 3 * SR3 / 2], [3 * SR3 / 2, 3 * SR3 / 2],
                                   [0, 2 * SR3], [SR3, 2 * SR3],
                                   [SR3 / 2, 5 * SR3 / 2], [3 * SR3 / 2, 5 * SR3 / 2],
                                   [0, 3 * SR3], [SR3, 3 * SR3],
                                   [SR3 / 2, 7 * SR3 / 2], [3 * SR3 / 2, 7 * SR3 / 2],
                                   [0, 4 * SR3], [SR3, 4 * SR3],
                                   [SR3 / 2, 9 * SR3 / 2], [3 * SR3 / 2, 9 * SR3 / 2],
                                   [0, 5 * SR3], [SR3, 5 * SR3],
                                   [SR3 / 2, 11 * SR3 / 2]
                                   ]),
        '2d_edge_index': torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9,
             9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 16,
             16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21],
            [1, 2, 4, 0, 3, 4, 5, 0, 4, 1, 5, 0, 1, 2, 5, 6, 8, 1, 3, 4, 7, 8, 9, 4, 8, 5, 9, 4, 5, 6, 9, 10, 12, 5, 7,
             8, 11, 12, 13, 8, 12, 9, 13, 8, 9, 10, 13, 14, 16, 9, 11, 12, 15, 16, 17, 12, 16, 13, 17, 12, 13, 14, 17,
             18, 20, 13, 15, 16, 19, 20, 21, 16, 20, 17, 21, 16, 17, 18, 21, 17, 19, 20]]),
        '3d_num_edges': 30,
        '3d_nodes': range(12),
        '3d_coords': torch.tensor([(0, Phi, 1), (0, Phi, -1), (0, -Phi, 1), (0, -Phi, -1),
                                   (1, 0, Phi), (1, 0, -Phi), (-1, 0, Phi), (-1, 0, -Phi),
                                   (Phi, 1, 0), (Phi, -1, 0), (-Phi, 1, 0), (-Phi, -1, 0)],
                                  dtype=torch.float),
        '3d_coords_dict': {
            0: [[0, d, -c], [d, c, 0], [-d, c, 0]],
            1: [[0, d, c], [-d, c, 0], [d, c, 0]],
            2: [[0, d, c], [0, -d, c], [-c, 0, d]],
            3: [[0, d, c], [c, 0, d], [0, -d, c]],
            4: [[0, d, -c], [0, -d, -c], [c, 0, -d]],
            5: [[0, d, -c], [-c, 0, -d], [0, -d, -c]],
            6: [[0, -d, c], [d, -c, 0], [-d, -c, 0]],
            7: [[0, -d, -c], [-d, -c, 0], [d, -c, 0]],
            8: [[d, c, 0], [-c, 0, d], [-c, 0, -d]],
            9: [[d, -c, 0], [-c, 0, -d], [-c, 0, d]],
            10: [[d, c, 0], [c, 0, -d], [c, 0, d]],
            11: [[d, -c, 0], [c, 0, d], [c, 0, -d]],
            12: [[0, d, c], [-c, 0, d], [-d, c, 0]],
            13: [[0, d, c], [d, c, 0], [c, 0, d]],
            14: [[0, d, -c], [-d, c, 0], [-c, 0, -d]],
            15: [[0, d, -c], [c, 0, -d], [d, c, 0]],
            16: [[0, -d, -c], [-c, 0, -d], [-d, -c, 0]],
            17: [[0, -d, -c], [d, -c, 0], [c, 0, -d]],
            18: [[0, -d, c], [-d, -c, 0], [-c, 0, d]],
            19: [[0, -d, c], [c, 0, d], [d, -c, 0]]}
    }
}


def plot_2d(num_faces):
    shape_dict = platonics_dict[num_faces]
    fig, ax = plt.subplots()
    graph = Data(y=shape_dict['2d_nodes'], edge_index=shape_dict['2d_edge_index'], pos=shape_dict['2d_coords'])
    G = to_networkx(graph)
    node_pos = {node: pos.tolist() for node, pos in zip(shape_dict['2d_nodes'], shape_dict['2d_coords'])}
    labels = {node: node for node in shape_dict['2d_nodes']}
    nx.draw(G, pos=node_pos, labels=labels)
    plt.show()


def plot_3d(num_faces):
    # https://stackoverflow.com/questions/65752590/converting-a-networkx-2d-graph-into-a-3d-interactive-graph
    shape_dict = platonics_dict[num_faces]
    graph = Data(y=shape_dict['3d_nodes'], edge_index=shape_dict['3d_edge_index'], pos=shape_dict['3d_coords'])
    G = to_networkx(graph)
    edges = G.edges()

    # ## update to 3d dimension
    spring_3D = nx.spring_layout(G, dim=3, k=0.5)  # k regulates the distance between nodes

    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: spring_3D is a dictionary where the keys are 1,...,6
    x_nodes = [spring_3D[key][0] for key in spring_3D.keys()]  # x-coordinates of nodes
    y_nodes = [spring_3D[key][1] for key in spring_3D.keys()]  # y-coordinates
    z_nodes = [spring_3D[key][2] for key in spring_3D.keys()]  # z-coordinates

    # we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # create lists holding midpoints that we will use to anchor text
    xtp = []
    ytp = []
    ztp = []

    # need to fill these with all of the coordinates
    for edge in edges:
        # format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0], spring_3D[edge[1]][0], None]
        x_edges += x_coords
        xtp.append(0.5 * (spring_3D[edge[0]][0] + spring_3D[edge[1]][0]))

        y_coords = [spring_3D[edge[0]][1], spring_3D[edge[1]][1], None]
        y_edges += y_coords
        ytp.append(0.5 * (spring_3D[edge[0]][1] + spring_3D[edge[1]][1]))

        z_coords = [spring_3D[edge[0]][2], spring_3D[edge[1]][2], None]
        z_edges += z_coords
        ztp.append(0.5 * (spring_3D[edge[0]][2] + spring_3D[edge[1]][2]))

    # etext = [f'weight={w}' for w in edge_weights]

    trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
                                 mode='markers',
                                 marker=dict(color='rgb(125,125,125)', size=1))  # ,
    # set the same color as for the edge lines
    # text=etext, hoverinfo='text')

    # create a trace for the edges
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')

    # create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(symbol='circle',
                    size=10,
                    color='skyblue')
    )

    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_weights]
    fig = go.Figure(data=data)

    fig.show()


# https://networkx.org/documentation/stable/auto_examples/3d_drawing/mayavi2_spring.html#sphx-glr-auto-examples-3d-drawing-mayavi2-spring-py
# pip install mayavi
# pip install pyqt5
# from mayavi import mlab
def plot_3d_ii(num_faces):
    shape_dict = platonics_dict[num_faces]
    graph = Data(y=shape_dict['3d_nodes'], edge_index=shape_dict['2d_edge_index'], pos=shape_dict['3d_coords'])
    G = to_networkx(graph)
    # # some graphs to try
    # # H=nx.krackhardt_kite_graph()
    # # H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
    # # H=nx.grid_2d_graph(4,5)
    # H = nx.cycle_graph(20)
    #
    # # reorder nodes from 0,len(G)-1
    # G = nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars = np.array(list(G.nodes())) + 5

    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()


if __name__ == "__main__":
    # plot_2d(6)
    # plot_3d(6)
    for num_faces in [4, 6, 8, 20]:
        plot_2d(num_faces)
        # plot_3d(num_faces)
