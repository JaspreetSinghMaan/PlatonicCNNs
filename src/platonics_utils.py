from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from platonics import platonics_dict

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
    # TODO: the '3d_coords' attribute does not exist anymore; replaced with '3d_vertex_coords'
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
