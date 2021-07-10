import numpy as np
import torch
from base_classes import Shape
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


class Icosaheadron(Shape):
    def __init__(self, num_faces=20, resolution=1):
        super(Shape, self).__init__(num_faces, resolution)

        self.resolution = resolution
        self.faces = self.generate_faces()


SR3 = np.sqrt(3)
platonics = {'Tetrahedron': {
    'nodes': range(6),
    'coords': [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
               [0.5, SR3 / 2], [1.5, SR3 / 2],
               [1.0, SR3]],
    'edge_index': [[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
                   [1, 3, 0, 3, 4, 2, 1, 4, 0, 1, 4, 5, 1, 2, 3, 5, 3, 4]]},
    'Cube': {
        'nodes': range(14),
        'coords': [[1.0, 0.0], [2.0, 0.0],
                   [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0],
                   [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0],
                   [1.0, 3.0], [2.0, 3.0],
                   [1.0, 4.0], [2.0, 4.0]],
        'edge_index': [
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11,
             12, 12, 13, 13],
            [1, 3, 0, 4, 3, 6, 0, 2, 4, 7, 1, 3, 5, 8, 4, 9, 2, 7, 3, 6, 8, 10, 4, 7, 9, 11, 5, 8, 7, 11, 12, 8, 10, 13,
             10, 13, 11, 12]]
    },
    'Octahedron': {
        'nodes': range(10),
        'coords': [[SR3, 0.0],
                   [SR3 / 2, SR3 / 2],
                   [SR3, SR3],
                   [SR3 / 2, 3 * SR3 / 2], [3 * SR3 / 2, 3 * SR3 / 2],
                   [0, 2 * SR3], [SR3, 2 * SR3],
                   [SR3 / 2, 5 * SR3 / 2],
                   [SR3, 3 * SR3],
                   [SR3 / 2, 7 * SR3 / 2]],
        'edge_index': [
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9],
            [1, 2, 0, 2, 3, 0, 1, 3, 4, 6, 1, 2, 5, 6, 7, 2, 6, 3, 7, 2, 3, 4, 7, 8, 3, 5, 6, 8, 9, 6, 7, 9, 7, 8]]},
    'Icosahedron': {
        'nodes': range(22),
        'coords': [[SR3, 0.0],
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
                   ],

        'edge_index': [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9,
             9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 16,
             16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21],
            [1, 2, 4, 0, 3, 4, 5, 0, 4, 1, 5, 0, 1, 2, 5, 6, 8, 1, 3, 4, 7, 8, 9, 4, 8, 5, 9, 4, 5, 6, 9, 10, 12, 5, 7,
             8, 11, 12, 13, 8, 12, 9, 13, 8, 9, 10, 13, 14, 16, 9, 11, 12, 15, 16, 17, 12, 16, 13, 17, 12, 13, 14, 17,
             18, 20, 13, 15, 16, 19, 20, 21, 16, 20, 17, 21, 16, 17, 18, 21, 17, 19, 20]]}
}

if __name__ == "__main__":
    for shape, values in platonics.items():
        fig, ax = plt.subplots()
        graph = Data(y=torch.tensor(values['nodes']), edge_index=torch.tensor(values['edge_index']),
                     pos=torch.tensor(values['coords']))
        G = to_networkx(graph)
        node_pos = {node: pos for node, pos in zip(values['nodes'], values['coords'])}
        labels = {node: node for node, pos in zip(values['nodes'], values['coords'])}
        nx.draw(G, pos=node_pos, labels=labels)
        plt.show()
