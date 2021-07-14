from platonics import platonics_dict
from base_classes import Shape
from utils import get_edge_pairs


class Icosahedron(Shape):
    def __init__(self):
        super(Icosahedron, self).__init__(num_faces=20)
        edges = get_edge_pairs(self.num_faces)

        self.atlas_dict = {
            'chart_faces': [[0], [1], [2], [3], [4], [5]],
            # a dictionary to describe which faces form which charts within an atlas, or # 'chart_faces':[[0,1,2,3,4,5]]
            'edges': edges,
            'transition_map': {}  # {#k f1:f1 #v 2x2 matrices in C4  #store the rotational information
        }
