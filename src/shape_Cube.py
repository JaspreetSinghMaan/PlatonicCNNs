from platonics import platonics_dict
from base_classes import Shape
from utils import get_edge_pairs


class Cube(Shape):
    def __init__(self):
        super(Cube, self).__init__(num_faces=6)
        edges = get_edge_pairs(self.num_faces)

        self.atlas_dict = {
            'chart_faces': [[0], [1], [2], [3], [4], [5]],
            # a dictionary to describe which faces form which charts within an atlas, or # 'chart_faces':[[0,1,2,3,4,5]]
            'edges': edges,
            'transition_map': {}  # {#k f1:f1 #v 2x2 matrices in C4  #store the rotational information
        }
        # '3d_face_adjacencies'  #subset of the 24 edges where two charts share an edge. (in the case where each chart is a face it's actually the full 24 edges)
        # #apply for all 24 edges, noting many times this will be the 2x2 identity matrix
        # #identify edges where theres a gauge transforamtion
        # 'transition_map':# {#k f1:f1 #v 2x2 matrices in C4  #store the rotational information
        # }
