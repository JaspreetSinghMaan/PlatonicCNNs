from abc import ABC, abstractclassmethod
import abc
import torch
from platonics import platonics_dict
from utils import find_3d_rotation, find_3d_frame_transform

class Shape(ABC):
    def __init__(self, num_faces, resolution):
        if num_faces not in [4, 6, 8, 12, 20]:
            raise ValueError("Not a Platonic number of faces!")

        if num_faces == 12 and resolution > 1:
            raise ValueError("Subdivision not currently implemented for Dodecahedron!")
        # elif num_faces in [4, 8, 20] and resolution % 2 != 1:
        #     raise ValueError("Must be an odd number of traingles per edge for these Platonics")

        self.num_faces = num_faces
        self.resolution = resolution
        self.faces = self.generate_faces()


    @abstractclassmethod
    def generate_atlas(self):
        pass


class Chart(ABC):
    def __init__(self, num_faces, resolution, vertices):
        self.num_faces = num_faces
        self.resolution = resolution
        self.vertices = vertices
        self.orientation = None
        self.gauge_transform = None
        self.exterior_points = {}
        self.interior_points = {}



    def get_gauge_transformation(self, face1, face2):
        # get indices for vertexes that describing x and y axis for both faces
        [[f1x1,f1x2], [f1y1, f1y2]] = platonics_dict[self.num_faces]['3d_face_frame_indx'][face1]
        [[f2x1,f2x2], [f2y1, f2y2]] = platonics_dict[self.num_faces]['3d_face_frame_indx'][face2]

        #use indices to look up 3d coords
        coords = platonics_dict[self.num_faces]['3d_face_vertex_coords_tensor']
        [[cf1x1,cf1x2], [cf1y1, cf1y2]] = [[torch.tensor(coords[f1x1],dtype=torch.float),torch.tensor(coords[f1x2],dtype=torch.float)], [torch.tensor(coords[f1y1],dtype=torch.float), torch.tensor(coords[f1y2],dtype=torch.float)]]
        [[cf2x1,cf2x2], [cf2y1, cf2y2]] = [[torch.tensor(coords[f2x1],dtype=torch.float),torch.tensor(coords[f2x2],dtype=torch.float)], [torch.tensor(coords[f2y1],dtype=torch.float), torch.tensor(coords[f2y2],dtype=torch.float)]]

        #use difference in coords to get f1ex, f1ey and f2ex f2ey,
        f1ex = cf1x2 - cf1x1
        f1ey = cf1y2 - cf1y1
        f2ex = cf2x2 - cf2x1
        f2ey = cf2y2 - cf2y1

        # then use those vectors to calculate the 3x3 matrix
        R = find_3d_frame_transform(f1ex, f2ex, f1ey, f2ey)

class Face(ABC):
    def __init__(self, resolution, vertices):
        self.resolution = resolution
        self.vertices = vertices
        self.orientation = None
        self.gauge_transform = None
        self.exterior_points = {}
        self.interior_points = {}

    @abstractclassmethod
    def build_big_face_triangles(self):
        pass

    @abstractclassmethod
    def build_small_triangle_grid(self, face):
        pass


# class Big_triangle(object):
#     def __init__(self, resolution, vertices):
#         self.resolution = resolution
#         self.vertices = vertices

class Points(object):
    def __init__(self, type, coords, value):
        self.type = type
        self.coords = coords
        self.value = value


