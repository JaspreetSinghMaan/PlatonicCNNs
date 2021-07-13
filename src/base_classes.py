from abc import ABC, abstractclassmethod
import abc
from platonics import platonics_dict

class Shape(ABC):
    def __init__(self, num_faces, resolution):

        if num_faces not in [4, 6, 8, 12, 20]:
            raise ValueError("Not a Platonic number of faces!")

        if num_faces == 12 and resolution > 1:
            raise ValueError("Subdivision not currently implemented for Dodecahedron!")
        elif num_faces in [4, 8, 20] and resolution % 2 != 1:
            raise ValueError("Must be an odd number of traingles per edge for these Platonics")

        self.num_faces = num_faces
        self.resolution = resolution
        self.faces = self.generate_faces()

    @abstractclassmethod
    def generate_faces(self):
        pass


class Charts(ABC):
    def __init__(self, resolution, vertices):
        self.resolution = resolution
        self.vertices = vertices
        self.orientation = None
        self.gauge_transform = None
        self.exterior_points = {}
        self.interior_points = {}

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


class Big_triangle(object):
    def __init__(self, resolution, vertices):
        self.resolution = resolution
        self.vertices = vertices

class Points(object):
    def __init__(self, type, coords, value):
        self.type = type
        self.coords = coords
        self.value = value


class Icosaheadron(Shape):
    def __init__(self, num_faces=20, resolution=1):
        super(Shape, self).__init__(num_faces, resolution)
        self.resolution = resolution
        self.vertices = platonics_dict[self.num_faces]['3d_coords_dict']
        self.faces_dict = self.generate_faces()

    def generate_faces(self):
        faces_dict = {}
        for i in range(self.num_faces):
            faces_dict[i] = Face(self.resolution, self.vertices)