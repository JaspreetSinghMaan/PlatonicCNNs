from abc import ABC, abstractclassmethod


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

    @abstractclassmethod
    def get_faces_dict(resolution):
        pass


class Face(object):
    def __init__(self, resolution):
        self.resolution = resolution

    @abstractclassmethod
    def build_face_grid():
        pass

    def generate_faces(self):
        face_dict = {}
        return face_dict
