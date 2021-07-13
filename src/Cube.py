from platonics import platonics_dict
from base_classes import Shape, Chart, Face


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
