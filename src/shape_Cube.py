from platonics import platonics_dict
from base_classes import Shape, Chart, Face

# atlas = {charts:[]}
# charts

class Cube(Shape):
    def __init__(self, num_faces=20, resolution=1):
        super(Cube, self).__init__(num_faces, resolution)
        self.resolution = resolution
        self.vertices = platonics_dict[self.num_faces]['3d_coords_dict']
        self.faces_dict = self.generate_faces()

    def generate_faces(self):
        faces_dict = {}
        for i in range(self.num_faces):
            faces_dict[i] = Face(self.resolution, self.vertices)


    def project_data(self):
        # code to project data onto each face
        pass

    def run_G_padding(self, dataset):
        # code to create each chart
        pass

