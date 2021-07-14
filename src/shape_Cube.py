from platonics import platonics_dict
from base_classes import Shape, Chart


# atlas = {charts:[]} #a dictionary to describe which faces form which charts within an atlas,
# also a place to store the rotational information

class Cube(Shape):
    def __init__(self):
        super(Cube, self).__init__(num_faces = 6)
        # self.num_faces = 6

