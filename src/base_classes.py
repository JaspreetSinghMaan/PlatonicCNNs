from abc import ABC, abstractclassmethod
import abc
import torch
from platonics import platonics_dict
from utils import find_3d_rotation, find_3d_frame_transform
import torch.nn as nn

class Shape(ABC):
    '''
    Description - generic class for platonic shape object
    '''
    def __init__(self, num_faces):
        if num_faces not in [4, 6, 8, 12, 20]:
            raise ValueError("Not a Platonic number of faces!")

        self.num_faces = num_faces
        self.name = platonics_dict[self.num_faces]['name']
        self.num_edges = platonics_dict[self.num_faces]['3d_num_edges']
        self.nodes_index = platonics_dict[self.num_faces]['3d_nodes']
        self.edge_index = platonics_dict[self.num_faces]['3d_edge_index']
        self.face_vertex_indx = platonics_dict[self.num_faces]['3d_face_vertex_indx']
        self.face_adjacencies = platonics_dict[self.num_faces]['3d_face_adjacencies']
        self.vertex_coords = platonics_dict[self.num_faces]['3d_vertex_coords'] #3d coordinates for each vertex
        self.face_frame_indx = platonics_dict[self.num_faces]['3d_face_frame_indx']
        self.face_vertex_coords = platonics_dict[self.num_faces]['3d_face_vertex_coords'] #3d coordinates for vertex defining a face, can be composed using '3d_face_vertex_indx' and '3d_vertex_coords'
        self.group_rotation = platonics_dict[self.num_faces]['group_rotation']


class Grid(ABC):
    '''
    an object that constructs and stores the 3d points and meta-information about a discretisation of a given platonic solid
    '''
    def __init__(self, shape, resolution):
        self.shape = shape
        self.resolution = resolution
        self.grid_dict = self.generate_grid()

    def generate_grid(self):
        '''
        returns a nested dict
        {face_index:
            {grid_point_coordinates: n x n x 3,
             grid_point_meta information: n x n x 3}
        }
        :return:
        '''
        grid_dict = {}
        for face in range(self.shape.num_faces):
            grid_dict[face] = self.generate_face_grid()
        return grid_dict

    def generate_face_grid(self):
        '''
        generates the grid for one face
        :return: 1 tensor of n x n x 3 for the coordinates and
                1 tensor of n x n x 3 where (-1 = edge, 0 = interior, 1 = vertex)     ----potentially -2 for padding
                1 tensor of n x n x 1 where 1 contains index for face
                dict for vertex k: v:
                dict for edge k: v:
        '''
        face_cords = None
        face_meta_data = None

        return face_cords, face_meta_data


class Atlas(ABC):
    '''
    Atlas takes the Grid and defines and constructs each Chart
    Each individual platonic shape, will contain hard coded dictionary (atlas_dict) describing Atlas and Chart construction via Face indices

    It remains to be decided if:
    the charts will be processed serparately in the batch dimension
    or if the atlas is to be processed in one large concated block
    '''
    def __init__(self, shape, grid):
        self.shape = shape
        self.atlas_dict = shape.atlas_dict
        self.grid = grid
        self.charts_dict = self.generate_charts_dict()

    def generate_charts_dict(self):
        '''
        function to return dictionary of charts the construct the atlas
        or
        method that concatonates charts to form 2D atlas
        :return:
        '''
        charts_dict = {}
        for chart_faces in self.atlas_dict:
            for chart_num in chart_faces:
                charts_dict[chart_num] = Chart(self.grid, chart_faces, self.atlas_dict)
        return charts_dict


class Chart(ABC):
    '''
    defines the set of points on the 3d face and their mapping to 2D set in the plane
    '''
    def __init__(self, grid, chart_faces, atlas_dict):
        self.grid = grid
        self.chart_faces = chart_faces
        self.atlas_dict = atlas_dict

    def construct_chart(self):
        '''
        given an atlas_dict and 3D grid, method that constructs 2d charts
        :return:
        '''
        pass

    def chart_bij(self):
        '''
        maps point on the cube to point on the plane
        '''
        # 3d cordinates + meta inforamtion
        face_indx = 0
        face_cords, face_meta_data = self.grid.grid_dict[face_indx]

    def chart_bij_inv(self):
        '''
        maps a point on the plane to corresponding point on the cube
        :return:
        '''
        pass


class Signal(ABC):
    '''
    Description - class to project the signal from data to the grid and then to the 2d charts and plot in 3d for visualisation
    It's a torch.tensor object that is passed to the gauge_CNN in nthe forward pass
    '''
    def __init__(self, spherical_data, grid, atlas):
        self.spherical_data = spherical_data
        self.grid = grid
        self.platonic_data = self.transform_to_platonic()


    def transform_to_platonic(self):
        '''
        transforms spherical data to platonic data
        it's expected spherical data will be ((x,y,z),(r,g,b))
        :return: platonic data will be of the same format ((x,y,z),(r,g,b))
        '''
        # this is done via projection of a the grid point coordinates to the sphere
        # find the closest pixel value to this point #Warning comutational complexity here!!
        # and then recording the coordinates and pixel value for each point in the grid
        self.spherical_data
        platonic_data = None
        return platonic_data

    def visualise_platonic(self):
        '''
        here we visualise the spherical data projected onto the platonic shape
        use mayavi or matplotlib 3d
        :return:
        '''
        pass


    def transform_to_atlas(self, platonic_data, atlas):
        '''
        for each chart
        transforms 3D platonic data of form ((x,y,z),(r,g,b))
        to large 2d data arrays of form (height,width,(rbg)) ie a tensor with values but no explicit coordinates
        :return: dict{chart number: 2d array}
        '''
        twoD_array_dict = {}

        for chart_num, chart in atlas.charts_dict.items():
            twoD_array_dict[chart_num] = chart.chart_bij_inv()


    def atlas_to_tensor(self):
        '''
        takes the twoD_array_dict of 2d charts and converts to torch tensor ready to pass through model
        :return:
        '''


    def apply_group_sym_action(self, rho):
        '''
        for each chart
        Implement a function that applies a symmetry transformation to a signal stored in a rectangular array.
        This is done by mapping the integer pixel coordinates in the array to pixel positions on the platonic solid,
        rotating those, and mapping the transformed pixels back to the plane using an inverse chart.
        :return: 2d data array of form (height,width,(rbg))
        '''



class gauge_CNN(nn.Module):
    '''
    base class for gauge_CNN model could be S2S, S2R, R2R
    performs forward pass on the signal object

    Methods required:
    S2S:    G-padding - trivial as scalar signal
            N/A Filter expansion

    S2R:    G-padding
            Filter expansion
            Max-pooling required for chaining

    R2R:    G-padding
            Filter expansion
    '''
    def __init__(self, opt):
        self.opt = opt  #arguments for model