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
        self.vertex_coords = platonics_dict[self.num_faces]['3d_coords_dict']
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
    def __init__(self, shape, resolution, ):
        self.shape = shape
        self.resolution = resolution
        self.face_cord, self.face_meta_data = self.generate_grid()

    def generate_grid(self):
        '''
        returns a nested dict
        {face_index:
            {grid_point_coordinates: n x n x 3,
             grid_point_meta information: n x n x 3}
        }
        :return:
        '''

    def generate_face_grid(self):
        '''
        generates the grid for one face
        :return: 1 tensor of n x n x 3 for the coordinates and
                1 tensor of n x n x 3 where (-1 = exterior, 0 = interior, 1 = vertex)
                1 tensor of n x n x 1 where 1 contains index for face
                dict for vertex k: v:
                dict for edge k: v:
        '''
        face_cords = None
        face_meta_data = None

        return face_cords, face_meta_data


class Atlas(ABC):
    '''
    Atlas takes the grid and defines and constructs each 2D chart
    Each individual platonic shape, will contain hard coded dictionary describing atlas and chart construction via face indices

    It remains to be decided if:
    the charts will be processed serparately in the batch dimension
    or if the atlas is to be processed in one large concated block
    '''
    def __init__(self, atlas_dict, grid):
        self.atlas_dict = atlas_dict
        self.grid = grid


    def generate_charts_dict(self):
        '''
        function to return dictionary of charts the construct the atlas
        or
        method that concatonates charts to form 2D atlas
        :return:
        '''


    def construct_chart(self):
        '''
        given an atlas_dict and 3D grid, method that constructs 2d charts
        :return:
        '''
        pass



class Chart(ABC):
    '''
    defines the set of points on the 3d face and their mapping to 2D set in the plane
    '''
    def __init__(self, grid, atlas):
        self.orientation = None
        self.gauge_transform = None
        self.exterior_points = {}
        self.interior_points = {}


class Signal(ABC):
    '''
    Description - class to project the signal from data to the grid and then to the 2d charts and plot in 3d for visualisation
    It's a torch.tensor object that is passed to the gauge_CNN in nthe forward pass
    '''
    def __init__(self, spherical_data, grid, charts):
        self.spherical_data = spherical_data
        self.grid = grid

    def transform_to_platonic(self):
        '''
        transforms spherical data to platonic data
        :return:
        '''
        pass

    def transform_to_atlas(self):
        '''
        transforms platonic data to large 2d data
        :return:
        '''
        pass

    def visualise(self):
        pass


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