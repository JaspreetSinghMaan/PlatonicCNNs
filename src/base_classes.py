from abc import ABC, abstractclassmethod
import abc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from charts import chart, chart_inverse

from utils import find_3d_rotation, find_3d_frame_transform
from platonics import platonics_dict


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
        self.face_ignore_dim = platonics_dict[self.num_faces]['3d_face_ignore_dim'] # used in grids: axis to be ignored during per-face grid generation
        self.group_rotation = platonics_dict[self.num_faces]['group_rotation']


class Grid(ABC):
    '''
    an object that constructs and stores the 3d points and meta-information about a discretisation of a given platonic solid
    '''
    def __init__(self, shape, resolution):
        self.shape = shape
        
        self.resolution = resolution  # sets how many times we recursively sub-divide, resolution = 1 --> no sub-division

        # determine number of points in a 2D grid at given resolution
        self.npoints = self.resolution_to_npoints(resolution)
        # for Cube, total number of points in the square grid per face will be (npoints x npoints)
        
        self.grid_dict = self.generate_grid()

    def generate_grid(self):
        '''
        returns a tuple of arrays of form: (grid_point_coordinates, grid_point_meta) =

        NOT # where N is num_faces * num_nodes in a face (this includes duplication)

        N is |G|
        (N x 3), grid_point_coordinates

        for each face
        (N_f,1) - node_list = indexing values of grid_point_coordinates
        (N_f,1), grid_point_meta where type in [-1,0,1],

        grid_pad list: index value of nodes that are in the padding to that face
        maybe list of face adjacencies - using (3d_face_adjacencies from platonics_dict

        Not this:
        padding dict of list of indexes of format {face_num:
                                                interior: [list of indexes of pixels on interior]
                                                vertex: [list of indexes of pixels on vertexes]
                                                edges: [list of indexes of pixels on edges not on vertexes]}}

        :return:
        '''
        grid_dict = {}
        for face in range(self.shape.num_faces):
            grid_dict[face] = self.generate_face_grid(face)
        return grid_dict

    def generate_face_grid(self, face):
        '''
        generates the grid for one face
        :return: 1 tensor of n x n x 3 for the coordinates and
                1 tensor of n x n x 1 where 1 contains (-1 = edge, 0 = interior, 1 = vertex)     ----potentially -2 for padding
                TODO 1 tensor of n x n x 1 where 1 contains index for face
                TODO dict for vertex k: v:
                TODO dict for edge k: v:
        '''
        # get 3D coordinates of end points for the given face
        end_coords = self.shape.face_vertex_coords[face]
        # end_coords array must be of length 4.
        assert len(end_coords) == 4
        
        # get dimension to be ignored during grid construction
        ignore_dim = self.shape.face_ignore_dim[face]
        assert ignore_dim in [0, 1, 2] # (x, y, z)

        # check that all end points in the ignore_dim dimension have the same value
        assert end_coords[0][ignore_dim] == end_coords[1][ignore_dim] == end_coords[2][ignore_dim] == end_coords[3][ignore_dim]

        # if resolution is 1, just return the end_coords list
        if self.resolution <= 1:
            return end_coords, np.ones(end_coords.shape)

        ###########################################################################
        # we will create the grid in 2D and then add back the ignored 3D dimension
        ###########################################################################
        
        # create 2D grid
        ###############################################################################################
        # TODO currently, 2D grid generation only supports the Cube, which uses square grids
        # square grids can be generated super efficiently via numpy's linspace + meshgrid functions
        # linspace -- generates evenly spaces numbers on a 1D line between two end points
        # meshgrid -- efficient function to create a 'grid' from the results of two linspaces 
        #             (one for the x coords, one for the y coords)
        # TODO we hardcode the end points of the linspaces for the x coords and y coords to be from 
        # -1 to 1 at the moment; if we implement trinagular grids, we will have to determine end points
        ################################################################################################
        xx, yy = np.meshgrid( np.linspace(-1, 1, self.npoints), np.linspace(-1, 1, self.npoints) )
        
        # create metadata associated with each pixel on the grid
        # (-1 = edge, 0 = interior, 1 = vertex)
        meta = np.zeros(xx.shape)
        # edges --> (x in {1, -1}) OR (y in {1, -1})
        meta[xx == 1] = 1; meta[xx == -1] = 1
        meta[yy == 1] = 1; meta[yy == -1] = 1
        # vertices --> (x in {1, -1}) AND (y in {1, -1})
        # TODO in this implementation, we can leverage the fact that linspace ensures that the first and final entry 
        # in each row of the meshgrid are the vertices/end points
        meta[(0,0,-1,-1), (0,-1,0,-1)] = -1
        
        # add back the ignored 3d dimensions
        pad = np.ones(xx.shape) * end_coords[0][ignore_dim]
        if ignore_dim == 0:
            return np.stack((pad, xx, yy), axis=2), meta
        elif ignore_dim == 1:
            return np.stack((xx, pad, yy), axis=2), meta
        elif ignore_dim == 2:
            return np.stack((xx, yy, pad), axis=2), meta

    def resolution_to_npoints(self, resolution):
        """
        Helper method to recursively calculate the number of points that a line can be divided into at a given recursion resolution.

        Currently only supports Square grids/Cube as the shape.
        """
        if self.shape.num_faces == 6:
            npoints = [2, 3, 5, 9, 17]  # beyond resolution 5, we can compute the number of points recursively

            if resolution <= 5:
                return npoints[resolution-1]
            else:
                n = npoints[-1]
                for _ in range(5, resolution):
                    n = n* 2 - 1
                return n
        else:
            raise NotImplementedError

    def viz_face_grid(self, face):
        grid_point_coordinates = self.grid_dict[face][0]
        ignore_dim = self.shape.face_ignore_dim[face]
        assert ignore_dim in [0, 1, 2] # (x, y, z)

        xx, yy, zz = np.moveaxis(grid_point_coordinates, source=2, destination=0)
        if ignore_dim == 0:
            plt.plot(yy, zz, marker='.', color='k', linestyle='none')
        elif ignore_dim == 1:
            plt.plot(xx, zz, marker='.', color='k', linestyle='none')
        elif ignore_dim == 2:
            plt.plot(xx, yy, marker='.', color='k', linestyle='none')
        plt.show()


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
        for chart_num, chart_nums in enumerate(self.atlas_dict['chart_faces']):
            for face_num in chart_nums:
                charts_dict[chart_num] = Chart(self.grid.grid_dict[chart_num], chart_nums, self.atlas_dict)
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

    def chart_bij(self, i,r,x): #chart(i,r,x)
        '''
        maps point on the cube to point on the plane
        '''
        # 3d cordinates + meta inforamtion
        # face_indx = 0
        # face_cords, face_meta_data = self.grid.grid_dict[face_indx]
        return chart(i,r,x)

    def chart_bij_inv(self): #chart_inverse(i,r,x)
        '''
        maps a point on the plane to corresponding point on the cube
        :return:
        '''
        pass


class Signal(ABC):
    '''
    Description - class to project the signal from data to the grid and then to the 2d charts and plot in 3d for visualisation
    It's a torch.tensor object that is passed to the gauge_CNN in the forward pass
    '''
    def __init__(self, spherical_data, grid, atlas):
        self.spherical_data = spherical_data
        self.grid = grid
      
        
        # JGP TODO: How is this spherical data *actually* stored?
        # Here we just need the **locations** since the correspondence only depends on the two grids
        # and not on the values.  
        # Assuming grid to be a tensor of #grid_pts x 3
        # JGP TODO: we can probably move the projection code in here
        from project_data import project_to_solid
        
        # self.solid_projection_corresp = project_to_solid(self.spherical_data.LOCATIONS??, self.grid)

        # JGP: We probably don't want to store actual 'signals' like this, but rather as big tensors for vectorized computations
        self.platonic_data = self.project_to_solid()


    def project_to_solid(self):
        '''
        transforms spherical data to platonic data
        it's expected spherical data will be ((x,y,z),(r,g,b))
        :return: platonic data will be of the same format ((x,y,z),(r,g,b))
        '''
        # JGP TODO: How is this spherical data *actually* stored?
        # Just need to index the values with whichever point in sphere corresponded to each grid point 
        
        # return self.spherical_data.VALUES??[self.solid_projection_corresp]

    def visualise_solid(self):
        '''
        here we visualise the spherical data projected onto the platonic shape
        use mayavi or matplotlib 3d
        :return:
        '''
        # JGP: some initial code exists in platonics_utils and project_data
        pass


    def transform_to_atlas(self, platonic_data, atlas):
        '''
        for each chart
        transforms 3D platonic data of form ((x,y,z),(r,g,b))
        to large 2d data arrays of form (height,width,(rbg)) ie a tensor with values but no explicit coordinates
        :return: dict{chart number: 3d array of (3,H,W) }
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