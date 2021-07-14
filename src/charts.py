import numpy as np
# import torch
# from base_classes import Shape
# from torch_geometric.data import Data
# from torch_geometric.utils.convert import to_networkx
# import networkx as nx
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# dictionaries of symmetries bringing face-frame i to face-frame 0
symm_face_frame_i_to_0={
    1: np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0]
    ])
    # TO BE ADDED add other symmetries
}

# dictionaries of symmetries bringing face-frame 0 to face-frame i
symm_face_frame_0_to_i={
    1: np.array([
        [1,0,0],
        [0,0,1],
        [0,-1,0]
    ])
    # TO BE ADDED add other symmetries
}

import numpy as np

def chart(i,r,x):
    # chart brings points on the cube to Z^2
    # i is an index between 0 and 5, denoting the corresponding face index
    # r is the resolution factor
    # x is the array for which we compute the image via the chart
    # we only define explicitely a chart for face 0, and use the action of the group of
    # symmetries of the cube to define other charts coherently
    if i==0:
        # bottom face
        if x[2]==-1: # if x belongs to the bottom face
            return np.array([
                (2**(r-1))*(x[0]+1+(1/2**r)),
                (2**(r-1))*(-x[1]+1+(1/2**r))
                ])
            # flips y, translate, dilatate (padding space is left)
        else: # x is an extra-pixel (x[2]>-1):
            if x[1]==-1: # front face
                return np.array([\
                    (2**(r-1))*(x[0]+1+(1/2**r)),\
                    (2**r)+1\
                    ])
            if x[0]==1: # right face
                return np.array([\
                    (2**r)+1,\
                    (2**(r-1))*(-x[1]+1+(1/2**r))\
                    ])
            if x[1]==1: # back face
                return np.array([\
                    (2**(r-1))*(x[0]+1+(1/2**r)),\
                    0\
                    ])
            if x[0]==-1: # left face
                return np.array([\
                    0,\
                    (2**(r-1))*(-x[1]+1+(1/2**r))\
                    ])
    else:
        # if i not 0, bring face+frame i to face+frame 0 and use chart on 0
        A=symm_face_frame_i_to_0[i] # TO BE ADDED unique matrix bringing face+frame i to face+frame 0
        return chart(0,r,np.dot(A,x))+np.array([i*(2**r+2),0])
        # the result in translated in x, the image of the six charts is a rectangle 6*(2**r+2) by 2**r+2

def chart_inverse(i,r,x):
# chart_inverse brings points on Z^2 to points on the cube
# i is an index between 0 and 5, denoting the corresponding face index
# r is the resolution factor
# x is the 2-dimensional array for which we recover the correponding point on the cube
# we only define explicitely chart_inverse for face 0, and use the action of the group of
# symmetries of the cube to define the other inverse maps coherently
    if i==0:
        if x[0]==0: # left face
            return np.array([\
                -1,\
                -x[1]/(2**(r-1))+1+1/2**r,\
                -1+1/(2**(r-1))\
                ])
        elif x[1]==(2**r)+1: # front face
            return np.array([\
                x[0]/(2**(r-1))-1-1/2**r,\
                -1,\
                -1+1/(2**(r-1))\
                ])
        elif x[0]==(2**r)+1: # right face
            return np.array([\
                1,\
                -x[1]/(2**(r-1))+1+1/2**r,\
                -1+1/(2**(r-1))\
                ])
        elif x[1]==0: # back face
            return np.array([\
                x[0]/(2**(r-1))-1-1/2**r,\
                1,
                -1+1/(2**(r-1))\
                ])
        else: # x does not correspond to an extra pixel, and hence to the bottom face
            return np.array([\
                x[0]/(2**(r-1))-1-1/2**r,\
                -x[1]/(2**(r-1))+1+1/2**r,\
                -1])
    else:
        # if i not 0, translate the point back to the image of face 0 via the chart,
        # use chart_inverse on face 0 and move it to face i by using the action of
        # the group of symmetries of the cube
        A=symm_face_frame_0_to_i[i] # TO BE ADDED unique matrix bringing face+frame 0 to face+frame i
        return np.dot(A,chart_inverse(0,r,x-np.array([i*(2**r+2),0])))