import torch
from platonics import platonics_dict

def skew_symmetric_cross_product(v):
    '''
    applies skew_symmetric_cross_product as per
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    :param v: a (3,) vector
    :return: Sv a (3,3) matrix
    '''
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]
    return torch.tensor([[0, -v3, v2],[v3, 0, -v1],[-v2, v1, 0]])


def find_3d_rotation(a, b):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = torch.cross(a, b)
    s = torch.linalg.norm(v)
    c = torch.dot(a, b)
    R = torch.eye(len(a)) + skew_symmetric_cross_product(v) + skew_symmetric_cross_product(v) @ skew_symmetric_cross_product(v) * (1-c) / s**2
    return R

def find_3d_frame_transform(f1ex, f2ex, f1ey, f2ey):
    #I think you can do this: assume that we consider e1,e2 orthonormal and wanna send them to f1,f2, again orthonormal.
    # Then the unique rotation doing this is f_1^T * e_1 + f_2^T * e_2 + f_3^T * e_3,
    # where e_3 and f_3 are the cross product of e_1,e_2 and f_1,f_2 respectively


    return R
