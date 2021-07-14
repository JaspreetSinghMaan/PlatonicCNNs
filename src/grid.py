import numpy as np
import matplotlib.pyplot as plt


def square_grid(end_coords, resolution=1, ignore_dim=2):
    """
    Helper method to generate recursive square grid given 3D coordinates of end points.

    Args:
        end_coords: 3D coordinates of 4 end points, i.e. list contains exactly 4 tensors/arrays of dimension 3: (x, y, z)
        resolution: Resolution of the recursive grid, i.e. how many times we recursively sub-divide
        ignore_dim: Dimension to be ignored during grid construction (must be the same for all end points)

    Returns:
        List of 3D coordinates of square grid enclosed by end_coords.
    """

    # end_coords array must be of length 4.
    assert len(end_coords) == 4

    assert ignore_dim in [0, 1, 2] # (x, y, z)

    # check that all end points in the ignore_dim dimension have the same value
    assert end_coords[0][ignore_dim] == end_coords[1][ignore_dim] == end_coords[2][ignore_dim] == end_coords[3][ignore_dim]

    # if resolution is 1, just return the end_coords list
    if resolution <= 1:
        return end_coords

    # determine number of points in square grid at given resolution
    npoints = resolution_to_npoints(resolution)
    # total number of points in the square grid will be (npoints x npoints)
    
    ###########################################################################
    # we will create the grid in 2D and then add back the ignored 3D dimension
    ###########################################################################
    
    # create 2D mesh grid
    xx, yy = np.meshgrid( np.linspace(-1, 1, npoints), np.linspace(-1, 1, npoints) )
    
    # uncomment below to visualize the 2D grid
    # plt.plot(xx, yy, marker='.', color='k', linestyle='none'); plt.show()

    # add back the ignored 3d dimensions
    pad = np.ones(xx.shape) * end_coords[0][ignore_dim]
    if ignore_dim == 0:
        return np.stack((pad.flatten(), xx.flatten(), yy.flatten()), axis=1)
    elif ignore_dim == 1:
        return np.stack((xx.flatten(), pad.flatten(), yy.flatten()), axis=1)
    elif ignore_dim == 2:
        return np.stack((xx.flatten(), yy.flatten(), pad.flatten()), axis=1)


def resolution_to_npoints(resolution):
    """
    Helper method to recursively calculate the number of points that a line can be divided into at a given recursion resolution.
    """
    npoints = [2, 3, 5, 9, 17]  # beyond resolution 5, we can compute the number of points recursively

    if resolution <= 5:
        return npoints[resolution-1]
    else:
        n = npoints[-1]
        for _ in range(5, resolution):
            n = n* 2 - 1
        return n


if __name__ == "__main__":

    # from platonics import platonics_dict

    # cube = platonics_dict[6]
    # coords = cube['3d_vertex_coords']

    coords = {  # 3d coordinates with centre mass at (0, 0, 0)
            0: [-1, -1, -1],
            1: [1, -1, -1],
            2: [1, 1, -1],
            3: [-1, 1, -1],
            4: [-1, -1, 1],
            5: [1, -1, 1],
            6: [1, 1, 1],
            7: [-1, 1, 1]
    }
    
    # test square grid method
    end_coords = [coords[0], coords[1], coords[2], coords[3]]
    resolution = 5
    ignore_dim = 2

    print(square_grid(end_coords, resolution, ignore_dim))
