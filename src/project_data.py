import numpy as np
import torch
from platonics import platonics_dict

def project_to_solid(sphere_locations, grid):
    # sphere_locations: [n1 x 3] array
    # grid: [n2 x 3] array

    # returns (not-necessarily bijective) correspondence grid -> sphere
    # each point in grid is projected to the sphere and then attached to 
    # the closest point from sphere_locations wrt geodesic distance

    # Re-normalize given locations to ensure correctness of geodesic formula
    # Also, the grid locations need to be projected on the sphere
    norm_sphere = sphere_locations / torch.norm(sphere_locations, dim=1).unsqueeze(-1)
    norm_grid = grid / torch.norm(grid, dim=1).unsqueeze(-1)

    # https://www.wikiwand.com/en/Great-circle_distance
    # Sphere geodesic distance = arccos( dot product between normal vectors)
    # Wiki says arccos definition can be numerically unstable, but seems to work fine,
    # Alternatively, could use arctan formulation
    dot_products = norm_grid @ norm_sphere.T
    geo_dist = torch.arccos(dot_products)

    return geo_dist.argmin(axis=1)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Using cube as running example
    grid = platonics_dict[6]['3d_vertex_coords']
    
    # sphere_locations = grid
    sphere_locations = torch.randn((10, 3))
    sphere_locations = sphere_locations / torch.norm(sphere_locations, dim=1).unsqueeze(-1)

    normalized_grid = grid / torch.norm(grid, dim=1).unsqueeze(-1)
    matching = project_to_solid(sphere_locations, grid)
    print(matching)

    fig = plt.figure()
    ax = Axes3D(fig)

    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    ax.plot_wireframe(x, y, z, alpha=0.1)

    ax.scatter(*grid.T, color='g')
    ax.scatter(*normalized_grid.T, color='purple')
    ax.scatter(*sphere_locations.T, color='r')

    for item_id in range(len(grid)):
        x0, y0, z0 = grid[item_id]
        x1, y1, z1 = normalized_grid[item_id]
        x2, y2, z2 = sphere_locations[matching[item_id]]
        ax.plot([x0, x1], [y0, y1], [z0, z1], '-', color = 'red')
        ax.plot([x1, x2], [y1, y2], [z1, z2], color = 'gray')

    plt.show()