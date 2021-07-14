import torch
import numpy as np

# Redundancy level (tet, 96/12) (oct, 288/24) (ico, 480/60)
# Number of matrix operations used to generate full group vs # of elements in group

def rollout_generators(generators):
    # 'Generates' group based on list of generators

    # Start from trivial group
    group = [torch.eye(3, dtype=torch.double)]

    # This brute force approach can be slow for larger groups, but is fast enough for 
    # symmetries of Platonic solids
    add_new = True
    i = 0 
    while add_new:
        # If this flag does not change by end of cycle, we did not come up with any
        # new elements, and by then we should have the full group 
        add_new = False
        # Loop over items and 'create' new proposed elements by multiplying with
        for x in group:
            for g in generators:
                for (a1, b1) in [(x, g), (g, x)]: # Try both orders since rotations might not commute
                    proposal = a1 @ b1
                    # Filter out precision artifacts before comparing (zero out if abs value < 1e-8)
                    proposal =  proposal * (torch.abs(proposal) > 1e-8)
                    if not(is_in_list(proposal, group)):
                        # If this is a previously unseen element, add it to list
                        # TODO: might want to hash 'proposal' to speed up checks wrt dict
                        add_new = True
                        group += [proposal]
                        # print(len(group))

    return group

def generate_group(generators, group_size):
    group = rollout_generators(generators)

    # Check we are not missing any rotation wrt theoretical number of group elements
    assert len(group) == group_size

    return group

def tetrahedral_symmetries():
    # Source: https://tu-dresden.de/mn/ressourcen/dateien/international-summer-school-symmetries-and-phase-transitions/King-Group_Theory_Lectures_Day_1.pdf?lang=en
    # Slide 22

    group_size = 12
    # TODO: do we also want to consider reflections? Icosahedral paper seems to only consider
    # rotational symmetries. If so, need to add extra generators.
    generators = [torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.double),
                  torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.double)]

    return generate_group(generators, group_size)

def octahedral_symmetries():
    # Also the symmetry group for the cube

    # Source: https://www.wikiwand.com/en/Octahedral_symmetry#/Rotation_matrices

    group_size = 24
    # TODO: do we also want to consider reflections? Icosahedral paper seems to only consider
    # rotational symmetries. If so, need to add extra generators.
    generators = [torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.double),
                  torch.tensor([[0, 1., 0], [0, 0, 1], [1, 0, 0]], dtype=torch.double),
                  torch.tensor([[0, 1., 0], [1, 0, 0], [0, 0, -1]], dtype=torch.double)]

    return generate_group(generators, group_size)

def icosahedral_symmetries():
    # Also the symmetry group for the dodecahedron

    # Source: https://www.wikiwand.com/en/Icosahedral_symmetry#/Group_structure

    group_size = 60
    Phi = (1. + np.sqrt(5)) / 2  # 1.61803
    # TODO: do we also want to consider reflections? Icosahedral paper seems to only consider
    # rotational symmetries. If so, need to add extra generators.
    generators = [torch.tensor([[-1., 0, 0], [0, -1., 0], [0, 0, 1]], dtype=torch.double),
                  0.5*torch.tensor([[1-Phi, Phi, -1.], [-Phi, -1, 1-Phi], [-1, Phi-1, Phi]], dtype=torch.double)]

    return generate_group(generators, group_size)


def check_group_axioms(group):
    # *Very naively* checking all the group axioms

    for x in group:

        # Make sure x is a rotationally symmetric transformation (det = 1)
        # TODO: this might need to be extended if we consider reflections to allow -1 det
        assert np.isclose(torch.linalg.det(x).item(), 1.)

        double_coset = [x@g for g in group] + [g@x for g in group]
        double_coset = list(set(double_coset))

        # Not checking identity element as we include it by construction
        # Associativity follows from associativity of matrix multiplication

        # Checking closure of the group operation
        for h in double_coset:
            assert is_in_list(h, group)

        # Check that there is an inverse element for x 
        assert is_in_list(torch.eye(3, dtype=torch.double), double_coset)
    
    print('All group axioms satsified')

def is_in_list(my_tensor, tensor_list):
    # Checks if there is any tensor in tensor_list numerically equal to my_tensor
    return any(torch.allclose(my_tensor, x_) for x_ in tensor_list)


if __name__ == "__main__":

    for group_fn in [tetrahedral_symmetries, octahedral_symmetries, icosahedral_symmetries]:
        group_elements = group_fn()
        print('Generated ' + group_fn.__name__ + ' group')
        check_group_axioms(group_elements)
