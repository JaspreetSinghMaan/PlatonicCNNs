import gzip
import pickle
import numpy as np
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm


def load_data(path, batch_size):
    with gzip.open(path, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))

    # TODO normalize dataset
    # mean = train_data.mean()
    # stdv = train_data.std()

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset

def plot_grid(data):
    '''
    here we visualise the spherical data in 3d
    use mayavi or matplotlib 3d
    :return:
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    x = data[0,:,:]
    y = data[1,:,:]
    z = data[2,:,:]

    ax.plot_wireframe(x, y, z, alpha=0.1)

    plt.show()

def plot_surface(data):
    #https://stackoverflow.com/questions/15134004/colored-wireframe-plot-in-matplotlib
    x = data[0][0,:,:]
    y = data[0][1,:,:]
    z = data[0][2,:,:]
    colors = data[1]

    # Normalize to [0,1]
    # s = x + y + z
    # norm = plt.Normalize(z.min(), z.max())
    # colors = cm.viridis(norm(z))
    # norm = plt.Normalize(s.min(), s.max())
    # colors = cm.viridis(norm(x+y+z))
    # colors = cm.rainbow(norm(x+y+z))

    rcount, ccount, _ = colors.shape

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rcount=rcount, ccount=ccount,
                           facecolors=colors, shade=False)
    # surf = ax.plot_surface(x, y, z, cmap=cm.jet)
    surf.set_facecolor((0, 0, 0, 0)) #option to fill in the 2d surface squares
    plt.show()

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Platonic_Dataset(Dataset):
    def __init__(self, root_dir, num_faces):
        pass

def create_dummy_data(shape):
    if shape == 'sphere':
        u = np.linspace(0, np.pi, 30)
        v = np.linspace(0, 2 * np.pi, 30)
        x = np.outer(np.sin(u), np.sin(v))
        y = np.outer(np.sin(u), np.cos(v))
        z = np.outer(np.cos(u), np.ones_like(v))

    elif shape == 'surface':
        x, y, z = axes3d.get_test_data(0.2)

    s = x + y + z
    norm = plt.Normalize(s.min(), s.max())
    norm = plt.Normalize(s.min(), s.max())
    colors = cm.rainbow(norm(x+y+z))

    data = (np.stack([x, y, z]), colors)
    return data

if __name__ == "__main__":
    data = create_dummy_data('sphere')
    # plot_grid(data)
    plot_surface(data)