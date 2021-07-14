import gzip
import pickle
import numpy as np
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Platonic_Dataset(Dataset):
    def __init__(self, root_dir, num_faces):
        pass


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

def visualise_spherical(data):
    '''
    here we visualise the spherical data in 3d
    use mayavi or matplotlib 3d
    :return:
    '''
    pass