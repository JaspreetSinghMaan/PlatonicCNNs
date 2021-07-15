import argparse
import time
import torch
import matplotlib.pyplot as plt

from base_classes import Shape, Grid, Atlas, Signal
from shape_Cube import Cube
from shape_Icosahedron import Icosahedron
from data import load_data, create_dummy_data, plot_surface
from eqCNN import CubeConv2D

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)
    lf = torch.nn.CrossEntropyLoss()
    loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, opt=None):  # opt required for runtime polymorphism
  model.eval()
  logits, accs = model(data.x), []
  for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    accs.append(acc)
  return accs


def main(opt):
    # instatiate shape
    if opt['num_faces'] == 6:
        shape = Cube()
    elif opt['num_faces'] == 20:
        shape = Icosohedron()
    else:
        raise ValueError("Platonic not yet implemented!")
    #instatiate grid
    grid = Grid(shape, opt['resolution'])
    #instantiate Atlas - this will instantiate atlas of charts
    atlas = Atlas(shape, grid)

    # load omni data
    # train_loader, test_loader, train_dataset, _ = load_data(f"../data/{opt['data_file_path']}", opt['batch_size'])
    # todo this is still 2d data from s2conv paper
    # next try here https://github.com/ChiWeiHsiao/SphereNet-pytorch/blob/master/spherenet/dataset.py

    # for _ in range(5):
    #     images, labels = next(iter(train_loader))
    #     plt.imshow(images[0].squeeze().numpy(), cmap="gray")
    #     plt.show()

    #todo inside the load_data need to create custom torch dataset from spherical mnist following below steps
    # spherical_data = train_dataset #this insn't yet spherical data
    spherical_data = create_dummy_data('sphere')
    plot_surface(spherical_data)
    signal = Signal(spherical_data, grid, atlas)
    platonic_data = signal.project_to_solid()
    signal.transform_to_atlas(platonic_data, atlas)
    chart_data = signal.atlas_to_tensor()


    # instantiate PCNN.py
    if opt['g_conv_type'] == 'S2S':
        model = S2SCubeConv2D(opt)
    elif opt['g_conv_type'] == 'S2R':
        pass
    elif opt['g_conv_type'] == 'R2R':
        pass


    # standard torch training code
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_val_acc = test_acc = train_acc = best_epoch = 0
    for epoch in range(1, opt['epoch']):
        start_time = time.time()
        loss = train(model, optimizer, chart_data.data)
        train_acc, val_acc, tmp_test_acc = test(model, chart_data.data, opt)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_epoch = epoch
        print(f"Epoch: {epoch:.3d}, Runtime {time.time() - start_time:.3f}, Loss {loss:.3f}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}")
    print(f"best val accuracy {best_val_acc:.3f} with test accuracy {test_acc:.3f} at epoch {best_epoch:.d}")

    return train_acc, best_val_acc, test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--store_true', action='store_true', help='')
    # parser.add_argument('--string', type=str, default='string', help='string')
    # parser.add_argument('--int', type=int, default=4, help='int')
    # parser.add_argument('--float', type=float, default=0.5, help='float')

    #data args
    parser.add_argument('--dataset', type=str, default='omni_mnist', help = 'omni_mnist, environment')
    parser.add_argument('--data_file_path', type=str, default="s2_mnist.gz", help = 'S2 data file')

    #main args
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

    #Platonic Shape args
    # parser.add_argument('--shape', type=str, default='Cube', help='Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron')
    parser.add_argument('--num_faces', type=int, default=6, help='4, 6, 8, 12, 20')
    parser.add_argument('--resolution', type=int, default=4, help='int')

    #PCNN args
    parser.add_argument('--g_conv_type', type=str, default='S2S', help='S2S, S2R, R2R')
    parser.add_argument('--c_in', type=int, default=6, help='int')
    parser.add_argument('--c_out', type=int, default=4, help='int')
    parser.add_argument('--r_in', type=int, default=1, help='int')
    parser.add_argument('--r_out', type=int, default=1, help='int')
    parser.add_argument('--kernel_size', type=int, default=3, help='int')
    parser.add_argument('--stride', type=int, default=1, help='int')
    parser.add_argument('--bias', action='store_true', help='')
    parser.add_argument('--padding', type=int, default=0, help='int')

    args = parser.parse_args()
    opt = vars(args)
    main(opt)