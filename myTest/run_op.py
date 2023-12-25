import numpy as np
import torch
import torch.nn as nn
from torchsparse.utils.quantize import sparse_quantize
import torchsparse as ts
from datetime import datetime


def generate_random_point_cloud(config):
    size = config['size']
    c_in = config['channel_in']
    voxel_size = config['voxel_size']
    dtype = config['dtype']
    coords = np.random.randn(size, 3) * 10
    feats = np.random.randn(size, c_in)
    coords -= np.min(coords, axis=0, keepdims=True)
    coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
    nnz = coords.shape[0]
    batchs = np.zeros((nnz, 1))
    coords = np.hstack([batchs, coords])

    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats[indices], dtype=dtype)

    feed_dict = {'coords': coords, 'feats':feats}
    return feed_dict


def forward(model, inputs, run_times=50):
    device = 'cuda:0'
    # warm up
    torch.cuda.synchronize(device=device)
    for _ in range(50):
        outputs = model(inputs)
    torch.cuda.synchronize(device=device)

    time = datetime.now()
    for _ in range(run_times):
        outputs = model(inputs)
    torch.cuda.synchronize(device=device)
    forward_time = datetime.now() - time 
    print('forward time: ', forward_time)
    return outputs 


def backward(model, features, run_times=50):
    device = 'cuda:0'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    labels = torch.ones(features.shape[0], dtype=torch.int64).to(device)
    optimizer.zero_grad()
    loss = criterion(features, labels)

    time = datetime.now()
    torch.cuda.synchronize(device=device)
    for _ in range(run_times):
        loss.backward(retain_graph=True)
    torch.cuda.synchronize(device=device)
    backward_time = datetime.now() - time 
    print('backward time: ', backward_time)
    loss.backward()


def get_ts_data(data_dict):
    coords = data_dict['coords'].clone()
    feats = data_dict['feats'].clone()
    ts_input = ts.SparseTensor(coords=coords, feats=feats)
    return ts_input


def test_ts_model(data_dict, config, mode):
    c_in = config['channel_in']
    c_out = config['channel_out']
    device = 'cuda:0'
    from torchsparse.nn import functional as F
    F.set_conv_mode(mode)
    model = ts.nn.Conv3d(c_in, c_out, kernel_size=3, stride=1).to(device)
    inputs = get_ts_data(data_dict)
    inputs = inputs.to(device)
    outputs = forward(model, inputs)
    backward(model, outputs.F)


def generate_configs():
    configs = []
    channels = [32, 64, 128, 256, 512, 1024]
    sizes = [1000, 10000, 50000, 100000]
    dtypes = [torch.float, torch.half]

    for size in sizes:
        for c_in in channels:
            for c_out in channels:
                for dtype in dtypes:
                    config = {
                        'size': size,
                        'channel_in': c_in, 
                        'channel_out': c_out, 
                        'dtype': dtype,
                        'voxel_size': 0.2
                    }
                    configs.append(config)
    return configs


if __name__ == '__main__':
    configs = generate_configs()
    for config in configs:
        print(config)
        feed_dict = generate_random_point_cloud(config)
        test_ts_model(feed_dict, config, 0)
        test_ts_model(feed_dict, config, 1)
        test_ts_model(feed_dict, config, 2)
