import argparse
import numpy as np
import random
import os
import time
import random
import functools
from collections import defaultdict
import torch
import torchsparse.nn as spnn
from torchsparse.utils import make_ntuple
from core import builder
from core.utils.tensor_conversion import to_minkowski_sparseconv_tensor, to_spconv_sparseconv_tensor
from utils.config import configs
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

global dumps
global configs_all


@torch.no_grad()
def profile_adaptive_layer(kmap_mode, tiled_scatter_gather):
    for name, dump in dumps.items():
        # print(dump)
        # print(name)
        for sample in dump:
            # if sample['params']['kernel_size'][0] == 3:
            for epsilon in np.arange(0.0, 0.6, 0.1):
                # for mm_thresh in [15000, 17500, 20000, 22500, 25000]:
                for mm_thresh in [0, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]:
                    config = {'epsilon': epsilon, 'mm_thresh': mm_thresh, 'kmap_mode': kmap_mode, 'tiled_scatter_gather': tiled_scatter_gather}
                    x = sample['inputs']
                    p = sample['params']
                    # print(p)
                    layer = spnn.Conv3d(p['in_channels'], p['out_channels'], p['kernel_size'], p['stride'], p['dilation'], \
                                        transposed=p['transposed'], config=config)
                    layer = layer.cuda().eval().half()
                    torch.cuda.synchronize()
                    st = time.time()
                    # print(x.F.shape)
                    layer(x)
                    torch.cuda.synchronize()
                    ed = time.time()
                    configs_all[name][epsilon][mm_thresh] += (ed-st)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')


    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    # print(configs)
    backend = configs.model.get('backend', 'torchsparse')

    dataset = builder.make_dataset()
    dataflow = dict()
    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        #sampler=sampler,
        num_workers=configs.workers_per_gpu,
        pin_memory=False,
        collate_fn=dataset.collate_fn)


    model = builder.make_model(dataset).cuda()
    # fold bn
    for name, module in model.named_modules():
        if 'batchnorm' in module.__class__.__name__.lower():
            module.forward = lambda x: x
    model.eval()
    enable_fp16 = True
    if enable_fp16:
        model = model.half()

    for name, module in model.named_modules():
       if isinstance(module, spnn.Conv3d):
           module.config = dict(epsilon=0.0, mm_thresh=0, kmap_mode=configs.model.kmap_mode)

    dumps = defaultdict(list)

    def dump(module, inputs, outputs, name):
        if not module.transposed:
            kmap = inputs[0].kmaps.get((inputs[0].stride, make_ntuple(module.kernel_size, ndim=3), make_ntuple(module.stride, ndim=3), make_ntuple(module.dilation, ndim=3)))
        else:
            tensor_stride = tuple(inputs[0].stride[k] // make_ntuple(module.stride, ndim=3)[k] for k in range(3))
            kmap = inputs[0].kmaps[(tensor_stride, make_ntuple(module.kernel_size, ndim=3), make_ntuple(module.stride, ndim=3), make_ntuple(module.dilation, ndim=3))]
        # kernel_volumn = functools.reduce(lambda a, b: a*b, module.kernel_size)
        dumps[name].append({
            'inputs': inputs[0],
            'neighbor_offset': kmap[1].tolist() if kmap != None else None,
            'params': {'in_channels': module.in_channels, 'out_channels': module.out_channels, \
                    'kernel_size': module.kernel_size, 'stride': module.stride, 'dilation': module.dilation, 'transposed': module.transposed}
        })

    handler_collection = []
    for name, module in model.named_modules():
        if isinstance(module, spnn.Conv3d):
            if (len(module.kernel.data.shape) == 3):
                _handler = module.register_forward_hook(functools.partial(dump, name=name))
                handler_collection.append(_handler)

    # fold bn
    for name, module in model.named_modules():
        if 'batchnorm' in module.__class__.__name__.lower():
            module.forward = lambda x: x

    configs_all = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    n_sample = 100
    # if configs.dataset.name == 'waymo':
    #     # illegal memory otherwise
    #     n_sample = 30
    with torch.no_grad():
        # for idx in range(2):
        count = 0
        pbar = tqdm(enumerate(dataflow))
        for sample_idx, feed_dict in pbar:
            _inputs = dict()
            for key, value in feed_dict.items():
                if not 'name' in key and hasattr(value, 'cuda'):
                    _inputs[key] = value.cuda()
                elif not 'name' in key:
                    _inputs[key] = value    
            inputs = _inputs
            inputs['pts_input'].F = inputs['pts_input'].F.half()
            
            with torch.cuda.amp.autocast(enabled=enable_fp16):
                if count == 0:
                    for _ in range(50):
                        model(inputs)
                        inputs['pts_input'].cmaps = {}
                        inputs['pts_input'].kmaps = {}
                        configs_all = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
                        dumps = defaultdict(list)
                    print("finish doing the warm-up")

                if hasattr(inputs['pts_input'], 'build_buffer'):
                    inputs['pts_input'].build_buffer(4000000 * 64, torch.half)

                model(inputs)
            
            profile_adaptive_layer(configs.model.kmap_mode, configs.model.tiled_scatter_gather)
            
            # print(dumps['stem.0'][0])
            dumps = defaultdict(list)

            count += 1           
            if count == n_sample:
                break

    conv_configs = {}
    time_all = 0
    for name in configs_all:
        time_layer_min = 0
        for ep in configs_all[name]:
            for thresh in configs_all[name][ep]:
                if time_layer_min == 0 or time_layer_min > configs_all[name][ep][thresh]:
                    time_layer_min = configs_all[name][ep][thresh]
                    ep_best = ep
                    thresh_best = thresh
        conv_configs[name] = {'epsilon': ep_best, 'mm_thresh': thresh_best}
        time_all += time_layer_min
    is_half = 'fp16' if enable_fp16 else 'fp32'
    np.save(f"group_configs/{configs.dataset.name}_{configs.dataset.max_sweeps}_{configs.model.name}_{configs.model.cr}_{is_half}_{configs.hardware}_configs.npy", conv_configs)
    print(f"{n_sample} samples runtime upper bound is {time_all}")
