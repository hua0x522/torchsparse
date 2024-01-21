import argparse
import numpy as np
import random
import os
import sys
import time
import torch
import torchsparse
import torchsparse.backends
import torchsparse.nn as spnn
from torchsparse.nn import functional as F

from core import builder
from core.utils.tensor_conversion import to_minkowski_sparseconv_tensor, to_spconv_sparseconv_tensor
from utils.config import configs
from tqdm import tqdm

import warnings
from collections import defaultdict
from distutils.util import strtobool

from scipy.stats import gmean
from tabulate import tabulate


def user_prompt(question: str) -> bool:
    while True:
        response = input(question + " [y/n]: ")
        try:
            return bool(strtobool(response))
        except ValueError:
            print("Please use y/n or yes/no.\n")


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def get_config(dataset, backend):
    if dataset in ["semantic_kitti", "nuscenes_lidarseg"]:
        if backend == "spconv":
            config_file = f"configs/{dataset}/spconv/minkunet.yaml"
        else:
            config_file = f"configs/{dataset}/torchsparse/minkunet.yaml"
    else:
        if backend == "spconv":
            config_file = f"configs/{dataset}/centerpoint_spconv/default.yaml"
        else:
            config_file = f"configs/{dataset}/centerpoint/default.yaml"

    return config_file


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp16")
    args = parser.parse_args()
    
    precision_tag = args.precision.lower()
    assert precision_tag in ["fp16", "fp32", "tf32"], f"Unknown precision {precision_tag}. Please choose from fp16, fp32, and tf32."
    
    if precision_tag == "fp16":
        if torchsparse.backends.allow_fp16 == False:
            print("[Warning] The current device does not support fp16. Set precision to fp32")
            precision_tag = "fp32"  
    elif precision_tag == "tf32":
        if torchsparse.backends.allow_tf32 == False:
            print("[Warning] The current device does not support tf16. Set precision to fp32")
            precision_tag = "fp32"
            from spconv import constants
            constants.SPCONV_ALLOW_TF32 = False
        else:
            from spconv import constants
            constants.SPCONV_ALLOW_TF32 = True
    elif precision_tag == "fp32":
        from spconv import constants
        constants.SPCONV_ALLOW_TF32 = False
        torchsparse.backends.allow_tf32 = False   

    torch.backends.cudnn.benchmark = False
    batch_size = args.batch_size

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    BENCHMARKS = [
        ("semantic_kitti", "SemanticKITTI (1x width) segmentation"),
        ("semantic_kitti", "SemanticKITTI (0.5x width) segmentation"),
        ("nuscenes_lidarseg", "nuScenes-LiDARSeg (1 frame) segmentation"),
        ("nuscenes_lidarseg", "nuScenes-LiDARSeg (3 frames) segmentation"),
        ("nuscenes", "nuScenes detection"),
        ("waymo", "Waymo (1 frame) detection"),
        ("waymo", "Waymo (3 frames) detection"),
    ]
    BACKENDS = [
        ("torchsparse", precision_tag),
        # ("spconv", precision_tag),
        # ("ME", "fp32"),         # Minkowski Engine only supports fp32
    ]

    # TorchSparse Tuning Settings
    force_retune = True
    dataflow_prune = False
    dataflow_range = [
                    F.Dataflow.ImplicitGEMM,
                    F.Dataflow.FetchOnDemand,
                    #   F.Dataflow.GatherScatter,
                    ]

    F.set_kmap_mode("hashmap_on_the_fly")

    results = defaultdict(lambda: defaultdict(list))
    ratios = defaultdict(list)
    results_folder_prefix = "results" if not args.fast else "results_fast"

    for t, (d, benchmark) in enumerate(tqdm(BENCHMARKS, leave=False)):
        for backend, precision in tqdm(
            BACKENDS, desc=f"Benchmark: {benchmark}", leave=False
        ):
            config_file = get_config(d, backend)
            configs.reload(config_file, recursive=True)
            configs.model.cr = 1.0
            configs.model.backend = backend
            configs.model.enable_fp16 = True if precision == "fp16" else False

            if d == "semantic_kitti":
                configs.model.cr = 1.0 if t == 0 else 0.5
            elif d == "nuscenes_lidarseg":
                configs.dataset.max_sweeps = 1 if t == 2 else 3
            elif d == "waymo":
                configs.dataset.max_sweeps = 1 if t == 5 else 3

            n_frame = configs.dataset.get("max_sweeps", 1)

            save_path = f"{results_folder_prefix}/times/{configs.dataset.name}_{configs.model.cr}_{n_frame}/{backend}_{precision}.npy"
            if not args.restart and os.path.exists(save_path):
                tqdm.write("Saved results found, resuming...")
                times = np.load(save_path)
            else:                
                if args.fast:
                    dataset = builder.make_dataset(n_sample=100)
                    tuning_sample_num = 20
                else:
                    dataset = builder.make_dataset()
                    tuning_sample_num = 100
                dataflow = dict()
                dataflow = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=configs.workers_per_gpu,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn,
                )

                model = builder.make_model(dataset).cuda()

                # fold bn layers
                for key, module in model.named_modules():
                    if "batchnorm" in module.__class__.__name__.lower():
                        module.forward = lambda x: x
                model.eval()

                enable_fp16 = configs.model.enable_fp16
                if enable_fp16:
                    model = model.half()
                
                if backend == 'torchsparse': # activate torchsparse tuner
                    torchsparse.backends.benchmark = True
                    tune_tag = f'{configs.dataset.name}_{configs.model.name}_{n_frame}frame_{configs.model.cr}x_{precision_tag}_fwd'

                    torchsparse.tune(
                        model=model,
                        data_loader=dataflow,
                        n_samples=tuning_sample_num,
                        collect_fn=lambda data: data["pts_input"],
                        enable_fp16 = enable_fp16,
                        force_retune = force_retune,
                        dataflow_range = dataflow_range,
                        dataflow_prune = dataflow_prune,
                        tune_with_bwd = False,
                        tune_tag = tune_tag,
                        verbose = False,
                    )

                times = []
                for i, feed_dict in enumerate(
                    tqdm(
                        dataflow,
                        desc=f"Backend: {backend} ({precision})",
                        leave=False,
                    )
                ):
                    _inputs = dict()
                    for key, value in feed_dict.items():
                        if not 'name' in key and hasattr(value, 'cuda'):
                            _inputs[key] = value.cuda()
                        elif not 'name' in key:
                            _inputs[key] = value

                    inputs = _inputs
                    if enable_fp16:
                        inputs["pts_input"].F = inputs["pts_input"].F.half()
                    if backend == "ME":
                        inputs["pts_input"] = to_minkowski_sparseconv_tensor(inputs["pts_input"])
                    elif backend == "spconv":
                        inputs['pts_input'] = to_spconv_sparseconv_tensor(inputs['pts_input'], shape=configs.model.get('spatial_shape', None))

                    # if hasattr(inputs["pts_input"], "build_buffer"):
                    #     inputs["pts_input"].build_buffer(
                    #         4000000 * 64,
                    #         torch.half if enable_fp16 else torch.float,
                    #     )

                    with torch.cuda.amp.autocast(enabled=enable_fp16):
                        if i == 0:
                            for _ in range(10):
                                _ = model(inputs["pts_input"])
                                if backend == "torchsparse":
                                    inputs["pts_input"]._caches.cmaps.clear()
                                    inputs["pts_input"]._caches.kmaps.clear()
                                    inputs["pts_input"]._caches.hashmaps.clear()

                        torch.cuda.synchronize()
                        st = time.time()

                        _ = model(inputs["pts_input"])

                        torch.cuda.synchronize()
                        ed = time.time()
                        times.append(ed - st)

            fps = 1 / np.mean(times)
            if backend == "torchsparse":
                times_ts = times
                # ratios[backend].append(1)
                results[benchmark][f"TorchSparse++ ({precision})"].append(
                    f"{fps:.1f} FPS"
                )
            elif backend == "spconv":
                # ratio = np.mean(times_ts) / np.mean(times)
                # ratios[f"SpConv ({precision})"].append(ratio)
                results[benchmark][f"SpConv 2.3.5 ({precision})"].append(
                    f"{fps:.1f} FPS"
                )
            elif backend == "ME":
                # ratio = np.mean(times_ts) / np.mean(times)
                # ratios[f"MinkowskiEngine ({precision})"].append(ratio)
                results[benchmark][f"MinkowskiEngine ({precision})"].append(
                    f"{fps:.1f} FPS"
                )

            filename = f"{configs.dataset.name}_{configs.model.name}_result_{configs.hardware}.txt"

            os.makedirs(f"{results_folder_prefix}/summaries", exist_ok=True)
            with open(f"{results_folder_prefix}/summaries/{filename}", "a+") as fd:
                fd.write(
                    f"{backend},"
                    + f"{n_frame} frame,"
                    + f"{configs.model.cr},"
                    + f"{precision},"
                    + f"{np.sum(times):.4f}s ± {0:.4f}s,"
                    + f"{np.mean(times)*1000:.4f}ms per sample,\n"
                )

            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            np.save(save_path, times)

        tqdm.write("")
        tqdm.write(f"Evaluation results on {benchmark}:")
        tqdm.write(tabulate(results[benchmark], headers="keys", tablefmt="grid"))

    results_all = defaultdict(list)
    if len(results) > 0:
        for key in results:
            results_all["Task"].append(key)
            for benchmark in results[key]:
                results_all[benchmark] += results[key][benchmark]
        # results_all["Task"].append("Geometric Mean")
        # for benchmark in results[key]:
        #     if benchmark == "TorchSparse (fp16)":
        #         results_all[benchmark].append("1.00")
        #     else:
        #         results_all[benchmark].append(f"{gmean(ratios[benchmark]):.2f}")

        tqdm.write("")
        tqdm.write("Final evaluation results on all benchmarks:")
        tqdm.write(tabulate(results_all, headers="keys", tablefmt="grid"))
        # tqdm.write("See table 1 and table 2 in the artifact evaluation instruction to validate results.")
    else:
        tqdm.write("")
        tqdm.write("No evaluation results are found!")


if __name__ == "__main__":
    
    # SITES = {
    #     "KITTI": "http://www.cvlibs.net/datasets/kitti/user_register.php",
    #     "NuScenes": "https://nuscenes.org/sign-up?prevpath=nuscenes&prevhash=download",
    #     "Waymo Open": "https://waymo.com/open/licensing/",
    # }
    # for name, url in SITES.items():
    #     if not user_prompt(f"Have you registered at the website of {name} dataset?"):
    #         print(f"Please register at the website of {name} dataset ({url})!")
    #         sys.exit(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
