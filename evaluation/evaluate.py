import argparse
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch
import torchsparse.nn as spnn
from scipy.stats import gmean
from tabulate import tabulate
from tqdm import tqdm

from core import builder
from core.utils.tensor_conversion import (
    to_minkowski_sparseconv_tensor,
    to_spconv_sparseconv_tensor,
)
from utils.config import configs


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False

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
        ("torchsparse", "fp32"),
        ("spconv", "fp16"),
        ("spconv", "fp32"),
        ("ME", "fp32"),
    ]

    results = defaultdict(lambda: defaultdict(list))
    ratios = defaultdict(list)
    results_folder_prefix = "results" if not args.fast else "results_fast"

    for t, (d, benchmark) in enumerate(BENCHMARKS):
    # for t, (d, benchmark) in enumerate(tqdm(BENCHMARKS, leave=False)):
        # for backend, precision in tqdm(
        #     BACKENDS, desc=f"Benchmark: {benchmark}", leave=False
        # ):
        for backend, precision in BACKENDS:
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

            if backend == "torchsparse":
                configs.conv = 3
            n_frame = configs.dataset.get("max_sweeps", 1)
            mm_types = "adaptive groups" if configs.conv == 3 else "baseline"

            save_path = f"{results_folder_prefix}/times/{configs.dataset.name}_{configs.model.cr}_{n_frame}/{backend}_{precision}.npy"
            if not args.restart and os.path.exists(save_path):
                tqdm.write("Saved results found, resuming...")
                times = np.load(save_path)
            else:
                dataset = builder.make_dataset(100 if args.fast else -1)
                dataflow = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=configs.workers_per_gpu,
                    pin_memory=False,
                    collate_fn=dataset.collate_fn,
                )

                model = builder.make_model(dataset).cuda()
                for key, module in model.named_modules():
                    if "batchnorm" in module.__class__.__name__.lower():
                        module.forward = lambda x: x
                model.eval()

                enable_fp16 = configs.model.enable_fp16
                if enable_fp16:
                    model = model.half()

                tiled_scatter_gather = configs.model.get("tiled_scatter_gather", False)
                if configs.conv == 3:
                    # assert enable_fp16
                    conv_config_fn = f"group_configs/{configs.dataset.name}_{n_frame}_{configs.model.name}_{configs.model.cr}_fp16_{configs.hardware}_configs.npy"
                    if not os.path.exists(conv_config_fn):
                        print("profiling best config for each layer...")
                        configs_path = get_config(d, "torchsparse")
                        os.system(
                            f"OMP_NUM_THREADS=1 python group_profile.py {configs_path} --model.enable_fp16 True --model.cr {configs.model.cr} \
                            --hardware {configs.hardware} --dataset.max_sweeps {n_frame} --model.tiled_scatter_gather {tiled_scatter_gather}"
                        )
                    conv_configs = np.load(conv_config_fn, allow_pickle=True).item()

                for key, module in model.named_modules():
                    if isinstance(module, spnn.Conv3d):
                        if configs.conv == 0:
                            module.config = dict(
                                epsilon=0,
                                mm_thresh=0,
                                kmap_mode=configs.model.kmap_mode,
                                tiled_scatter_gather=tiled_scatter_gather,
                            )
                        elif configs.conv == 3:
                            if key in conv_configs:
                                module.config = dict(
                                    epsilon=conv_configs[key]["epsilon"],
                                    mm_thresh=conv_configs[key]["mm_thresh"],
                                    kmap_mode=configs.model.kmap_mode,
                                    tiled_scatter_gather=tiled_scatter_gather,
                                )
                            else:
                                module.config = dict(
                                    epsilon=0,
                                    mm_thresh=0,
                                    kmap_mode=configs.model.kmap_mode,
                                    tiled_scatter_gather=tiled_scatter_gather,
                                )

                times = []
                for i, feed_dict in enumerate(
                    tqdm(
                        dataflow,
                        desc=f"Backend: {backend} ({precision})",
                        leave=False,
                    )
                ):
                    inputs = {}
                    for key, val in feed_dict.items():
                        if "name" in key:
                            continue
                        if hasattr(val, "cuda"):
                            val = val.cuda()
                        inputs[key] = val

                    if enable_fp16:
                        inputs["pts_input"].F = inputs["pts_input"].F.half()

                    if backend == "ME":
                        inputs["pts_input"] = to_minkowski_sparseconv_tensor(
                            inputs["pts_input"]
                        )
                    elif backend == "spconv":
                        inputs["pts_input"] = to_spconv_sparseconv_tensor(
                            inputs["pts_input"],
                            shape=configs.model.get("spatial_shape", None),
                        )

                    if hasattr(inputs["pts_input"], "build_buffer"):
                        inputs["pts_input"].build_buffer(
                            4000000 * 64,
                            torch.half if enable_fp16 else torch.float,
                        )

                    with torch.cuda.amp.autocast(enabled=enable_fp16):
                        if i == 0:
                            for _ in range(10):
                                _ = model(inputs)
                                if backend == "torchsparse":
                                    inputs["pts_input"].cmaps.clear()
                                    inputs["pts_input"].kmaps.clear()

                        start_time = cuda_time()
                        _ = model(inputs)
                        times.append(cuda_time() - start_time)

            fps = 1 / np.mean(times)
            if backend == "torchsparse":
                times_ts = times
                ratios[backend].append(1)
                results[benchmark][f"TorchSparse ({precision})"].append(
                    f"1.00 ({fps:.1f} FPS)"
                )
            elif backend == "spconv":
                ratio = np.mean(times_ts) / np.mean(times)
                ratios[f"SPConv ({precision})"].append(ratio)
                results[benchmark][f"SPConv ({precision})"].append(
                    f"{ratio:.2f} ({fps:.1f} FPS)"
                )
            elif backend == "ME":
                ratio = np.mean(times_ts) / np.mean(times)
                ratios[f"MinkowskiEngine ({precision})"].append(ratio)
                results[benchmark][f"MinkowskiEngine ({precision})"].append(
                    f"{ratio:.2f} ({fps:.1f} FPS)"
                )

            filename = f"{configs.dataset.name}_{configs.model.name}_result_{configs.hardware}.txt"

            os.makedirs(f"{results_folder_prefix}/summaries", exist_ok=True)
            with open(f"{results_folder_prefix}/summaries/{filename}", "a+") as fd:
                fd.write(
                    f"{backend},"
                    + f"{n_frame} frame,"
                    + f"{configs.model.cr},"
                    + f"{precision},"
                    + f"{mm_types},"
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
        results_all["Task"].append("Geometric Mean")
        for benchmark in results[key]:
            if benchmark == "TorchSparse (fp16)":
                results_all[benchmark].append("1.00")
            else:
                results_all[benchmark].append(f"{gmean(ratios[benchmark]):.2f}")

        tqdm.write("")
        tqdm.write("Final evaluation results on all benchmarks:")
        tqdm.write(tabulate(results_all, headers="keys", tablefmt="grid"))
        tqdm.write("See table 1 and table 2 in the artifact evaluation instruction to validate results.")
    else:
        tqdm.write("")
        tqdm.write("No evaluation results are found!")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    SITES = {
        "KITTI": "http://www.cvlibs.net/datasets/kitti/user_register.php",
        "NuScenes": "https://nuscenes.org/sign-up?prevpath=nuscenes&prevhash=download",
        "Waymo Open": "https://waymo.com/open/licensing/",
    }
    # for name, url in SITES.items():
    #     if not user_prompt(f"Have you registered at the website of {name} dataset?"):
    #         print(f"Please register at the website of {name} dataset ({url})!")
    #         sys.exit(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()