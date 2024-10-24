import argparse
import os
from pathlib import Path
from pprint import pprint

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate result by FID and Sort.",
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="The images folder for comparison, e.g. ./datasets/horse2zebra/testB.",
    )
    parser.add_argument(
        "result_dir", type=str, help="The result folder of the experiment."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Which epochs are to be evaluated. By default, evaluate all epochs.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        help="Which phase are to be evaluated.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="Which order of metric value is used in sorting.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes to use for data loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
        help=(
            "Dimensionality of Inception features to use. "
            "By default, uses pool3 features"
        ),
    )
    args = parser.parse_args()
    assert os.path.isdir(args.images_dir) and os.path.isdir(args.result_dir)
    assert args.topk > 0
    return args


if __name__ == "__main__":
    args = parse_args()

    result_dir = Path(args.result_dir)
    if args.epochs is None:
        epoch_dirs = list(result_dir.glob(f"{args.phase}_*"))
    else:
        epoch_dirs = [result_dir / f"{args.phase}_{epoch}" for epoch in args.epochs]

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    # if args.num_workers is None:
    #     try:
    #         num_cpus = len(os.sched_getaffinity(0))
    #     except AttributeError:
    #         # os.sched_getaffinity is not available under Windows, use
    #         # os.cpu_count instead (which may not return the *available* number
    #         # of CPUs).
    #         num_cpus = os.cpu_count()

    #     num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    # else:
    #     num_workers = args.num_workers
    num_workers = args.num_workers

    results = {}
    for epoch_dir in tqdm(epoch_dirs):
        fake = epoch_dir / "images" / "fake_B"
        fid = calculate_fid_given_paths(
            [args.images_dir, fake.as_posix()],
            args.batch_size,
            device,
            args.dims,
            num_workers,
        )
        results[epoch_dir] = fid

    results = sorted(
        results.items(), key=lambda v: v[1], reverse=(args.order == "desc")
    )[: args.topk]

    pprint(dict(results), sort_dicts=False)
