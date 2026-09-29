"""Apply a trained brain aging model to test population pairs."""

import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Trusted best.pt or last.pt from this release")
    parser.add_argument("--data", required=True, help="HDF5 file containing preprocessed test MRI volumes")
    parser.add_argument("--pairs-csv", required=True, help="Test pair manifest; additional metadata columns are preserved")
    parser.add_argument("--output", required=True, help="New or empty directory for predictions and patch weights")
    parser.add_argument("--batch-size", type=int, default=1, help="Image pairs per inference batch")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--age-scale", type=float, default=1.0, help="Multiply this manifest's ages by this value to convert to years; use 100 for legacy CSVs")
    parser.add_argument("--interval-range", type=float, nargs=2, default=[1.0, 16.0],
                        metavar=("MIN", "MAX"), help="Inclusive chronological interval range in years")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")
    if not math.isfinite(args.age_scale) or args.age_scale <= 0:
        parser.error("age-scale must be finite and positive")
    minimum, maximum = args.interval_range
    if not all(math.isfinite(v) for v in args.interval_range) or not 0 < minimum <= maximum:
        parser.error("interval-range must satisfy 0 < MIN <= MAX with finite values")
    return args


if __name__ == "__main__":
    args = parse_args()
    from aging.inference import predict
    predict(args)
