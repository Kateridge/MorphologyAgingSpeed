"""Train the paper's joint local-to-global brain aging model."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", required=True, help="HDF5 file containing preprocessed MRI volumes")
    parser.add_argument("--train-csv", required=True, help="Training pair manifest")
    parser.add_argument("--val-csv", help="Optional validation manifest for checkpoint selection")
    parser.add_argument("--output", required=True, help="Run directory for configuration, losses, and checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs, including completed epochs on resume")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of image pairs per batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial Adam learning rate")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers; 0 is supported")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--seed", type=int, default=20250224, help="Random seed")
    parser.add_argument("--shape", type=int, nargs=3, default=[176, 192, 176], metavar=("D", "H", "W"), help="Cropped volume shape")
    parser.add_argument("--crop-start", type=int, nargs=3, default=[8, 18, 2], metavar=("D", "H", "W"), help="Crop origin for volumes larger than --shape")
    parser.add_argument("--patches-per-axis", type=int, default=4, help="Number of patches along each spatial axis")
    parser.add_argument("--age-scale", type=float, default=1.0, help="Multiply CSV ages by this value to convert to years; use 100 for legacy RegV4 CSVs")
    parser.add_argument("--resume", help="Trusted last.pt checkpoint from this release")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not 0 < args.lr < float("inf"):
        parser.error("epochs/batch-size/lr must be positive and workers must be nonnegative")
    if any(start < 0 for start in args.crop_start):
        parser.error("crop-start must be nonnegative")
    return args


if __name__ == "__main__":
    args = parse_args()
    from aging.training import train
    train(args)
