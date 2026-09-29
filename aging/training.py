"""Joint training, optional validation-loss monitoring, and resumable checkpoints."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .data import MRIPairs
from .losses import training_loss
from .model import BrainAgingModel


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    sums = dict.fromkeys(("total", "registration", "smoothness", "local", "global"), 0.0)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["starting_image"], batch["followup_image"])
            losses = training_loss(outputs, batch["followup_image"], batch["interval"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite loss; check image values and training settings")
            if optimizer is not None:
                losses["total"].backward()
                optimizer.step()
            size = batch["interval"].shape[0]
            count += size
            for name, loss in losses.items():
                sums[name] += loss.item() * size
    return {name: value / count for name, value in sums.items()}


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    config = vars(args).copy()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; use --resume or a new --output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu for small smoke runs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader_rng = torch.Generator().manual_seed(args.seed)
    dataset_args = dict(h5_path=args.data, shape=args.shape, crop_start=args.crop_start,
                        age_scale=args.age_scale, interval_range=(1.0, 16.0))
    training = MRIPairs(csv_path=args.train_csv, **dataset_args)
    validation = MRIPairs(csv_path=args.val_csv, **dataset_args) if args.val_csv else None
    if validation is not None and training.subjects & validation.subjects:
        raise ValueError("Training and validation CSVs contain overlapping subject IDs")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                       worker_init_fn=seed_worker)
    # Keep sampling RNG separate from worker seeds: persistent-worker startup
    # must not change the next epoch's shuffle order after resuming.
    training_loader = DataLoader(
        training, sampler=RandomSampler(training, generator=loader_rng),
        generator=torch.Generator().manual_seed(args.seed + 1), **loader_args,
    )
    validation_loader = DataLoader(
        validation, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 2),
        **loader_args,
    ) if validation is not None else None
    model = BrainAgingModel(tuple(args.shape), args.patches_per_axis).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    first_epoch, best = 0, float("inf")
    if args.resume:
        # Checkpoints contain RNG and optimizer state; load only trusted files.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Expected a checkpoint produced by this release")
        allowed_changes = {"resume", "output", "epochs", "device", "workers"}
        for name, value in config.items():
            if name not in allowed_changes and checkpoint["config"].get(name) != value:
                raise ValueError(f"Resume configuration differs for {name}; use the original setting")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch, best = checkpoint["epoch"], checkpoint["best_loss"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        loader_rng.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if first_epoch >= args.epochs:
        raise ValueError("--epochs must exceed the number of completed epochs in the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device}; train pairs: {len(training)}; filtered: {training.filtered_count}", flush=True)
    if validation is not None:
        print(f"Validation pairs: {len(validation)}; filtered: {validation.filtered_count}", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    log_path = output / "losses.csv"
    terms = ("total", "registration", "smoothness", "local", "global")
    fields = ["epoch", "lr"] + [f"train_{name}" for name in terms]
    if validation is not None:
        fields += [f"val_{name}" for name in terms]
    # A resumed branch gets its own log, preventing duplicate/contradictory epochs.
    if args.resume and log_path.exists():
        log_path = output / f"losses_from_epoch_{first_epoch + 1}.csv"
        if log_path.exists():
            raise ValueError(f"Resume log already exists: {log_path}; choose a new output directory")
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(first_epoch, args.epochs):
            lr = optimizer.param_groups[0]["lr"]
            train_losses = run_epoch(model, training_loader, device, optimizer)
            row = {"epoch": epoch + 1, "lr": lr,
                   **{f"train_{name}": value for name, value in train_losses.items()}}
            score = train_losses["total"]
            if validation_loader is not None:
                val_losses = run_epoch(model, validation_loader, device)
                row.update({f"val_{name}": value for name, value in val_losses.items()})
                score = val_losses["total"]
            writer.writerow(row)
            stream.flush()
            # Preserve RegV4's schedule: decay after zero-based epochs 0, 2, 4, ...
            if epoch % 2 == 0:
                scheduler.step()
            improved = score < best
            best = min(best, score)
            checkpoint = {
                "format_version": 1, "epoch": epoch + 1, "best_loss": best, "config": config,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "loader_rng": loader_rng.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            save_checkpoint(output / "last.pt", checkpoint)
            if improved:
                save_checkpoint(output / "best.pt", checkpoint)
            print(f"Epoch {epoch + 1}/{args.epochs}: train={train_losses['total']:.6f}, "
                  f"selection_loss={score:.6f}, lr={lr:.6g}", flush=True)
    training.close()
    if validation is not None:
        validation.close()
