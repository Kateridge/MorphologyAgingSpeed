"""Checkpoint loading and pair-wise local/global aging-speed export."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MRIPairs
from .model import BrainAgingModel


def load_model(checkpoint_path, device):
    """Restore a trusted release checkpoint, without loading an optimizer."""
    # Training checkpoints include Python/NumPy RNG state; use trusted files only.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Expected best.pt or last.pt produced by this release")
    config = checkpoint.get("config", {})
    if not all(name in config for name in ("shape", "patches_per_axis", "crop_start")):
        raise ValueError("Checkpoint is missing the model or preprocessing configuration")
    model = BrainAgingModel(tuple(config["shape"]), config["patches_per_axis"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, config, checkpoint.get("epoch")


def predict(args):
    destination = Path(args.output)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    model, training_config, epoch = load_model(args.checkpoint, device)
    dataset = MRIPairs(
        args.data, args.pairs_csv, shape=training_config["shape"],
        crop_start=training_config["crop_start"], age_scale=args.age_scale,
        interval_range=tuple(args.interval_range),
    )
    count = len(model.predictor.experts)
    local_interval_columns = [f"local_interval_years_{index:03d}" for index in range(count)]
    local_speed_columns = [f"local_aging_speed_{index:03d}" for index in range(count)]
    generated_columns = ["chronological_interval_years", "predicted_interval_years", "aging_speed",
                         *local_interval_columns, *local_speed_columns]
    conflicts = set(dataset.columns) & set(generated_columns)
    if conflicts:
        raise ValueError(f"Input CSV contains reserved output columns: {sorted(conflicts)}")
    weights = model.predictor.relevance_logits.softmax(dim=0).detach().cpu().numpy()
    if not np.isfinite(weights).all():
        raise ValueError("Checkpoint contains non-finite patch relevance weights")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    print(f"Checkpoint epoch: {epoch}; device: {device}; pairs: {len(dataset)}; "
          f"filtered outside interval range: {dataset.filtered_count}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "predictions.csv.tmp"
    processed = 0
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream, torch.inference_mode():
            writer = csv.DictWriter(stream, fieldnames=[*dataset.columns, *generated_columns])
            writer.writeheader()
            for batch in loader:
                outputs = model(batch["starting_image"].to(device, non_blocking=True),
                                batch["followup_image"].to(device, non_blocking=True))
                local = outputs["local_interval"].cpu().numpy().astype(np.float64)
                global_interval = outputs["global_interval"].cpu().numpy().reshape(-1).astype(np.float64)
                # Read the same accepted chronological intervals in manifest order,
                # retaining CSV precision rather than converting back from float32.
                intervals = np.asarray([row[2] for row in dataset.rows[processed:processed + len(local)]])
                speeds = global_interval / intervals
                local_speeds = local / intervals[:, None]
                if not all(np.isfinite(values).all() for values in (local, global_interval, speeds, local_speeds)):
                    raise FloatingPointError("Non-finite predicted interval or aging speed")
                for index in range(len(local)):
                    row = dict(dataset.metadata[processed + index])
                    row.update(chronological_interval_years=intervals[index],
                               predicted_interval_years=global_interval[index], aging_speed=speeds[index])
                    row.update(zip(local_interval_columns, local[index]))
                    row.update(zip(local_speed_columns, local_speeds[index]))
                    writer.writerow(row)
                processed += len(local)
                print(f"Processed {processed}/{len(dataset)} pairs", flush=True)
        if processed != len(dataset):
            raise RuntimeError("Prediction count does not match the filtered manifest")
        temporary.replace(destination / "predictions.csv")
    finally:
        dataset.close()
    with (destination / "patch_weights.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patch_index", "weight", "d_start", "d_stop", "h_start", "h_stop", "w_start", "w_stop"])
        for index, (weight, patch) in enumerate(zip(weights, model.predictor.patches)):
            writer.writerow([index, float(weight), *(value for axis in patch for value in (axis.start, axis.stop))])
    run_config = {
        **vars(args), "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": epoch, "shape": training_config["shape"],
        "crop_start": training_config["crop_start"], "patches_per_axis": training_config["patches_per_axis"],
        "pairs_written": processed, "pairs_filtered": dataset.filtered_count,
    }
    (destination / "inference_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions to {destination / 'predictions.csv'}", flush=True)
