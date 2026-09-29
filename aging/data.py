"""Longitudinal MRI pairs stored as HDF5 volumes and a CSV manifest."""

import csv
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIPairs(Dataset):
    def __init__(self, h5_path, csv_path, shape=(176, 192, 176),
                 crop_start=(8, 18, 2), age_scale=1.0, interval_range=(1.0, 16.0)):
        self.h5_path = str(h5_path)
        self.shape = tuple(shape)
        self.crop_start = tuple(crop_start)
        self._file = None
        self._pid = None
        self.rows = []
        self.metadata = []
        self.subjects = set()
        self.filtered_count = 0
        if age_scale <= 0 or not np.isfinite(age_scale):
            raise ValueError("age_scale must be positive and finite")
        required = {"subject_id", "starting_session_id", "followup_session_id",
                    "starting_age", "followup_age"}
        with open(csv_path, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            self.columns = list(reader.fieldnames or ())
            if len(self.columns) != len(set(self.columns)):
                raise ValueError(f"{csv_path}: duplicate CSV column names")
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{csv_path}: missing CSV columns {sorted(missing)}")
            for line, row in enumerate(reader, start=2):
                try:
                    subject, start, followup = (
                        row[key].strip() for key in
                        ("subject_id", "starting_session_id", "followup_session_id")
                    )
                    start_age = float(row["starting_age"]) * age_scale
                    followup_age = float(row["followup_age"]) * age_scale
                    interval = followup_age - start_age
                    if not subject or not start or not followup or start == followup:
                        raise ValueError("IDs must be nonempty and sessions must differ")
                    if not np.isfinite([start_age, followup_age]).all() or start_age <= 0 or interval <= 0:
                        raise ValueError("ages must be finite, positive, and increasing")
                    if row.get("age_diff", "").strip():
                        supplied = float(row["age_diff"]) * age_scale
                        if not np.isfinite(supplied) or supplied <= 0 or abs(supplied - interval) > 0.02:
                            raise ValueError("age_diff disagrees with followup_age - starting_age")
                        interval = supplied
                except (ValueError, AttributeError, TypeError) as error:
                    raise ValueError(f"{csv_path}, line {line}: {error}") from error
                # Check split leakage even for subjects whose pairs are filtered out.
                self.subjects.add(subject)
                if not interval_range[0] <= interval <= interval_range[1]:
                    self.filtered_count += 1
                    continue
                self.rows.append((f"{subject}_{start}", f"{subject}_{followup}", interval))
                self.metadata.append(row.copy())
        if not self.rows:
            raise ValueError(f"{csv_path}: no pairs remain in the {interval_range} year interval range")
        # Check keys and shapes up front without retaining a handle across workers.
        with h5py.File(self.h5_path, "r") as volumes:
            for key in {key for row in self.rows for key in row[:2]}:
                if key not in volumes or not isinstance(volumes[key], h5py.Dataset):
                    raise ValueError(f"HDF5 volume missing: {key}")
                size = volumes[key].shape
                if len(size) != 3 or (size != self.shape and any(
                        n < start + length for n, start, length in zip(size, self.crop_start, self.shape))):
                    raise ValueError(f"{key}: volume shape {size} cannot provide crop {self.shape} at {self.crop_start}")

    def __len__(self):
        return len(self.rows)

    def _volume(self, key):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.h5_path, "r")
            self._pid = os.getpid()
        volume = self._file[key]
        slices = tuple(slice(start, start + size) for start, size in zip(self.crop_start, self.shape))
        image = np.asarray(volume[:] if volume.shape == self.shape else volume[slices], dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"{key}: image contains NaN or infinity")
        minimum, maximum = image.min(), image.max()
        if maximum <= minimum:
            raise ValueError(f"{key}: constant image cannot be used for registration")
        image = (image - minimum) / (maximum - minimum)
        return torch.from_numpy(image[None])

    def __getitem__(self, index):
        start, followup, interval = self.rows[index]
        return {"starting_image": self._volume(start), "followup_image": self._volume(followup),
                "interval": torch.tensor(interval, dtype=torch.float32)}

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()
