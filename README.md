# Morphology-based Brain Aging Speed

Implementation for **Interpretable Local-to-Global Estimation of Brain
Aging Speed From Morphological Changes Using Longitudinal Structural MRI Data**,
Yuanwang Zhang, Hongming Li, and Yong Fan, MICCAI 2026.
[Paper DOI](https://doi.org/10.1007/978-3-032-38172-9_31).

Given two longitudinal T1-weighted MRIs, the model learns diffeomorphic
registration, computes a Jacobian determinant map, predicts age intervals from
spatial patches, and combines them with learned softmax relevance weights.
Predictions are intervals in years;
brain aging speed is the predicted interval divided by the actual interval.

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment, install
PyTorch for your hardware, then install the remaining requirements:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset preparation

### Participants and splits

Prepare longitudinal pairs from cognitively unimpaired (CU) participants. In the
paper, a subject is CU only if **all of its follow-up sessions** are CU.
Within each subject, the paper constructs all chronologically ordered pairs with an
interval of **1 through 16 years**.

### MRI preprocessing
Please refer to the paper for preprocessing steps. The default network input is **176 x 192 x 176** in array-axis order `(D, H, W)`. Each processed image is then min-max scaled to `[0, 1]`.

### Directory layout and HDF5 format

```text
data/
  images.h5
  training_pairs.csv
  validation_pairs.csv       # optional
  test_pairs.csv             # for inference on the test population
```

Store each image as a root-level HDF5 dataset named
`<subject_id>_<session_id>`. Its value is a numeric **3D** MRI array without
batch or channel dimensions. For example:

```text
images.h5
  example001_baseline    (176, 192, 176), float32
  example001_month24     (176, 192, 176), float32
  example001_month48     (176, 192, 176), float32
  example002_baseline    (176, 192, 176), float32
  example002_month36     (176, 192, 176), float32
```

### Pair manifests and age units

Each CSV requires these five columns:

```csv
subject_id,starting_session_id,followup_session_id,starting_age,followup_age
example001,baseline,month24,65.0,67.0
example001,baseline,month48,65.0,69.0
example001,month24,month48,67.0,69.0
```

Ages are **years by default**. The target is `followup_age - starting_age`.
An optional `age_diff` column can supply the interval in the same units. The manifests in `examples/` illustrate the schema
with fictitious IDs.

## Training

Run from the repository root. The following uses the paper's image shape,
`4 x 4 x 4` patches, batch size 8, initial learning rate `1e-3`, and 100 epochs:

```bash
python train.py --data data/images.h5 --train-csv data/training_pairs.csv --val-csv data/validation_pairs.csv --output runs/paper
```

Omit `--val-csv` to train without validation.

The available hyperparameters and runtime options are listed below. Override a
default by adding the corresponding option to the training command.

| Option | Default | Description |
| --- | --- | --- |
| `--epochs` | `100` | Total training epochs, including completed epochs when resuming. |
| `--batch-size` | `8` | Number of longitudinal image pairs per batch. |
| `--lr` | `0.001` | Initial Adam learning rate. |
| `--patches-per-axis` | `4` | Number of patches along each spatial axis; the default gives 64 local regressors. |
| `--seed` | `20250224` | Random seed for model initialization and data shuffling. |

`--data`, `--train-csv`, and `--output` are required paths with no defaults.
`--val-csv` and `--resume` default to unset, so validation is omitted and training
starts from scratch unless these options are supplied.
Run `python train.py --help` for the complete argument list.

Each run produces:

- `config.json`: all command-line settings.
- `losses.csv`: sample-weighted epoch training losses and optional validation losses.
- `last.pt`: the latest completed epoch, model, optimizer, scheduler, configuration,
  and random-number-generator states.
- `best.pt`: the checkpoint with the lowest total validation loss, or lowest total
  training loss when validation is omitted.

Resume with the same data/model/optimizer arguments and a larger total epoch
count if needed:

```bash
python train.py --data data/images.h5 --train-csv data/training_pairs.csv --val-csv data/validation_pairs.csv --output runs/paper --resume runs/paper/last.pt --epochs 100
```

## Inference on test populations

Use a model trained on CU participants to predict intervals and aging speeds for
test pairs. Prepare the images with the same preprocessing and spatial alignment
used for training. The test CSV uses the same five required columns as the training
manifest. Optional columns such as `cohort`, `starting_diagnosis`,
`followup_diagnosis`, and `conversion_type` are preserved in the results.
See `examples/test_pairs.csv` for a fictitious example.

```bash
python test.py --checkpoint runs/aaa/best.pt --data data/images.h5 --pairs-csv data/test_pairs.csv --output runs/aaa/test
```

| Option | Default | Description |
| --- | --- | --- |
| `--batch-size` | `1` | Number of pairs per inference batch. |
| `--device` | `cuda` | Device for inference; `cpu` is also supported. |
| `--interval-range MIN MAX` | `1.0 16.0` | Inclusive chronological interval range in years; pairs outside this range are counted and skipped. Both bounds must be positive. |

`--checkpoint`, `--data`, `--pairs-csv`, and `--output` are required. The output
directory must be new or empty. Use `python test.py --help` for all options.
Changing the interval range allows inference outside the training range; very
short chronological intervals make the speed ratio sensitive to prediction error.

For each accepted pair, the script calculates:

```text
global aging speed = predicted global interval / chronological interval
local aging speed  = predicted patch interval / chronological interval
```

Each inference run produces:

- `predictions.csv`: original manifest columns and values, in the original order
  after interval filtering, plus `chronological_interval_years`,
  `predicted_interval_years`, `aging_speed`, and each patch's
  `local_interval_years_000` / `local_aging_speed_000` (through index `063` for
  the default 64 patches). Original age columns retain their input units;
  all generated interval columns use years. Rows represent pairs, not subject averages.
- `patch_weights.csv`: learned softmax relevance weights and patch boundaries
  in the cropped volume. Patch indices are zero-based, with W varying fastest,
  then H, then D; slice stops are exclusive. The weighted local intervals yield
  the global interval, and the same weights combine local speeds into global speed.
- `inference_config.json`: checkpoint path and epoch, preprocessing/model settings,
  runtime options, and counts of processed and filtered pairs.

## Repository contents

```text
aging/data.py       HDF5/CSV loading, cropping, normalization, pair checks
aging/spatial.py    Warping, velocity integration, Jacobian determinants
aging/model.py      Registration U-Net and local-to-global regressors
aging/losses.py     The four training losses
aging/training.py   Training, optional validation losses, checkpoint/resume
aging/inference.py  Checkpoint loading and local/global aging-speed export
train.py            Command-line entry point
test.py             Inference on test pair manifests
examples/           Fictitious pair-manifest examples
```

## Acknowledgments

- Balakrishnan et al., *VoxelMorph: A Learning Framework for Deformable Medical
  Image Registration*, IEEE Transactions on Medical Imaging, 2019.
- Dalca et al., *Unsupervised Learning of Probabilistic Diffeomorphic Registration
  for Images and Surfaces*, Medical Image Analysis, 2019.
- Pombo et al., *Equitable Modelling of Brain Imaging by Counterfactual
  Augmentation with Morphologically Constrained 3D Deep Generative Models*,
  Medical Image Analysis, 2023.
- Peng et al., *Accurate Brain Age Prediction with Lightweight Deep Neural
  Networks*, Medical Image Analysis, 2021.

## Citation

```bibtex
@inproceedings{zhang2027brainagingspeed,
  title={Interpretable Local-to-Global Estimation of Brain Aging Speed From Morphological Changes Using Longitudinal Structural MRI Data},
  author={Zhang, Yuanwang and Li, Hongming and Fan, Yong},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  series={Lecture Notes in Computer Science},
  volume={16887},
  pages={323--332},
  year={2027},
  doi={10.1007/978-3-032-38172-9_31}
}
```
