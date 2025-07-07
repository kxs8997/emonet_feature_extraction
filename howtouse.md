# How to Use: EmoNet Feature Extraction for AFEW-VA

This project provides scripts to extract deep features from the AFEW-VA dataset using EmoNet, saving them as `.npz` files with a manifest for easy sharing and reproducibility.

## 1. Setup

- Clone this repository.
- Download the AFEW-VA dataset (see download instructions or use `download_afew_va.py`).
- Create and activate the Python virtual environment:
  ```bash
  python3 -m venv emonet_env
  source emonet_env/bin/activate
  pip install -r requirements.txt  # or manually: kagglehub numpy torch scikit-image opencv-python
  ```
- Download the EmoNet pretrained weights (`pretrained/emonet_8.pth` or `pretrained/emonet_5.pth`) and place them in the `pretrained/` directory.

## 2. Extract Features

To extract features for the entire dataset:

```bash
python extract_features_all.py
```

- Features will be saved in the `features/` folder, mirroring the dataset structure (e.g., `features/001/00000.npz`).
- The manifest file `features_manifest.json` will map each feature file to its ground truth arousal and valence.

## 3. Manifest Example

```json
{
  "features/001/00000.npz": {"arousal": 5.0, "valence": 0.0},
  ...
}
```

## 4. Loading a Single Feature File

Each `.npz` file contains a single array named `feature` with shape `(256,)` (float32), corresponding to the deep features from EmoNet's penultimate layer.

**Example (Python):**
```python
import numpy as np
feat = np.load('features/001/00000.npz')['feature']
print(feat.shape)  # (256,)
```

- These features can be used for downstream tasks (e.g., regression, clustering, transfer learning).

## 5. Sharing and Reproducibility

- Only the `features/` folder and `features_manifest.json` are needed to share the extracted features.
- The scripts use relative paths for portability.
- The `features/` folder is gitignored by default (see `.gitignore`).

## 5. Custom Extraction

- To extract features for a subset, edit `extract_features_subset.py` and adjust `NUM_VIDEOS` and `FRAMES_PER_VIDEO`.

## 6. Requirements

- Python 3.8+
- torch, numpy, scikit-image, opencv-python, kagglehub

## 7. Citing EmoNet
If you use this code or features, please cite:

> Toisoul, A., Kossaifi, J., Bulat, A., Tzimiropoulos, G., & Pantic, M. (2021). Estimation of continuous valence and arousal levels from faces in naturalistic conditions. Nature Machine Intelligence.

---

**Contact:** For questions or issues, open an issue on GitHub or contact the repository maintainer.
