# AI Age Recognition

This project is a small **PyTorch-based age range classifier** trained on face images.  
Given a face image, the model predicts one of five age ranges:

- `18-20`
- `21-30`
- `31-40`
- `41-50`
- `51-60`

All of the core logic lives in a single script, `main.py`, which loads the data, builds a simple fully connected neural network, trains it, and reports the training and test loss.

## Project Structure

- `main.py` – End‑to‑end training + evaluation script.
  - Loads the CSV metadata.
  - Loads and preprocesses face images.
  - Defines and trains a small neural network (`NN_Regression`).
  - Prints train and test losses at the end.
- `archive/`
  - `age_detection.csv` – Metadata file with image filenames, split (`train` / `test`), and age range labels.
  - Image files referenced in the CSV (faces used for training/testing).
- `requirements.txt` – Python dependencies.
- `LICENSE` – MIT License.
- `.gitignore`, `.DS_Store` – Repo housekeeping files.

## How It Works

### Data

The script expects:

- CSV file: `archive/age_detection.csv`
- Columns:
  - `file` – Relative path to the image file inside `archive/`.
  - `split` – Either `"train"` or `"test"`.
  - `age` – One of the defined age ranges (`18-20`, `21-30`, `31-40`, `41-50`, `51-60`).

Images are:

- Loaded from `archive/<file>`.
- Transposed according to EXIF orientation.
- Converted to RGB.
- Resized to `64 x 64`.
- Normalized to `[0, 1]` and flattened into a 1D vector.

Age labels are mapped to integer class indices:

```python
AGE_CLASSES = ["18-20", "21-30", "31-40", "41-50", "51-60"]
AGE_TO_INDEX = {label: i for i, label in enumerate(AGE_CLASSES)}
```

### Model

Training settings (as defined in `main.py`):

- `NUM_EPOCHS = 100`
- `LEARNING_RATE = 1e-3`
- `BATCH_SIZE = 5`
- `EPOCH_CHECK_INTERVAL = 50`
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Optional loss curve plotting with `matplotlib`.

## Getting Started

### Prerequisites

- Python 3.8+ (recommended)
- `pip` (or `conda`)
- A working C/C++ build toolchain may be needed for some PyTorch builds.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cgarciapicazo/AI-AgeRecognition.git
   cd AI-AgeRecognition
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure the dataset is in place:

   - `archive/age_detection.csv`
   - Image files referenced in `archive/age_detection.csv` under the `archive/` folder.

## Usage

The project currently uses a **single script** that trains the model and then evaluates it on both the training and test splits.

Run:

```bash
python main.py
```

The script will:

1. Load training and test examples from `archive/age_detection.csv`.
2. Convert images to flattened tensors.
3. Train the `NN_Regression` model for `NUM_EPOCHS`.
4. Optionally display a loss graph (every `EPOCH_CHECK_INTERVAL` epochs) if `SHOW_GRAPH = True`.
5. Print the final training and test loss:

```text
<train_loss_value> <test_loss_value>
```

### Visualizing Individual Images (optional)

You can uncomment and adjust this block inside `__main__` to inspect samples:

```python
# for i in range(50, 53):
#     display_picture(X_train[i], label=y_train[i])
```

## Configuration

You can tweak the main hyperparameters at the top of `main.py`:

```python
IMG_SIZE = (64, 64)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 5
EPOCH_CHECK_INTERVAL = 50
SHOW_GRAPH = True
```

Tuning these parameters will change training duration and model performance.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

- **GitHub:** [@cgarciapicazo](https://github.com/cgarciapicazo)
- **Repository:** [AI-AgeRecognition](https://github.com/cgarciapicazo/AI-AgeRecognition)
