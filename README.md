# Frame-Level Phoneme Recognition with an MLP

Course project for **Deep Learning (E-JUST)**, based on the Kaggle competition
**11-785 HW1P2 (Fall 2024)** from Carnegie Mellon's Introduction to Deep Learning course.

For every 10 ms frame of speech (28 MFCC features), the model predicts which of **40 phoneme classes**
is being spoken. The model is a plain multilayer perceptron that sees each frame together with its
neighbours (a context window).

**Result: 84.55% frame accuracy on the dev set** with a 10.5M-parameter MLP (the limit is 20M).

## Approach

1. **CMVN:** normalise each recording's MFCCs to mean 0 and std 1.
2. **Context window:** pad every recording with K = 20 zero frames and feed the MLP the 41 frames around
   each frame (41 × 28 = 1148 inputs).
3. **Model:** a pyramid MLP 2048-2048-1024-1024-512-512. Each hidden layer is
   Linear → BatchNorm → GELU → Dropout 0.2.
4. **Training:** cross-entropy loss, AdamW (lr 1e-3, weight decay 0.01), cosine learning-rate schedule,
   mixed precision, batch size 4096, 20 epochs (about 68 min on an RTX 4050 Laptop GPU).

## Ablation study (3 epochs per step, each step adds one change)

| Step | Change | Params | Dev acc |
|---|---|---|---|
| 1 | Baseline: K=5, 2×512, ReLU, Adam | 0.44M | 68.06% |
| 2 | + CMVN | 0.44M | 71.36% |
| 3 | + context K=20 | 0.87M | 75.24% |
| 4 | + pyramid MLP + BatchNorm | 10.5M | 81.28% |
| 5 | + Dropout 0.2 | 10.5M | 81.74% |
| 6 | + GELU | 10.5M | 82.55% |
| 7 | + AdamW + cosine schedule | 10.5M | 82.90% |
| **Final** | step 7 trained for 20 epochs | 10.5M | **84.55%** |

## Repository contents

| Path | Contents |
|---|---|
| `phoneme_recognition_mlp.ipynb` | the notebook, with all outputs |
| `report/report.pdf` | the project report (LaTeX source in `report/report.tex`) |
| `results/` | figures and result tables (ablation, per-phoneme accuracy, confusions, summary) |

## How to run

1. Install the requirements: Python 3.11, PyTorch 2.6 with CUDA, NumPy, pandas, matplotlib,
   scikit-learn, tqdm. A CUDA GPU with at least 6 GB of memory and 16 GB of RAM are needed.
2. Download the competition data from Kaggle and extract it to `dataset/11785-f24-hw1p2/`.
3. Open `phoneme_recognition_mlp.ipynb` and click **Run All**. Set `DEBUG = True` first for a quick test.

The notebook creates `cache/` (merged data), `runs/` (one folder per training run, with the weights of
every epoch), `results/` and `submission.csv`. These are not in the repository because of their size.

## Author

Yassin Mahmoud Alam Elden Abdelaziz
