# Lab Assignment 1 — Gaussian Naïve Bayes on the Iris Dataset

**Name:** Ghanshyam Ghimire  
**University:** Kathmandu University — BTech in Artificial Intelligence, 4th Semester  
**Subject:** Introduction to Machine Learning  
**Instructor:** Sandeep Gupta  

---

## Objective

Implement Gaussian Naïve Bayes **from scratch** (no `sklearn.naive_bayes`) on the Iris dataset (3 classes), and evaluate it with a confusion matrix, per-class precision/recall/F1, and macro-averaged metrics.

---

## Repository Structure

```
ml-lab-assignment1/
├── iris_dataset.csv                         full 150-sample Fisher Iris dataset
├── lab_assignment_1_naive_bayes_iris.ipynb  notebook with pre-run outputs
├── README.md
└── webapp/                                  browser-based notebook viewer
    ├── index.html
    ├── style.css
    ├── app.js
    └── vercel.json
```

---

## Results

| Metric | Value |
|---|---|
| Test set size | 30 samples (80/20 split, stratified) |
| Overall Accuracy | **93.33%** (28 / 30 correct) |

**Per-class metrics:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| setosa | 1.0000 | 1.0000 | 1.0000 |
| versicolor | 0.9000 | 0.9000 | 0.9000 |
| virginica | 0.9000 | 0.9000 | 0.9000 |
| **macro avg** | **0.9333** | **0.9333** | **0.9333** |

**Confusion matrix** (rows = actual, cols = predicted):

|  | setosa | versicolor | virginica |
|---|---|---|---|
| setosa | 10 | 0 | 0 |
| versicolor | 0 | 9 | 1 |
| virginica | 0 | 1 | 9 |

The 2 misclassifications are between versicolor and virginica, which is expected — these two classes overlap significantly in feature space.

---

## How It Works

### Gaussian Naïve Bayes (from scratch)

The classifier is implemented entirely in NumPy. No `sklearn.naive_bayes` is used anywhere.

**Training (`fit`)** — for each class c:
- Compute mean and variance of each feature
- Compute log-prior: `log(Nc / N)`

**Inference (`predict`)** — for each sample x, pick the class with the highest log-posterior:

```
log P(y=c | x)  ∝  log P(y=c)  +  Σ log N(xi ; μci, σ²ci)
```

where `log N(x; μ, σ²) = -0.5 * log(2πσ²) - (x−μ)² / (2σ²)`

A small epsilon (`1e-9`) is added to each variance to avoid division by zero.

---

## Running the Notebook

**Option 1 — Jupyter / JupyterLab**

```bash
pip install numpy scikit-learn jupyter
jupyter notebook lab_assignment_1_naive_bayes_iris.ipynb
```

**Option 2 — Google Colab**

Upload `lab_assignment_1_naive_bayes_iris.ipynb` directly. All outputs are already stored so you can read without re-running.

---

## Running the Webapp Locally

The webapp is a static site — no build step needed.

```bash
cd webapp
python3 -m http.server 8080
```

Open `http://localhost:8080` in your browser.

> Do not open `index.html` directly as a `file://` URL — Pyodide (WebAssembly) requires an HTTP origin to load correctly.

### Features

- Jupyter-style UI with `In [n]:` / `Out [n]:` cell labels
- All outputs pre-loaded on page open — nothing looks empty
- Each code cell is re-runnable in-browser via **Pyodide** (Python 3 + NumPy + scikit-learn, no backend)
- **Shift+Enter** runs the focused cell
- Toolbar buttons:
  - `Run All` — executes all cells top to bottom
  - `Restart Kernel` — clears Pyodide state and resets counters
  - `↓ iris_dataset.csv` — downloads the dataset
  - `↓ naive_bayes.py` — downloads all code as a `.py` file
  - `↓ Download .ipynb` — downloads a valid notebook with outputs
  - `GitHub` — links to this repository

---

## Dependencies

The notebook and webapp use only:

- `numpy` — all classifier math
- `sklearn.datasets.load_iris` — data loading only
- `sklearn.model_selection.train_test_split` — data splitting only

The classifier itself has zero sklearn dependency.
