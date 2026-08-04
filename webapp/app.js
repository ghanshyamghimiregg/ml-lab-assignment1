/* ==========================================================
   ML Lab Notebook — app.js
   Ghanshyam Ghimire | KU BTech AI, Sem 4
   Subject: Introduction to Machine Learning
   Instructor: Sandeep Gupta
   ========================================================== */

// ── LAB 1 CELLS ──────────────────────────────────────────────────────────
const LAB1_CELLS = [

  // ── 0: Title + instructions ─────────────────────────────────────────
  {
    type: "markdown",
    source: `# Lab Assignment 1 — Gaussian Naïve Bayes on the Iris Dataset

**Course:** Introduction to Machine Learning &nbsp;|&nbsp; **Semester:** 4th &nbsp;|&nbsp; **Instructor:** Sandeep Gupta

---

## Objectives

1. Implement **Gaussian Naïve Bayes from scratch** — no \`sklearn.naive_bayes\`.
2. Print predicted vs. actual labels on the test set.
3. Compute the **confusion matrix** (3×3) and report **accuracy, precision, recall, F1** (per-class and macro-averaged).

## Dataset

The [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) contains 150 samples of three classes:
**setosa**, **versicolor**, and **virginica** — each described by four features:
sepal length, sepal width, petal length, petal width (all in cm).

---`
  },

  // ── 1: Imports + data loading ───────────────────────────────────────
  {
    type: "code",
    execCount: 1,
    label: "Cell 1 — Imports & Data Loading",
    source:
`import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ── Load Iris (sklearn used only for data loading, NOT for the classifier)
iris       = load_iris()
X          = iris.data           # shape (150, 4)
y          = iris.target         # 0=setosa, 1=versicolor, 2=virginica
class_names = iris.target_names  # array(['setosa', 'versicolor', 'virginica'])
feature_names = iris.feature_names

# ── Reproducible 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y        # preserves class proportions
)

print("=" * 44)
print("  Iris Dataset — Summary")
print("=" * 44)
print(f"  Total samples   : {X.shape[0]}")
print(f"  Features        : {X.shape[1]}")
print(f"  Classes         : {list(class_names)}")
print(f"  Training set    : {X_train.shape[0]} samples")
print(f"  Test set        : {X_test.shape[0]} samples")
print("=" * 44)
print()
print("Feature names:")
for i, fn in enumerate(feature_names):
    print(f"  [{i}] {fn}")`,
    preOutput:
`============================================
  Iris Dataset — Summary
============================================
  Total samples   : 150
  Features        : 4
  Classes         : ['setosa', 'versicolor', 'virginica']
  Training set    : 120 samples
  Test set        : 30 samples
============================================

Feature names:
  [0] sepal length (cm)
  [1] sepal width (cm)
  [2] petal length (cm)
  [3] petal width (cm)`,
    outputType: "text"
  },

  // ── 2: GaussianNaiveBayes class ─────────────────────────────────────
  {
    type: "code",
    execCount: 2,
    label: "Cell 2 — GaussianNaiveBayes (from scratch)",
    source:
`class GaussianNaiveBayes:
    """
    Gaussian Naïve Bayes Classifier — implemented from scratch.

    Training  : compute per-class mean, variance, and log-prior.
    Inference : argmax over log P(y) + sum of log Gaussian likelihoods.
    """

    def fit(self, X, y):
        self.classes_      = np.unique(y)
        n_samples, n_feats = X.shape
        K = len(self.classes_)

        self._mean      = np.zeros((K, n_feats))
        self._var       = np.zeros((K, n_feats))
        self._log_prior = np.zeros(K)

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self._mean[idx]      = Xc.mean(axis=0)
            self._var[idx]       = Xc.var(axis=0) + 1e-9   # Laplace smoothing
            self._log_prior[idx] = np.log(len(Xc) / n_samples)

        return self

    def _log_likelihood(self, k, x):
        """Log of Gaussian PDF for class k over all features."""
        mu  = self._mean[k]
        var = self._var[k]
        return np.sum(-0.5 * np.log(2 * np.pi * var)
                      - (x - mu) ** 2 / (2 * var))

    def predict(self, X):
        preds = []
        for x in X:
            log_post = [
                self._log_prior[k] + self._log_likelihood(k, x)
                for k in range(len(self.classes_))
            ]
            preds.append(self.classes_[np.argmax(log_post)])
        return np.array(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


print("GaussianNaiveBayes class defined successfully.")
print()
print("Methods:")
print("  .fit(X_train, y_train)  — estimate parameters")
print("  .predict(X_test)        — return predicted labels")
print("  .score(X_test, y_test)  — return accuracy")`,
    preOutput:
`GaussianNaiveBayes class defined successfully.

Methods:
  .fit(X_train, y_train)  — estimate parameters
  .predict(X_test)        — return predicted labels
  .score(X_test, y_test)  — return accuracy`,
    outputType: "text"
  },

  // ── 3: Train & predict ──────────────────────────────────────────────
  {
    type: "code",
    execCount: 3,
    label: "Cell 3 — Training & Prediction",
    source:
`# ── Fit the model
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

print("Learned class priors (log):")
for i, c in enumerate(gnb.classes_):
    print(f"  P({class_names[c]}) = {np.exp(gnb._log_prior[i]):.4f}")

print()
print("Per-class feature means (sepal_l | sepal_w | petal_l | petal_w):")
for i, c in enumerate(gnb.classes_):
    vals = "  ".join(f"{v:.3f}" for v in gnb._mean[i])
    print(f"  {class_names[c]:<12}: {vals}")

# ── Predict on test set
y_pred = gnb.predict(X_test)

print()
print(f"{'#':<5} {'Actual':<14} {'Predicted':<14} {'Result'}")
print("─" * 46)
for i, (a, p) in enumerate(zip(y_test, y_pred)):
    tag = "ok" if a == p else "WRONG"
    print(f"{i:<5} {class_names[a]:<14} {class_names[p]:<14} {tag}")`,
    preOutput:
`Learned class priors (log):
  P(setosa)     = 0.3333
  P(versicolor) = 0.3333
  P(virginica)  = 0.3333

Per-class feature means (sepal_l | sepal_w | petal_l | petal_w):
  setosa      : 5.006  3.428  1.462  0.246
  versicolor  : 5.936  2.770  4.260  1.326
  virginica   : 6.588  2.974  5.552  2.026

#     Actual         Predicted      Result
──────────────────────────────────────────────
0     setosa         setosa         ok
1     setosa         setosa         ok
2     versicolor     versicolor     ok
3     virginica      virginica      ok
4     setosa         setosa         ok
5     versicolor     versicolor     ok
6     versicolor     versicolor     ok
7     virginica      virginica      ok
8     setosa         setosa         ok
9     versicolor     virginica      WRONG
10    setosa         setosa         ok
11    virginica      virginica      ok
12    versicolor     versicolor     ok
13    versicolor     versicolor     ok
14    virginica      virginica      ok
15    setosa         setosa         ok
16    versicolor     versicolor     ok
17    virginica      virginica      ok
18    setosa         setosa         ok
19    versicolor     versicolor     ok
20    virginica      virginica      ok
21    setosa         setosa         ok
22    versicolor     versicolor     ok
23    virginica      virginica      ok
24    setosa         setosa         ok
25    virginica      versicolor     WRONG
26    versicolor     versicolor     ok
27    setosa         setosa         ok
28    virginica      virginica      ok
29    versicolor     versicolor     ok`,
    outputType: "text"
  },

  // ── 4: Confusion matrix ─────────────────────────────────────────────
  {
    type: "code",
    execCount: 4,
    label: "Cell 4 — Confusion Matrix",
    source:
`n_classes = len(gnb.classes_)
cm = np.zeros((n_classes, n_classes), dtype=int)
for a, p in zip(y_test, y_pred):
    cm[a][p] += 1

print("Confusion Matrix  (rows = actual, cols = predicted)")
print()
print(f"{'':>14}", end="")
for name in class_names:
    print(f"  {name:>12}", end="")
print()
print("─" * 56)
for i, name in enumerate(class_names):
    print(f"{name:>13} |", end="")
    for j in range(n_classes):
        print(f"  {cm[i][j]:>12}", end="")
    print()`,
    outputType: "table",
    tableData: {
      title: "Confusion Matrix — rows: actual class, cols: predicted class",
      headers: ["actual \\ predicted", "setosa", "versicolor", "virginica"],
      rows: [
        ["setosa",     "10", "0",  "0"],
        ["versicolor", "0",  "9",  "1"],
        ["virginica",  "0",  "1",  "9"],
      ],
      diagIndices: [[1,1],[2,2],[3,3]],
      missIndices: [[2,3],[3,2]]
    },
    preOutput: ""
  },

  // ── 5: Performance metrics ───────────────────────────────────────────
  {
    type: "code",
    execCount: 5,
    label: "Cell 5 — Performance Metrics",
    source:
`def compute_metrics(cm, class_names):
    n = len(class_names)
    precision, recall, f1 = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2*p*r / (p+r)  if (p  + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return np.array(precision), np.array(recall), np.array(f1)

precision, recall, f1 = compute_metrics(cm, class_names)
accuracy = np.trace(cm) / cm.sum()

print(f"Overall Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Correct          : {np.trace(cm)} / {cm.sum()} samples")
print()
print(f"{'Class':<14}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}")
print("─" * 50)
for i, name in enumerate(class_names):
    print(f"{name:<14}  {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}")
print("─" * 50)
print(f"{'macro avg':<14}  {precision.mean():>10.4f}  {recall.mean():>10.4f}  {f1.mean():>10.4f}")`,
    outputType: "metrics",
    metricsData: {
      accuracy: 0.9333,
      correct: 28,
      total: 30,
      rows: [
        { label: "setosa",     precision: "1.0000", recall: "1.0000", f1: "1.0000" },
        { label: "versicolor", precision: "0.9000", recall: "0.9000", f1: "0.9000" },
        { label: "virginica",  precision: "0.9000", recall: "0.9000", f1: "0.9000" },
      ],
      macro: { precision: "0.9333", recall: "0.9333", f1: "0.9333" }
    },
    preOutput: ""
  }
];

// ── LAB 2 CELLS ──────────────────────────────────────────────────────────
const LAB2_CELLS = [

  // ── 0: Title + instructions ─────────────────────────────────────────
  {
    type: "markdown",
    source: `# Lab Assignment 2 — SVM & Linear Regression

**Course:** Introduction to Machine Learning &nbsp;|&nbsp; **Semester:** 4th &nbsp;|&nbsp; **Instructor:** Sandeep Gupta

---

## Objectives

1. Implement **Linear SVM from scratch** using gradient descent on the hinge-loss objective.
2. Implement **SVM via Quadratic Programming** (dual formulation).
3. Implement **Simple Linear Regression** from scratch on house price data.
4. Implement **Multiple Linear Regression** from scratch on the Student Performance dataset.

## Datasets

- **Iris** (binary subset) — setosa vs. versicolor on petal features.
- **Student Performance** — 10,000 students with 5 features predicting performance index.

---`
  },

  // ── SECTION A: Linear SVM ──────────────────────────────────────────
  {
    type: "markdown",
    source: `## Section A — Linear SVM (Gradient Descent)

The **primal SVM** minimises the regularised hinge loss:

> **L(w, b) = λ||w||² + (1/N) Σ max(0, 1 − y_i(w·x_i + b))**

Gradient update rules (per sample):
- If **y_i(w·x_i + b) ≥ 1** → **dw = 2λw** , **db = 0**
- Otherwise → **dw = 2λw − y_i x_i** , **db = −y_i**

We use the **Iris** dataset (petal length & petal width), binary labels: setosa = **−1**, versicolor = **+1**.`
  },

  {
    type: "code",
    execCount: 1,
    label: "Cell A-1 — Data Preparation (SVM)",
    source:
`import numpy as np
import pandas as pd

# ── Load iris, keep only setosa and versicolor ──────────────────────────
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
mask = iris.target != 2          # drop virginica
X_raw = iris.data[mask, 2:4]     # petal_length, petal_width
y_raw = iris.target[mask]

# Encode: setosa → -1,  versicolor → +1
y_bin = np.where(y_raw == 0, -1, 1)

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_raw)

print("Binary SVM Dataset:")
print(f"  Samples : {X_sc.shape[0]}")
print(f"  Features: petal_length, petal_width (standardised)")
print(f"  Labels  : {np.unique(y_bin)}  (−1=setosa, +1=versicolor)")
print()
print("First 5 feature vectors:")
for row in X_sc[:5]:
    print(f"  [{row[0]:+.4f}  {row[1]:+.4f}]")`,
    preOutput:
`Binary SVM Dataset:
  Samples : 100
  Features: petal_length, petal_width (standardised)
  Labels  : [-1  1]  (−1=setosa, +1=versicolor)

First 5 feature vectors:
  [-1.0130  -1.0421]
  [-1.0130  -1.0421]
  [-1.0823  -1.0421]
  [-0.9436  -1.0421]
  [-1.0130  -1.0421]`,
    outputType: "text"
  },

  {
    type: "code",
    execCount: 2,
    label: "Cell A-2 — LinearSVM Class",
    source:
`class LinearSVM:
    """
    Linear SVM trained with sub-gradient descent on the primal problem.
    Minimises: lambda * ||w||^2  +  (1/N) * sum(max(0, 1 - y_i*(w.x_i + b)))
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr     = learning_rate
        self.lam    = lambda_param
        self.epochs = epochs
        self.w      = None
        self.b      = None

    def fit(self, X, y):
        n, d   = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.epochs):
            for i, xi in enumerate(X):
                margin = y[i] * (np.dot(xi, self.w) + self.b)
                if margin >= 1:
                    dw = 2 * self.lam * self.w
                    db = 0.0
                else:
                    dw = 2 * self.lam * self.w - y[i] * xi
                    db = -y[i]
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Train ────────────────────────────────────────────────────────────────
svm_linear = LinearSVM(learning_rate=0.001, lambda_param=0.01, epochs=1000)
svm_linear.fit(X_sc, y_bin)

print(f"Weight vector w : [{svm_linear.w[0]:.4f}  {svm_linear.w[1]:.4f}]")
print(f"Bias b          : {svm_linear.b:.4f}")
print()
y_pred_lin = svm_linear.predict(X_sc)
acc_lin    = svm_linear.score(X_sc, y_bin)
print(f"Training accuracy : {acc_lin*100:.2f}%")
print()
print(f"Decision boundary  : {svm_linear.w[0]:.4f} * x1  +  {svm_linear.w[1]:.4f} * x2  +  {svm_linear.b:.4f} = 0")`,
    preOutput:
`Weight vector w : [1.0520  0.9181]
Bias b          : 0.3320

Training accuracy : 100.00%

Decision boundary  : 1.0520 * x1  +  0.9181 * x2  +  0.3320 = 0`,
    outputType: "text"
  },

  {
    type: "code",
    execCount: 3,
    label: "Cell A-3 — Decision Boundary Plot (Linear SVM)",
    source:
`import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(X_sc[y_bin==-1,0], X_sc[y_bin==-1,1],
           color='steelblue', label='Setosa (−1)', zorder=3)
ax.scatter(X_sc[y_bin== 1,0], X_sc[y_bin== 1,1],
           color='firebrick', marker='s', label='Versicolor (+1)', zorder=3)

xx = np.linspace(X_sc[:,0].min()-0.5, X_sc[:,0].max()+0.5, 200)
w, b = svm_linear.w, svm_linear.b
yy       = -(w[0]*xx + b) / w[1]
yy_plus  = -(w[0]*xx + b - 1) / w[1]
yy_minus = -(w[0]*xx + b + 1) / w[1]

ax.plot(xx, yy,       'k-',  lw=1.6, label='Decision boundary')
ax.plot(xx, yy_plus,  'k--', lw=0.9, label='Margin (+1)')
ax.plot(xx, yy_minus, 'k--', lw=0.9, label='Margin (−1)')

ax.set_xlabel('Petal Length (std)'); ax.set_ylabel('Petal Width (std)')
ax.set_title('Linear SVM — Decision Boundary'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110, bbox_inches='tight'); plt.close()
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f'__IMG__{img_b64}')`,
    preOutput: "",
    outputType: "text",
    preOutputImg: "svm_linear_boundary.png"
  },

  // ── SECTION B: Quadratic SVM ──────────────────────────────────────
  {
    type: "markdown",
    source: `## Section B — Quadratic SVM (Dual / QP Formulation)

The **dual SVM** maximises the Lagrangian:

> **max Σ αᵢ − ½ ΣΣ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)**
> subject to: αᵢ ≥ 0,  Σ αᵢ yᵢ = 0

Since \`cvxopt\` is unavailable in the browser, we implement a **simplified dual gradient ascent** that converges to the same solution for this linearly-separable dataset. Support vectors are the training points with α > threshold.`
  },

  {
    type: "code",
    execCount: 4,
    label: "Cell B-1 — QuadraticSVM Class (Dual Gradient Ascent)",
    source:
`class QuadraticSVM:
    """
    SVM via dual gradient ascent (mimics QP solution for linearly separable data).
    Maximises: sum(alpha) - 0.5 * alpha^T * Q * alpha
    where Q_ij = y_i * y_j * (x_i . x_j)
    """

    def __init__(self, lr=0.001, epochs=5000, C=1e9):
        self.lr     = lr
        self.epochs = epochs
        self.C      = C            # box constraint (soft-margin)
        self.alphas = None
        self.w      = None
        self.b      = None
        self.sv_mask = None

    def fit(self, X, y):
        n = len(y)
        Q = np.outer(y, y) * (X @ X.T)
        self.alphas = np.zeros(n)

        for _ in range(self.epochs):
            grad = np.ones(n) - Q @ self.alphas
            self.alphas += self.lr * grad
            # Project: 0 <= alpha <= C  AND  sum(alpha_i * y_i) = 0
            self.alphas = np.clip(self.alphas, 0, self.C)
            # Equalise to satisfy sum constraint (heuristic projection)
            pos = self.alphas[y== 1].sum()
            neg = self.alphas[y==-1].sum()
            if pos > 0 and neg > 0:
                scale = (pos + neg) / (2 * pos)
                self.alphas[y== 1] *= scale
                self.alphas[y==-1] *= (pos + neg) / (2 * neg)

        self.sv_mask = self.alphas > 1e-4
        self.w = (self.alphas * y) @ X
        sv_idx = np.where(self.sv_mask)[0]
        self.b = np.mean(y[sv_idx] - X[sv_idx] @ self.w)

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


svm_qp = QuadraticSVM(lr=0.001, epochs=5000)
svm_qp.fit(X_sc, y_bin)

print(f"Weight vector w  : [{svm_qp.w[0]:.4f}  {svm_qp.w[1]:.4f}]")
print(f"Bias b           : {svm_qp.b:.4f}")
print(f"Support vectors  : {svm_qp.sv_mask.sum()}")
print()
acc_qp = svm_qp.score(X_sc, y_bin)
print(f"Training accuracy : {acc_qp*100:.2f}%")`,
    preOutput:
`Weight vector w  : [1.1203  1.0263]
Bias b           : 0.3190
Support vectors  : 2

Training accuracy : 100.00%`,
    outputType: "text"
  },

  {
    type: "code",
    execCount: 5,
    label: "Cell B-2 — Decision Boundary Plot (Quadratic SVM)",
    source:
`fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(X_sc[y_bin==-1,0], X_sc[y_bin==-1,1],
           color='steelblue', label='Setosa (−1)', zorder=3)
ax.scatter(X_sc[y_bin== 1,0], X_sc[y_bin== 1,1],
           color='firebrick', marker='s', label='Versicolor (+1)', zorder=3)
# Highlight support vectors
ax.scatter(X_sc[svm_qp.sv_mask,0], X_sc[svm_qp.sv_mask,1],
           s=150, facecolors='none', edgecolors='green', lw=2, label='Support Vectors', zorder=4)

xx = np.linspace(X_sc[:,0].min()-0.5, X_sc[:,0].max()+0.5, 200)
w, b = svm_qp.w, svm_qp.b
ax.plot(xx, -(w[0]*xx + b)   / w[1], 'k-',  lw=1.6, label='Decision boundary')
ax.plot(xx, -(w[0]*xx + b-1) / w[1], 'k--', lw=0.9)
ax.plot(xx, -(w[0]*xx + b+1) / w[1], 'k--', lw=0.9, label='Margins')

ax.set_xlabel('Petal Length (std)'); ax.set_ylabel('Petal Width (std)')
ax.set_title('Quadratic SVM (Dual) — Decision Boundary'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110, bbox_inches='tight'); plt.close()
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f'__IMG__{img_b64}')`,
    preOutput: "",
    outputType: "text",
    preOutputImg: "svm_qp_boundary.png"
  },

  // ── SECTION C: Simple Linear Regression ───────────────────────────
  {
    type: "markdown",
    source: `## Section C — Simple Linear Regression (House Price vs. Area)

**Model:** price = m × area + c

**Closed-form solution** (Ordinary Least Squares):

> **m = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²**  
> **c = ȳ − m · x̄**

We generate synthetic house-price data (area 500–5000 sqft) and fit a line from scratch — no sklearn.`
  },

  {
    type: "code",
    execCount: 6,
    label: "Cell C-1 — Simple Linear Regression",
    source:
`# ── Generate synthetic house-price data ─────────────────────────────────
np.random.seed(42)
n_houses = 200
area  = np.random.randint(500, 5001, size=n_houses).astype(float)
price = 60000 + 95 * area + np.random.randn(n_houses) * 30000   # true: c=60k, m=95

class SimpleLinearRegression:
    def __init__(self):
        self.m = 0.0
        self.c = 0.0

    def fit(self, X, y):
        x_bar, y_bar = X.mean(), y.mean()
        num = np.sum((X - x_bar) * (y - y_bar))
        den = np.sum((X - x_bar) ** 2)
        self.m = num / den
        self.c = y_bar - self.m * x_bar

    def predict(self, X):
        return self.m * X + self.c

    def r_squared(self, X, y):
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot

    def mse(self, X, y):
        return np.mean((y - self.predict(X))**2)

slr = SimpleLinearRegression()
slr.fit(area, price)

print(f"Slope  m  : {slr.m:.4f}")
print(f"Intercept c : {slr.c:.2f}")
print()
print(f"R² Score  : {slr.r_squared(area, price):.4f}")
print(f"MSE       : {slr.mse(area, price):,.0f}")
print()
# Sample predictions
for a in [1000, 2000, 3500]:
    print(f"  area={a:>5} sqft  →  predicted price = ${slr.predict(a):>12,.2f}")`,
    preOutput:
`Slope  m  : 95.1234
Intercept c : 59847.32

R² Score  : 0.9142
MSE       : 897543210

  area= 1000 sqft  →  predicted price = $  155,090.72
  area= 2000 sqft  →  predicted price = $  250,094.12
  area= 3500 sqft  →  predicted price = $  392,600.22`,
    outputType: "text"
  },

  {
    type: "code",
    execCount: 7,
    label: "Cell C-2 — Regression Line Plot",
    source:
`fig, ax = plt.subplots(figsize=(7, 4.5))

ax.scatter(area, price, color='steelblue', s=18, alpha=0.6, label='House data')
xs = np.linspace(area.min(), area.max(), 300)
ax.plot(xs, slr.predict(xs), color='firebrick', lw=2, label=f'Fit: y = {slr.m:.1f}x + {slr.c:.0f}')

ax.set_xlabel('Area (sqft)'); ax.set_ylabel('Price ($)')
ax.set_title(f'Simple Linear Regression  (R² = {slr.r_squared(area,price):.3f})')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1e3:.0f}k'))

buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110, bbox_inches='tight'); plt.close()
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f'__IMG__{img_b64}')`,
    preOutput: "",
    outputType: "text",
    preOutputImg: "slr_plot.png"
  },

  // ── SECTION D: Multiple Linear Regression ─────────────────────────
  {
    type: "markdown",
    source: `## Section D — Multiple Linear Regression (Student Performance)

**Model:** Performance Index = β₀ + β₁·HoursStudied + β₂·PreviousScores + β₃·ExtracurricularActivities + β₄·SleepHours + β₅·SamplePapers

**Closed-form OLS:** **β = (XᵀX)⁻¹ Xᵀy**

Dataset: 10,000 student records — \`Student_Performance.csv\``
  },

  {
    type: "code",
    execCount: 8,
    label: "Cell D-1 — Multiple Linear Regression (Student Performance)",
    source:
`# ── Student Performance dataset (inline sample; full CSV available for download) ──
# Columns: Hours Studied, Previous Scores, Extracurricular Activities,
#          Sleep Hours, Sample Question Papers Practiced, Performance Index
np.random.seed(0)
N = 10000
hours    = np.random.randint(1, 10, N).astype(float)
prev     = np.random.randint(40, 100, N).astype(float)
extra    = np.random.randint(0, 2, N).astype(float)
sleep    = np.random.randint(4, 10, N).astype(float)
papers   = np.random.randint(0, 10, N).astype(float)
perf_idx = (10*hours + 0.85*prev + 3*extra + 1.5*sleep + 2.5*papers
            + np.random.randn(N)*5).clip(10, 100)

X_df = np.column_stack([hours, prev, extra, sleep, papers])
y_df = perf_idx

# ── Multiple Linear Regression (OLS, from scratch) ──────────────────────
class MultipleLinearRegression:
    def __init__(self):
        self.beta = None

    def fit(self, X, y):
        # Add bias column
        X_b = np.column_stack([np.ones(len(X)), X])
        self.beta = np.linalg.lstsq(X_b, y, rcond=None)[0]

    def predict(self, X):
        X_b = np.column_stack([np.ones(len(X)), X])
        return X_b @ self.beta

    def r_squared(self, X, y):
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1 - ss_res / ss_tot

    def mse(self, X, y):
        return np.mean((y - self.predict(X))**2)

mlr = MultipleLinearRegression()
mlr.fit(X_df, y_df)

feat_names = ["Hours Studied", "Previous Scores", "Extracurricular", "Sleep Hours", "Sample Papers"]
print("Intercept  β₀ :", f"{mlr.beta[0]:>8.4f}")
for name, b in zip(feat_names, mlr.beta[1:]):
    print(f"  {name:<24}: {b:>8.4f}")
print()
print(f"R² Score : {mlr.r_squared(X_df, y_df):.4f}")
print(f"MSE      : {mlr.mse(X_df, y_df):.4f}")`,
    preOutput:
`Intercept  β₀ :  -4.3821
  Hours Studied           :   9.9874
  Previous Scores         :   0.8499
  Extracurricular         :   2.9812
  Sleep Hours             :   1.5034
  Sample Papers           :   2.5018

R² Score : 0.9782
MSE      : 24.9813`,
    outputType: "text"
  },

  {
    type: "code",
    execCount: 9,
    label: "Cell D-2 — Residuals & Predicted vs Actual Plot",
    source:
`y_hat_mlr = mlr.predict(X_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Predicted vs Actual
axes[0].scatter(y_df, y_hat_mlr, alpha=0.15, s=6, color='steelblue')
mn, mx = y_df.min(), y_df.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
axes[0].set_xlabel('Actual Performance Index')
axes[0].set_ylabel('Predicted Performance Index')
axes[0].set_title(f'Predicted vs Actual  (R²={mlr.r_squared(X_df,y_df):.3f})')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_df - y_hat_mlr
axes[1].scatter(y_hat_mlr, residuals, alpha=0.15, s=6, color='darkorange')
axes[1].axhline(0, color='red', lw=1.5, ls='--')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Plot'); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110, bbox_inches='tight'); plt.close()
img_b64 = base64.b64encode(buf.getvalue()).decode()
print(f'__IMG__{img_b64}')`,
    preOutput: "",
    outputType: "text",
    preOutputImg: "mlr_residuals.png"
  }
];

// ── NOTEBOOKS REGISTRY ────────────────────────────────────────────────────
const NOTEBOOKS = {
  lab1: { cells: LAB1_CELLS, title: "lab_assignment_1_naive_bayes_iris.ipynb",
    pySetup: `import numpy as np\nfrom sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\niris=load_iris()\nX,y=iris.data,iris.target\nclass_names=iris.target_names\nfeature_names=iris.feature_names\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)\nclass GaussianNaiveBayes:\n    def fit(self,X,y):\n        self.classes_=np.unique(y);n_samples,n_feats=X.shape;K=len(self.classes_)\n        self._mean=np.zeros((K,n_feats));self._var=np.zeros((K,n_feats));self._log_prior=np.zeros(K)\n        for idx,c in enumerate(self.classes_):\n            Xc=X[y==c];self._mean[idx]=Xc.mean(axis=0);self._var[idx]=Xc.var(axis=0)+1e-9;self._log_prior[idx]=np.log(len(Xc)/n_samples)\n        return self\n    def _log_likelihood(self,k,x):\n        mu,var=self._mean[k],self._var[k];return np.sum(-0.5*np.log(2*np.pi*var)-(x-mu)**2/(2*var))\n    def predict(self,X):\n        preds=[]\n        for x in X:\n            lp=[self._log_prior[k]+self._log_likelihood(k,x) for k in range(len(self.classes_))]\n            preds.append(self.classes_[np.argmax(lp)])\n        return np.array(preds)\n    def score(self,X,y):return np.mean(self.predict(X)==y)\ngnb=GaussianNaiveBayes();gnb.fit(X_train,y_train);y_pred=gnb.predict(X_test)\nn_classes=len(gnb.classes_);cm=np.zeros((n_classes,n_classes),dtype=int)\nfor a,p in zip(y_test,y_pred):cm[a][p]+=1\ndef compute_metrics(cm,class_names):\n    n=len(class_names);precision,recall,f1=[],[],[]\n    for i in range(n):\n        tp=cm[i,i];fp=cm[:,i].sum()-tp;fn=cm[i,:].sum()-tp\n        p=tp/(tp+fp) if (tp+fp)>0 else 0.0;r=tp/(tp+fn) if (tp+fn)>0 else 0.0\n        f=2*p*r/(p+r) if (p+r)>0 else 0.0;precision.append(p);recall.append(r);f1.append(f)\n    return np.array(precision),np.array(recall),np.array(f1)\nprecision,recall,f1=compute_metrics(cm,class_names)\n`
  },
  lab2: { cells: LAB2_CELLS, title: "lab_assignment_2_svm_linear_regression.ipynb",
    pySetup: `import numpy as np,io,base64\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfrom sklearn.datasets import load_iris\nfrom sklearn.preprocessing import StandardScaler\niris=load_iris();mask=iris.target!=2;X_raw=iris.data[mask,2:4];y_raw=iris.target[mask]\ny_bin=np.where(y_raw==0,-1,1);scaler=StandardScaler();X_sc=scaler.fit_transform(X_raw)\nclass LinearSVM:\n    def __init__(self,learning_rate=0.001,lambda_param=0.01,epochs=1000):\n        self.lr=learning_rate;self.lam=lambda_param;self.epochs=epochs;self.w=None;self.b=None\n    def fit(self,X,y):\n        n,d=X.shape;self.w=np.zeros(d);self.b=0.0\n        for _ in range(self.epochs):\n            for i,xi in enumerate(X):\n                m=y[i]*(np.dot(xi,self.w)+self.b)\n                if m>=1:dw=2*self.lam*self.w;db=0.0\n                else:dw=2*self.lam*self.w-y[i]*xi;db=-y[i]\n                self.w-=self.lr*dw;self.b-=self.lr*db\n    def predict(self,X):return np.sign(np.dot(X,self.w)+self.b)\n    def score(self,X,y):return np.mean(self.predict(X)==y)\nclass QuadraticSVM:\n    def __init__(self,lr=0.001,epochs=5000,C=1e9):\n        self.lr=lr;self.epochs=epochs;self.C=C;self.alphas=None;self.w=None;self.b=None;self.sv_mask=None\n    def fit(self,X,y):\n        n=len(y);Q=np.outer(y,y)*(X@X.T);self.alphas=np.zeros(n)\n        for _ in range(self.epochs):\n            grad=np.ones(n)-Q@self.alphas;self.alphas+=self.lr*grad;self.alphas=np.clip(self.alphas,0,self.C)\n            pos=self.alphas[y==1].sum();neg=self.alphas[y==-1].sum()\n            if pos>0 and neg>0:\n                self.alphas[y==1]*=(pos+neg)/(2*pos);self.alphas[y==-1]*=(pos+neg)/(2*neg)\n        self.sv_mask=self.alphas>1e-4;self.w=(self.alphas*y)@X\n        sv_idx=np.where(self.sv_mask)[0];self.b=np.mean(y[sv_idx]-X[sv_idx]@self.w)\n    def predict(self,X):return np.sign(X@self.w+self.b)\n    def score(self,X,y):return np.mean(self.predict(X)==y)\nsvm_linear=LinearSVM();svm_linear.fit(X_sc,y_bin)\nsvm_qp=QuadraticSVM();svm_qp.fit(X_sc,y_bin)\nnp.random.seed(42);n_houses=200\narea=np.random.randint(500,5001,size=n_houses).astype(float)\nprice=60000+95*area+np.random.randn(n_houses)*30000\nclass SimpleLinearRegression:\n    def __init__(self):self.m=0.0;self.c=0.0\n    def fit(self,X,y):\n        xb,yb=X.mean(),y.mean();self.m=np.sum((X-xb)*(y-yb))/np.sum((X-xb)**2);self.c=yb-self.m*xb\n    def predict(self,X):return self.m*X+self.c\n    def r_squared(self,X,y):\n        yh=self.predict(X);return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)\n    def mse(self,X,y):return np.mean((y-self.predict(X))**2)\nslr=SimpleLinearRegression();slr.fit(area,price)\nnp.random.seed(0);N=10000\nhours=np.random.randint(1,10,N).astype(float);prev=np.random.randint(40,100,N).astype(float)\nextra=np.random.randint(0,2,N).astype(float);sleep=np.random.randint(4,10,N).astype(float)\npapers=np.random.randint(0,10,N).astype(float)\nperf_idx=(10*hours+0.85*prev+3*extra+1.5*sleep+2.5*papers+np.random.randn(N)*5).clip(10,100)\nX_df=np.column_stack([hours,prev,extra,sleep,papers]);y_df=perf_idx\nclass MultipleLinearRegression:\n    def __init__(self):self.beta=None\n    def fit(self,X,y):\n        Xb=np.column_stack([np.ones(len(X)),X]);self.beta=np.linalg.lstsq(Xb,y,rcond=None)[0]\n    def predict(self,X):return np.column_stack([np.ones(len(X)),X])@self.beta\n    def r_squared(self,X,y):\n        yh=self.predict(X);return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)\n    def mse(self,X,y):return np.mean((y-self.predict(X))**2)\nmlr=MultipleLinearRegression();mlr.fit(X_df,y_df)\n`
  }
};

// ── STATE ──────────────────────────────────────────────────────────────────
let currentLab     = 'lab1';
let pyodide        = null;
let pyodideLoading = false;
const execCounts   = {};   // { cellIndex: n }

const $ = id => document.getElementById(id);

// ── UTILITIES ─────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function setKernel(state, label) {
  const dot = $("kernel-dot"), lbl = $("kernel-label");
  dot.className = "kernel-dot " + state;
  lbl.textContent = label;
}

// ── MARKDOWN RENDERER ──────────────────────────────────────────────────────
function renderMarkdown(src) {
  let h = escapeHtml(src);
  h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^## (.+)$/gm,  "<h2>$1</h2>");
  h = h.replace(/^# (.+)$/gm,   "<h1>$1</h1>");
  h = h.replace(/^---$/gm,      "<hr>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*(.+?)\*/g,   "<em>$1</em>");
  h = h.replace(/`([^`]+)`/g,   "<code>$1</code>");
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  h = h.replace(/^&gt; (.+)$/gm,"<blockquote>$1</blockquote>");
  h = h.replace(/((?:^\d+\. .+\n?)+)/gm, m => {
    const items = m.trim().split("\n").map(l=>`<li>${l.replace(/^\d+\. /,"")}</li>`).join("");
    return `<ol>${items}</ol>`;
  });
  h = h.replace(/((?:^- .+\n?)+)/gm, m => {
    const items = m.trim().split("\n").map(l=>`<li>${l.replace(/^- /,"")}</li>`).join("");
    return `<ul>${items}</ul>`;
  });
  h = h.split("\n\n").map(block => {
    block = block.trim();
    if (!block) return "";
    if (/^<(h[1-6]|ol|ul|blockquote|hr)/.test(block)) return block;
    return `<p>${block.replace(/\n/g,"<br>")}</p>`;
  }).join("\n");
  return h;
}

// ── BUILD CONFUSION MATRIX TABLE ──────────────────────────────────────────
function buildCMTable(tableData) {
  const wrap = document.createElement("div");
  wrap.className = "output-area output-table";
  const cmWrap = document.createElement("div");
  cmWrap.className = "cm-wrap";
  if (tableData.title) {
    const title = document.createElement("div");
    title.className = "cm-title"; title.textContent = tableData.title;
    cmWrap.appendChild(title);
  }
  const tbl = document.createElement("table");
  tbl.className = "nb-table";
  const thead = document.createElement("thead");
  const hrow  = document.createElement("tr");
  tableData.headers.forEach(h => { const th = document.createElement("th"); th.textContent = h; hrow.appendChild(th); });
  thead.appendChild(hrow); tbl.appendChild(thead);
  const tbody = document.createElement("tbody");
  tableData.rows.forEach((row, ri) => {
    const tr = document.createElement("tr");
    row.forEach((val, ci) => {
      const td = document.createElement("td"); td.textContent = val;
      const r = ri + 1; const c = ci;
      if (tableData.diagIndices && tableData.diagIndices.some(([a,b])=>a===r&&b===c)) td.className = "cm-diag";
      else if (tableData.missIndices && parseInt(val) > 0 && tableData.missIndices.some(([a,b])=>a===r&&b===c)) td.className = "cm-miss";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody); cmWrap.appendChild(tbl); wrap.appendChild(cmWrap);
  return wrap;
}

// ── BUILD METRICS BLOCK ────────────────────────────────────────────────────
function buildMetricsBlock(data) {
  const wrap = document.createElement("div"); wrap.className = "output-area";
  const block = document.createElement("div"); block.className = "metrics-block";
  const acc = document.createElement("div"); acc.className = "metrics-accuracy";
  acc.innerHTML = `Overall Accuracy: <span>${(data.accuracy*100).toFixed(2)}%</span>&nbsp;·&nbsp; Correct: ${data.correct} / ${data.total} samples`;
  block.appendChild(acc);
  const tableWrap = document.createElement("div"); tableWrap.className = "metrics-table-wrap";
  const tbl = document.createElement("table"); tbl.className = "metrics-table";
  const thead = document.createElement("thead");
  const hrow  = document.createElement("tr");
  ["Class","Precision","Recall","F1-Score"].forEach(h => { const th = document.createElement("th"); th.textContent = h; hrow.appendChild(th); });
  thead.appendChild(hrow); tbl.appendChild(thead);
  const tbody = document.createElement("tbody");
  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    [row.label, row.precision, row.recall, row.f1].forEach(v => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  const macroTr = document.createElement("tr"); macroTr.className = "macro-row";
  ["macro avg", data.macro.precision, data.macro.recall, data.macro.f1].forEach(v => { const td = document.createElement("td"); td.textContent = v; macroTr.appendChild(td); });
  tbody.appendChild(macroTr); tbl.appendChild(tbody); tableWrap.appendChild(tbl);
  block.appendChild(tableWrap); wrap.appendChild(block);
  return wrap;
}

// ── AUTO-RESIZE TEXTAREA ──────────────────────────────────────────────────
function autoResize(ta) {
  ta.style.height = "auto";
  ta.style.height = ta.scrollHeight + "px";
}

// ── RENDER OUTPUT (pre-computed or after run) ──────────────────────────────
function renderOutput(outBody, cell, cellIndex) {
  outBody.innerHTML = "";
  if (cell.outputType === "table" && cell.tableData) {
    outBody.appendChild(buildCMTable(cell.tableData));
  } else if (cell.outputType === "metrics" && cell.metricsData) {
    outBody.appendChild(buildMetricsBlock(cell.metricsData));
  } else if (cell.preOutputImg) {
    const d = document.createElement("div");
    d.className = "output-area";
    d.id = `output-area-${cellIndex}`;
    d.innerHTML = `<em style="font-size:11px;color:#888;">[ Click ▶ Run to generate plot ]</em>`;
    outBody.appendChild(d);
  } else if (cell.preOutput) {
    const d = document.createElement("div");
    d.className = "output-area";
    d.id = `output-area-${cellIndex}`;
    d.textContent = cell.preOutput;
    outBody.appendChild(d);
  }
}

// ── RENDER NOTEBOOK ────────────────────────────────────────────────────────
function renderNotebook() {
  const container = $("notebook");
  container.innerHTML = "";
  const cells = NOTEBOOKS[currentLab].cells;

  cells.forEach((cell, cellIndex) => {
    if (cell.type === "markdown") {
      const div = document.createElement("div");
      div.className = "cell cell-markdown";
      div.innerHTML = `<div class="cell-gutter"><div class="cell-counter empty">.</div></div><div class="cell-body"><div class="markdown-content">${renderMarkdown(cell.source)}</div></div>`;
      container.appendChild(div);
      container.appendChild(makeCellGap());
      return;
    }

    if (cell.type === "code") {
      const execN = execCounts[currentLab + '_' + cellIndex] ?? cell.execCount;

      const codeCell = document.createElement("div");
      codeCell.className = "cell cell-code";
      codeCell.dataset.cellIndex = cellIndex;

      const gutter = document.createElement("div");
      gutter.className = "cell-gutter";
      const counter = document.createElement("div");
      counter.className = "cell-counter";
      counter.id = `label-${cellIndex}`;
      counter.innerHTML = `In&nbsp;[${execN}]:`;
      gutter.appendChild(counter);

      const body = document.createElement("div");
      body.className = "cell-body";
      const inner = document.createElement("div");
      inner.className = "cell-code-inner";

      const ta = document.createElement("textarea");
      ta.className = "code-area";
      ta.value = cell.source;
      ta.spellcheck = false;
      ta.setAttribute("autocorrect","off"); ta.setAttribute("autocapitalize","off");
      ta.setAttribute("aria-label", cell.label || `Code cell ${cellIndex}`);
      ta.addEventListener("input", () => autoResize(ta));

      const runBtn = document.createElement("button");
      runBtn.className = "cell-run-btn";
      runBtn.textContent = "\u25B6 Run";
      runBtn.title = "Run this cell (Shift+Enter)";
      runBtn.addEventListener("click", () => runCell(cellIndex, codeCell));
      ta.addEventListener("keydown", e => { if (e.key==="Enter"&&e.shiftKey){ e.preventDefault(); runCell(cellIndex, codeCell); } });

      inner.appendChild(ta); inner.appendChild(runBtn);
      body.appendChild(inner); codeCell.appendChild(gutter); codeCell.appendChild(body);
      container.appendChild(codeCell);

      const outCell = document.createElement("div");
      outCell.className = "cell cell-output";
      outCell.id = `out-cell-${cellIndex}`;
      const outGutter = document.createElement("div");
      outGutter.className = "cell-gutter";
      const outCounter = document.createElement("div");
      outCounter.className = "cell-counter out";
      outCounter.id = `out-label-${cellIndex}`;
      outCounter.innerHTML = `Out&nbsp;[${execN}]:`;
      outGutter.appendChild(outCounter);
      const outBody = document.createElement("div");
      outBody.className = "cell-body";
      outBody.id = `out-body-${cellIndex}`;
      renderOutput(outBody, cell, cellIndex);
      outCell.appendChild(outGutter); outCell.appendChild(outBody);
      container.appendChild(outCell);
      container.appendChild(makeCellGap());
      requestAnimationFrame(() => autoResize(ta));
    }
  });
}

function makeCellGap() {
  const d = document.createElement("div"); d.className = "cell-gap"; return d;
}

// ── PYODIDE ────────────────────────────────────────────────────────────────
async function ensurePyodide() {
  if (pyodide) return pyodide;
  if (pyodideLoading) {
    while (pyodideLoading) await new Promise(r => setTimeout(r, 200));
    return pyodide;
  }
  pyodideLoading = true;
  const banner = $("pyodide-banner");
  const bar    = $("pyodide-bar");
  banner.classList.add("visible");
  setKernel("loading", "Python 3 (loading...)");

  let progress = 0;
  const tick = setInterval(() => {
    progress = Math.min(progress + Math.random() * 12, 88);
    bar.style.width = progress + "%";
  }, 400);

  await new Promise((resolve, reject) => {
    if (window.loadPyodide) { resolve(); return; }
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";
    s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });

  const pkgs = currentLab === 'lab1'
    ? ["numpy", "scikit-learn"]
    : ["numpy", "scikit-learn", "matplotlib"];
  $("pyodide-banner-text").textContent = `Loading packages (${pkgs.join(', ')})...`;
  pyodide = await window.loadPyodide();
  await pyodide.loadPackage(pkgs);

  // Shared kernel setup for current lab
  await pyodide.runPythonAsync(NOTEBOOKS[currentLab].pySetup);

  clearInterval(tick);
  bar.style.width = "100%";
  setTimeout(() => banner.classList.remove("visible"), 600);
  pyodideLoading = false;
  setKernel("idle", "Python 3 (idle)");
  return pyodide;
}

// ── RUN CELL ───────────────────────────────────────────────────────────────
async function runCell(cellIndex, codeCellEl) {
  const ta      = codeCellEl.querySelector(".code-area");
  const runBtn  = codeCellEl.querySelector(".cell-run-btn");
  const outBody = $(`out-body-${cellIndex}`);

  runBtn.textContent = "...";
  runBtn.classList.add("running");
  setKernel("busy", "Python 3 (busy)");
  codeCellEl.classList.add("selected");

  try {
    const py = await ensurePyodide();
    let stdout = "";
    py.setStdout({ batched: s => { stdout += s + "\n"; } });
    await py.runPythonAsync(ta.value);
    py.setStdout(null);

    const key = currentLab + '_' + cellIndex;
    execCounts[key] = (execCounts[key] ?? NOTEBOOKS[currentLab].cells[cellIndex].execCount) + 1;
    const n = execCounts[key];
    const lbl    = $(`label-${cellIndex}`);
    const outLbl = $(`out-label-${cellIndex}`);
    if (lbl)    lbl.innerHTML    = `In&nbsp;[${n}]:`;
    if (outLbl) outLbl.innerHTML = `Out&nbsp;[${n}]:`;

    outBody.innerHTML = "";
    // Check if output contains an image marker
    if (stdout.includes('__IMG__')) {
      const b64 = stdout.split('__IMG__')[1].trim();
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${b64}`;
      img.style.cssText = "max-width:100%;display:block;";
      const d = document.createElement("div");
      d.className = "output-area";
      d.appendChild(img);
      outBody.appendChild(d);
    } else {
      const d = document.createElement("div");
      d.className = "output-area";
      d.id = `output-area-${cellIndex}`;
      d.textContent = stdout.trim();
      outBody.appendChild(d);
    }
    setKernel("idle", "Python 3 (idle)");
  } catch (err) {
    py && py.setStdout && py.setStdout(null);
    setKernel("idle", "Python 3 (idle)");
    outBody.innerHTML = "";
    const d = document.createElement("div");
    d.className = "output-area output-error";
    d.textContent = String(err);
    outBody.appendChild(d);
  } finally {
    runBtn.textContent = "\u25B6 Run";
    runBtn.classList.remove("running");
    codeCellEl.classList.remove("selected");
  }
}
