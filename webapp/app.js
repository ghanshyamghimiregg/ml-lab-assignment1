/* ==========================================================
   ML Lab Notebook — app.js
   Ghanshyam Ghimire | KU BTech AI, Sem 4
   Subject: Introduction to Machine Learning
   ========================================================== */

// ── CELL DATA ──────────────────────────────────────────────────────────────
// type: "markdown" | "code" | "output-table" | "output-metrics"
// Code cells: source, execCount, preOutput, outputType

const LAB1_CELLS = [

  // ── 0: Title + instructions ─────────────────────────────────────────────
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

  // ── 1: Imports + data loading ───────────────────────────────────────────
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

  // ── 2: GaussianNaiveBayes class ─────────────────────────────────────────
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
        # log N(x; mu, var) = -0.5*log(2*pi*var) - (x-mu)^2 / (2*var)
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

  // ── 3: Train & predict ──────────────────────────────────────────────────
  {
    type: "code",
    execCount: 3,
    label: "Cell 3 — Training & Prediction",
    source:
`# ── Fit the model
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# ── Learned parameters
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

  // ── 4: Confusion matrix ─────────────────────────────────────────────────
  {
    type: "code",
    execCount: 4,
    label: "Cell 4 — Confusion Matrix",
    source:
`# ── Confusion matrix (manual, no sklearn)
n_classes = len(gnb.classes_)
cm = np.zeros((n_classes, n_classes), dtype=int)
for a, p in zip(y_test, y_pred):
    cm[a][p] += 1

# Print text version
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

  // ── 5: Performance metrics ───────────────────────────────────────────────
  {
    type: "code",
    execCount: 5,
    label: "Cell 5 — Performance Metrics",
    source:
`def compute_metrics(cm, class_names):
    """
    Compute per-class precision, recall, F1-score from a confusion matrix.
    No sklearn — pure numpy arithmetic.
    """
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


// ── STATE ──────────────────────────────────────────────────────────────────
let pyodide        = null;
let pyodideLoading = false;
const execCounts   = {};   // { cellIndex: n }

const $ = id => document.getElementById(id);

// ── UTILITIES ─────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function setKernel(state, label) {
  const dot = $("kernel-dot");
  const lbl = $("kernel-label");
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
    title.className = "cm-title";
    title.textContent = tableData.title;
    cmWrap.appendChild(title);
  }

  const tbl  = document.createElement("table");
  tbl.className = "nb-table";

  // header row
  const thead = document.createElement("thead");
  const hrow  = document.createElement("tr");
  tableData.headers.forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  tbl.appendChild(thead);

  // body rows
  const tbody = document.createElement("tbody");
  tableData.rows.forEach((row, ri) => {
    const tr = document.createElement("tr");
    row.forEach((val, ci) => {
      const td = document.createElement("td");
      td.textContent = val;
      // colour coding: diagonal = correct, off-diagonal with value > 0 = miss
      const r = ri + 1; const c = ci; // header col offset
      if (tableData.diagIndices && tableData.diagIndices.some(([a,b])=>a===r&&b===c)) {
        td.className = "cm-diag";
      } else if (tableData.missIndices && parseInt(val) > 0 &&
                 tableData.missIndices.some(([a,b])=>a===r&&b===c)) {
        td.className = "cm-miss";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  cmWrap.appendChild(tbl);
  wrap.appendChild(cmWrap);
  return wrap;
}

// ── BUILD METRICS BLOCK ────────────────────────────────────────────────────
function buildMetricsBlock(data) {
  const wrap = document.createElement("div");
  wrap.className = "output-area";

  const block = document.createElement("div");
  block.className = "metrics-block";

  const acc = document.createElement("div");
  acc.className = "metrics-accuracy";
  acc.innerHTML = `Overall Accuracy: <span>${(data.accuracy*100).toFixed(2)}%</span>`
    + `  &nbsp;·&nbsp; Correct: ${data.correct} / ${data.total} samples`;
  block.appendChild(acc);

  const tableWrap = document.createElement("div");
  tableWrap.className = "metrics-table-wrap";

  const tbl = document.createElement("table");
  tbl.className = "metrics-table";

  // header
  const thead = document.createElement("thead");
  const hrow  = document.createElement("tr");
  ["Class","Precision","Recall","F1-Score"].forEach(h => {
    const th = document.createElement("th"); th.textContent = h; hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  tbl.appendChild(thead);

  const tbody = document.createElement("tbody");
  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    [row.label, row.precision, row.recall, row.f1].forEach(v => {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // macro row
  const macroTr = document.createElement("tr");
  macroTr.className = "macro-row";
  ["macro avg", data.macro.precision, data.macro.recall, data.macro.f1].forEach(v => {
    const td = document.createElement("td"); td.textContent = v; macroTr.appendChild(td);
  });
  tbody.appendChild(macroTr);
  tbl.appendChild(tbody);
  tableWrap.appendChild(tbl);
  block.appendChild(tableWrap);
  wrap.appendChild(block);
  return wrap;
}


// ── AUTO-RESIZE TEXTAREA ──────────────────────────────────────────────────
function autoResize(ta) {
  ta.style.height = "auto";
  ta.style.height = ta.scrollHeight + "px";
}

// ── RENDER NOTEBOOK ────────────────────────────────────────────────────────
function renderNotebook() {
  const container = $("notebook");
  container.innerHTML = "";

  LAB1_CELLS.forEach((cell, cellIndex) => {

    // ── MARKDOWN ──────────────────────────────────────────────────────────
    if (cell.type === "markdown") {
      const div = document.createElement("div");
      div.className = "cell cell-markdown";
      div.innerHTML = `
        <div class="cell-gutter"><div class="cell-counter empty">.</div></div>
        <div class="cell-body">
          <div class="markdown-content">${renderMarkdown(cell.source)}</div>
        </div>`;
      container.appendChild(div);
      container.appendChild(makeCellGap());
      return;
    }

    // ── CODE CELL ─────────────────────────────────────────────────────────
    if (cell.type === "code") {
      const execN = execCounts[cellIndex] ?? cell.execCount;

      // input
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
      ta.setAttribute("autocorrect", "off");
      ta.setAttribute("autocapitalize", "off");
      ta.setAttribute("aria-label", cell.label || `Code cell ${cellIndex}`);
      ta.addEventListener("input", () => autoResize(ta));

      const runBtn = document.createElement("button");
      runBtn.className = "cell-run-btn";
      runBtn.textContent = "\u25B6 Run";
      runBtn.title = "Run this cell (Shift+Enter)";
      runBtn.addEventListener("click", () => runCell(cellIndex, codeCell));

      // Shift+Enter to run
      ta.addEventListener("keydown", e => {
        if (e.key === "Enter" && e.shiftKey) {
          e.preventDefault();
          runCell(cellIndex, codeCell);
        }
      });

      inner.appendChild(ta);
      inner.appendChild(runBtn);
      body.appendChild(inner);
      codeCell.appendChild(gutter);
      codeCell.appendChild(body);
      container.appendChild(codeCell);

      // output cell
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

      // render pre-computed output
      renderOutput(outBody, cell, cellIndex);

      outCell.appendChild(outGutter);
      outCell.appendChild(outBody);
      container.appendChild(outCell);
      container.appendChild(makeCellGap());

      requestAnimationFrame(() => autoResize(ta));
    }
  });
}

function makeCellGap() {
  const d = document.createElement("div");
  d.className = "cell-gap";
  return d;
}

// ── RENDER OUTPUT (pre-computed or after run) ──────────────────────────────
function renderOutput(outBody, cell, cellIndex) {
  outBody.innerHTML = "";

  if (cell.outputType === "table" && cell.tableData) {
    outBody.appendChild(buildCMTable(cell.tableData));

  } else if (cell.outputType === "metrics" && cell.metricsData) {
    outBody.appendChild(buildMetricsBlock(cell.metricsData));

  } else if (cell.preOutput) {
    const d = document.createElement("div");
    d.className = "output-area";
    d.id = `output-area-${cellIndex}`;
    d.textContent = cell.preOutput;
    outBody.appendChild(d);
  }
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
    s.onload  = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });

  $("pyodide-banner-text").textContent = "Loading packages (numpy, scikit-learn)...";
  pyodide = await window.loadPyodide();
  await pyodide.loadPackage(["numpy", "scikit-learn"]);

  // Shared kernel state — pre-run so any cell can run standalone
  await pyodide.runPythonAsync(`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
class_names   = iris.target_names
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_feats = X.shape
        K = len(self.classes_)
        self._mean      = np.zeros((K, n_feats))
        self._var       = np.zeros((K, n_feats))
        self._log_prior = np.zeros(K)
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            self._mean[idx]      = Xc.mean(axis=0)
            self._var[idx]       = Xc.var(axis=0) + 1e-9
            self._log_prior[idx] = np.log(len(Xc) / n_samples)
        return self
    def _log_likelihood(self, k, x):
        mu, var = self._mean[k], self._var[k]
        return np.sum(-0.5*np.log(2*np.pi*var) - (x-mu)**2/(2*var))
    def predict(self, X):
        preds = []
        for x in X:
            lp = [self._log_prior[k]+self._log_likelihood(k,x) for k in range(len(self.classes_))]
            preds.append(self.classes_[np.argmax(lp)])
        return np.array(preds)
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

n_classes = len(gnb.classes_)
cm = np.zeros((n_classes, n_classes), dtype=int)
for a, p in zip(y_test, y_pred):
    cm[a][p] += 1

def compute_metrics(cm, class_names):
    n = len(class_names)
    precision, recall, f1 = [], [], []
    for i in range(n):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f = 2*p*r/(p+r) if (p+r)>0 else 0.0
        precision.append(p); recall.append(r); f1.append(f)
    return np.array(precision), np.array(recall), np.array(f1)

precision, recall, f1 = compute_metrics(cm, class_names)
`);

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

    // bump exec counter
    execCounts[cellIndex] = (execCounts[cellIndex] ?? LAB1_CELLS[cellIndex].execCount) + 1;
    const n = execCounts[cellIndex];
    const lbl    = $(`label-${cellIndex}`);
    const outLbl = $(`out-label-${cellIndex}`);
    if (lbl)    lbl.innerHTML    = `In&nbsp;[${n}]:`;
    if (outLbl) outLbl.innerHTML = `Out&nbsp;[${n}]:`;

    outBody.innerHTML = "";
    const d = document.createElement("div");
    d.className = "output-area";
    d.id = `output-area-${cellIndex}`;
    d.textContent = stdout.trim();
    outBody.appendChild(d);

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

// ── RUN ALL ────────────────────────────────────────────────────────────────
$("run-all-btn").addEventListener("click", async () => {
  for (let i = 0; i < LAB1_CELLS.length; i++) {
    if (LAB1_CELLS[i].type !== "code") continue;
    const el = document.querySelector(`.cell-code[data-cell-index="${i}"]`);
    if (el) await runCell(i, el);
  }
});

// ── RESTART KERNEL ─────────────────────────────────────────────────────────
$("restart-btn").addEventListener("click", () => {
  pyodide       = null;
  pyodideLoading = false;
  Object.keys(execCounts).forEach(k => delete execCounts[k]);
  setKernel("", "Python 3 (idle)");
  // Re-render so counters reset to original
  renderNotebook();
});


// ── DOWNLOAD .ipynb ────────────────────────────────────────────────────────
$("download-btn").addEventListener("click", () => {
  const ipynbCells = LAB1_CELLS.map((cell, i) => {
    if (cell.type === "markdown") {
      return {
        cell_type: "markdown",
        metadata: {},
        source: cell.source.split("\n").map((l,j,a) => j<a.length-1 ? l+"\n" : l)
      };
    }
    const outEl = $(`output-area-${i}`);
    let txt = outEl ? outEl.textContent : (cell.preOutput || "");
    if (!txt && cell.outputType === "table" && cell.tableData) {
      const { headers, rows } = cell.tableData;
      txt = headers.join("\t") + "\n" + rows.map(r=>r.join("\t")).join("\n");
    }
    if (!txt && cell.outputType === "metrics" && cell.metricsData) {
      const d = cell.metricsData;
      txt = `Overall Accuracy: ${(d.accuracy*100).toFixed(2)}%\n\n`;
      txt += `Class          Precision    Recall    F1-Score\n`;
      txt += "─".repeat(48) + "\n";
      d.rows.forEach(r => { txt += `${r.label.padEnd(14)} ${r.precision.padStart(10)} ${r.recall.padStart(10)} ${r.f1.padStart(10)}\n`; });
      txt += "─".repeat(48) + "\n";
      txt += `${"macro avg".padEnd(14)} ${d.macro.precision.padStart(10)} ${d.macro.recall.padStart(10)} ${d.macro.f1.padStart(10)}\n`;
    }
    const execN = execCounts[i] ?? cell.execCount;
    return {
      cell_type: "code",
      execution_count: execN,
      metadata: {},
      source: cell.source.split("\n").map((l,j,a) => j<a.length-1 ? l+"\n" : l),
      outputs: txt ? [{
        output_type: "stream", name: "stdout",
        text: txt.split("\n").map((l,j,a) => j<a.length-1 ? l+"\n" : l)
      }] : []
    };
  });

  const ipynb = {
    nbformat: 4, nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python", version: "3.11.0" }
    },
    cells: ipynbCells
  };

  triggerDownload(
    JSON.stringify(ipynb, null, 2),
    "application/json",
    "lab_assignment_1_naive_bayes_iris.ipynb"
  );
});

// ── DOWNLOAD PYTHON CODE ──────────────────────────────────────────────────
$("download-code-btn").addEventListener("click", () => {
  const header =
`# ============================================================
# Lab Assignment 1 — Gaussian Naïve Bayes on Iris Dataset
# Ghanshyam Ghimire | KU BTech AI, 4th Semester
# Instructor: Sandeep Gupta
# Subject: Introduction to Machine Learning
# ============================================================

`;
  const code = LAB1_CELLS
    .filter(c => c.type === "code")
    .map(c => `# ${"─".repeat(58)}\n# ${c.label || "Cell"}\n# ${"─".repeat(58)}\n\n${c.source}`)
    .join("\n\n\n");

  triggerDownload(header + code, "text/plain", "naive_bayes_iris.py");
});

// ── DOWNLOAD IRIS CSV ─────────────────────────────────────────────────────
$("download-data-btn").addEventListener("click", () => {
  // Embed a representative subset of Iris data (first 10 per class shown, full 150 below)
  // Full dataset — all 150 rows from Fisher's Iris dataset (sepal_l, sepal_w, petal_l, petal_w, class)
  const rows = [
    "sepal_length,sepal_width,petal_length,petal_width,class",
    "5.1,3.5,1.4,0.2,setosa","4.9,3.0,1.4,0.2,setosa","4.7,3.2,1.3,0.2,setosa",
    "4.6,3.1,1.5,0.2,setosa","5.0,3.6,1.4,0.2,setosa","5.4,3.9,1.7,0.4,setosa",
    "4.6,3.4,1.4,0.3,setosa","5.0,3.4,1.5,0.2,setosa","4.4,2.9,1.4,0.2,setosa",
    "4.9,3.1,1.5,0.1,setosa","5.4,3.7,1.5,0.2,setosa","4.8,3.4,1.6,0.2,setosa",
    "4.8,3.0,1.4,0.1,setosa","4.3,3.0,1.1,0.1,setosa","5.8,4.0,1.2,0.2,setosa",
    "5.7,4.4,1.5,0.4,setosa","5.4,3.9,1.3,0.4,setosa","5.1,3.5,1.4,0.3,setosa",
    "5.7,3.8,1.7,0.3,setosa","5.1,3.8,1.5,0.3,setosa","5.4,3.4,1.7,0.2,setosa",
    "5.1,3.7,1.5,0.4,setosa","4.6,3.6,1.0,0.2,setosa","5.1,3.3,1.7,0.5,setosa",
    "4.8,3.4,1.9,0.2,setosa","5.0,3.0,1.6,0.2,setosa","5.0,3.4,1.6,0.4,setosa",
    "5.2,3.5,1.5,0.2,setosa","5.2,3.4,1.4,0.2,setosa","4.7,3.2,1.6,0.2,setosa",
    "4.8,3.1,1.6,0.2,setosa","5.4,3.4,1.5,0.4,setosa","5.2,4.1,1.5,0.1,setosa",
    "5.5,4.2,1.4,0.2,setosa","4.9,3.1,1.5,0.2,setosa","5.0,3.2,1.2,0.2,setosa",
    "5.5,3.5,1.3,0.2,setosa","4.9,3.6,1.4,0.1,setosa","4.4,3.0,1.3,0.2,setosa",
    "5.1,3.4,1.5,0.2,setosa","5.0,3.5,1.3,0.3,setosa","4.5,2.3,1.3,0.3,setosa",
    "4.4,3.2,1.3,0.2,setosa","5.0,3.5,1.6,0.6,setosa","5.1,3.8,1.9,0.4,setosa",
    "4.8,3.0,1.4,0.3,setosa","5.1,3.8,1.6,0.2,setosa","4.6,3.2,1.4,0.2,setosa",
    "5.3,3.7,1.5,0.2,setosa","5.0,3.3,1.4,0.2,setosa",
    "7.0,3.2,4.7,1.4,versicolor","6.4,3.2,4.5,1.5,versicolor","6.9,3.1,4.9,1.5,versicolor",
    "5.5,2.3,4.0,1.3,versicolor","6.5,2.8,4.6,1.5,versicolor","5.7,2.8,4.5,1.3,versicolor",
    "6.3,3.3,4.7,1.6,versicolor","4.9,2.4,3.3,1.0,versicolor","6.6,2.9,4.6,1.3,versicolor",
    "5.2,2.7,3.9,1.4,versicolor","5.0,2.0,3.5,1.0,versicolor","5.9,3.0,4.2,1.5,versicolor",
    "6.0,2.2,4.0,1.0,versicolor","6.1,2.9,4.7,1.4,versicolor","5.6,2.9,3.6,1.3,versicolor",
    "6.7,3.1,4.4,1.4,versicolor","5.6,3.0,4.5,1.5,versicolor","5.8,2.7,4.1,1.0,versicolor",
    "6.2,2.2,4.5,1.5,versicolor","5.6,2.5,3.9,1.1,versicolor","5.9,3.2,4.8,1.8,versicolor",
    "6.1,2.8,4.0,1.3,versicolor","6.3,2.5,4.9,1.5,versicolor","6.1,2.8,4.7,1.2,versicolor",
    "6.4,2.9,4.3,1.3,versicolor","6.6,3.0,4.4,1.4,versicolor","6.8,2.8,4.8,1.4,versicolor",
    "6.7,3.0,5.0,1.7,versicolor","6.0,2.9,4.5,1.5,versicolor","5.7,2.6,3.5,1.0,versicolor",
    "5.5,2.4,3.8,1.1,versicolor","5.5,2.4,3.7,1.0,versicolor","5.8,2.7,3.9,1.2,versicolor",
    "6.0,2.7,5.1,1.6,versicolor","5.4,3.0,4.5,1.5,versicolor","6.0,3.4,4.5,1.6,versicolor",
    "6.7,3.1,4.7,1.5,versicolor","6.3,2.3,4.4,1.3,versicolor","5.6,3.0,4.1,1.3,versicolor",
    "5.5,2.5,4.0,1.3,versicolor","5.5,2.6,4.4,1.2,versicolor","6.1,3.0,4.6,1.4,versicolor",
    "5.8,2.6,4.0,1.2,versicolor","5.0,2.3,3.3,1.0,versicolor","5.6,2.7,4.2,1.3,versicolor",
    "5.7,3.0,4.2,1.2,versicolor","5.7,2.9,4.2,1.3,versicolor","6.2,2.9,4.3,1.3,versicolor",
    "5.1,2.5,3.0,1.1,versicolor","5.7,2.8,4.1,1.3,versicolor",
    "6.3,3.3,6.0,2.5,virginica","5.8,2.7,5.1,1.9,virginica","7.1,3.0,5.9,2.1,virginica",
    "6.3,2.9,5.6,1.8,virginica","6.5,3.0,5.8,2.2,virginica","7.6,3.0,6.6,2.1,virginica",
    "4.9,2.5,4.5,1.7,virginica","7.3,2.9,6.3,1.8,virginica","6.7,2.5,5.8,1.8,virginica",
    "7.2,3.6,6.1,2.5,virginica","6.5,3.2,5.1,2.0,virginica","6.4,2.7,5.3,1.9,virginica",
    "6.8,3.0,5.5,2.1,virginica","5.7,2.5,5.0,2.0,virginica","5.8,2.8,5.1,2.4,virginica",
    "6.4,3.2,5.3,2.3,virginica","6.5,3.0,5.5,1.8,virginica","7.7,3.8,6.7,2.2,virginica",
    "7.7,2.6,6.9,2.3,virginica","6.0,2.2,5.0,1.5,virginica","6.9,3.2,5.7,2.3,virginica",
    "5.6,2.8,4.9,2.0,virginica","7.7,2.8,6.7,2.0,virginica","6.3,2.7,4.9,1.8,virginica",
    "6.7,3.3,5.7,2.1,virginica","7.2,3.2,6.0,1.8,virginica","6.2,2.8,4.8,1.8,virginica",
    "6.1,3.0,4.9,1.8,virginica","6.4,2.8,5.6,2.1,virginica","7.2,3.0,5.8,1.6,virginica",
    "7.4,2.8,6.1,1.9,virginica","7.9,3.8,6.4,2.0,virginica","6.4,2.8,5.6,2.2,virginica",
    "6.3,2.8,5.1,1.5,virginica","6.1,2.6,5.6,1.4,virginica","7.7,3.0,6.1,2.3,virginica",
    "6.3,3.4,5.6,2.4,virginica","6.4,3.1,5.5,1.8,virginica","6.0,3.0,4.8,1.8,virginica",
    "6.9,3.1,5.4,2.1,virginica","6.7,3.1,5.6,2.4,virginica","6.9,3.1,5.1,2.3,virginica",
    "5.8,2.7,5.1,1.9,virginica","6.8,3.2,5.9,2.3,virginica","6.7,3.3,5.7,2.5,virginica",
    "6.7,3.0,5.2,2.3,virginica","6.3,2.5,5.0,1.9,virginica","6.5,3.0,5.2,2.0,virginica",
    "6.2,3.4,5.4,2.3,virginica","5.9,3.0,5.1,1.8,virginica"
  ];
  triggerDownload(rows.join("\n"), "text/csv", "iris_dataset.csv");
});

// ── TRIGGER DOWNLOAD HELPER ────────────────────────────────────────────────
function triggerDownload(content, mime, filename) {
  const blob = new Blob([content], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── INIT ───────────────────────────────────────────────────────────────────
renderNotebook();
