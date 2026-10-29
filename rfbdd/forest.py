from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Tree:
    """Single decision tree with array-based node representation (Section 2)."""

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    root: int = 0

    @property
    def n_nodes(self) -> int:
        """Total number of nodes in the tree."""
        return int(self.feature.shape[0])

    def is_leaf(self, node: int) -> bool:
        """Return True if node is a leaf."""
        return int(self.left[node]) < 0

    def leaves(self) -> list[int]:
        """Return indices of all leaf nodes."""
        return [i for i in range(self.n_nodes) if self.is_leaf(i)]

    def internal_nodes(self) -> list[int]:
        """Return indices of all internal nodes."""
        return [i for i in range(self.n_nodes) if not self.is_leaf(i)]

    def decide(self, x: np.ndarray) -> int:
        """Predict the class index for a real-valued sample x."""
        node = self.root
        while not self.is_leaf(node):
            f = int(self.feature[node])
            node = int(self.left[node]) if x[f] <= self.threshold[node] else int(self.right[node])
        return int(self.label[node])

    def decide_cells(self, cells: np.ndarray, taus: dict[int, int]) -> int:
        """Predict the class index for a discretized sample using threshold indices taus."""
        node = self.root
        while not self.is_leaf(node):
            f = int(self.feature[node])
            node = int(self.left[node]) if cells[f] <= taus[node] else int(self.right[node])
        return int(self.label[node])

    def paths(self) -> list[tuple[list[tuple[int, int, int]], int]]:
        """Return all root-to-leaf paths as ([(node, feature, branch_dir), ...], leaf_label)."""
        out: list[tuple[list[tuple[int, int, int]], int]] = []
        stack: list[tuple[int, list[tuple[int, int, int]]]] = [(self.root, [])]
        while stack:
            node, acc = stack.pop()
            if self.is_leaf(node):
                out.append((acc, int(self.label[node])))
                continue
            f = int(self.feature[node])
            stack.append((int(self.right[node]), acc + [(node, f, 0)]))
            stack.append((int(self.left[node]), acc + [(node, f, 1)]))
        return out


@dataclass
class Forest:
    """Random forest F = {T_0, ..., T_{D-1}} using majority voting (Section 2)."""

    trees: list[Tree]
    n_features: int
    n_classes: int
    classes: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_trees(self) -> int:
        """Number of trees in the forest."""
        return len(self.trees)

    @property
    def D(self) -> int:
        """Number of trees D."""
        return len(self.trees)

    @property
    def W(self) -> int:
        """Minimum number of votes needed to win: floor(D / 2) + 1."""
        return self.D // 2 + 1

    def prefix(self, k: int) -> "Forest":
        """Return a new forest containing only the first k trees."""
        k = max(1, min(k, self.n_trees))
        return Forest(
            trees=self.trees[:k],
            n_features=self.n_features,
            n_classes=self.n_classes,
            classes=self.classes,
            name=self.name,
            meta=dict(self.meta),
        )

    def votes(self, x: np.ndarray) -> np.ndarray:
        """Return vote counts per class for a real-valued sample x."""
        v = np.zeros(self.n_classes, dtype=np.int64)
        for t in self.trees:
            v[t.decide(x)] += 1
        return v

    def votes_cells(self, cells: np.ndarray, taus: list[dict[int, int]]) -> np.ndarray:
        """Return vote counts per class for a discretized sample."""
        v = np.zeros(self.n_classes, dtype=np.int64)
        for t_idx, t in enumerate(self.trees):
            v[t.decide_cells(cells, taus[t_idx])] += 1
        return v

    def predict_majority(self, X: np.ndarray) -> np.ndarray:
        """Return majority-vote predictions for an array of samples X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.array([int(np.argmax(self.votes(X)))], dtype=np.int64)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            out[i] = int(np.argmax(self.votes(X[i])))
        return out


def _sk_tree_to_tree(sk_tree) -> Tree:
    """Convert a scikit-learn DecisionTreeClassifier into a Tree."""
    t = sk_tree.tree_
    n = t.node_count
    feature = np.asarray(t.feature, dtype=np.int64).copy()
    threshold = np.asarray(t.threshold, dtype=np.float64).copy()
    left = np.asarray(t.children_left, dtype=np.int64).copy()
    right = np.asarray(t.children_right, dtype=np.int64).copy()
    value = np.asarray(t.value)
    label = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if left[i] < 0:
            label[i] = int(np.argmax(value[i, 0]))
            feature[i] = -1
            threshold[i] = np.nan
    return Tree(feature=feature, threshold=threshold, left=left, right=right, label=label)


def load_forest(path: str, max_trees: int | None = None) -> Forest:
    """Load a pickled scikit-learn random forest and convert it to a Forest."""
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blob = joblib.load(path)
    model = blob["model"] if isinstance(blob, dict) else blob
    trees = [_sk_tree_to_tree(e) for e in model.estimators_]
    if max_trees is not None:
        trees = trees[:max_trees]
    meta = {k: v for k, v in blob.items() if k != "model"} if isinstance(blob, dict) else {}
    return Forest(
        trees=trees,
        n_features=int(model.n_features_in_),
        n_classes=int(len(model.classes_)),
        classes=np.asarray(model.classes_),
        name=str(meta.get("dataset", os.path.basename(path))),
        meta=meta | {"sklearn_model": model},
    )


def recover_label_map(model, X_train: np.ndarray, y_raw) -> dict:
    """Map raw CSV labels to scikit-learn class indices."""
    cats = sorted(pd.unique(pd.Series(y_raw)))
    idx = {c: i for i, c in enumerate(cats)}
    C = max(len(cats), len(model.classes_))
    M = np.zeros((C, C))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict(X_train)
    for a, b in zip((idx[v] for v in y_raw), pred):
        M[a, b] += 1
    rows, cols = linear_sum_assignment(-M)
    return {cats[i]: int(j) for i, j in zip(rows, cols) if i < len(cats)}


@dataclass
class Dataset:
    """Train and test splits for a benchmark dataset."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_map: dict
    sk_test_acc: float
    stored_test_acc: float


def load_dataset(name: str, root: str = "datasets") -> Dataset:
    """Load train/test CSVs for a dataset and align labels with its pickled model."""
    import joblib

    base = os.path.join(root, name)
    tr = pd.read_csv(os.path.join(base, f"{name}_train.csv"))
    te = pd.read_csv(os.path.join(base, f"{name}_test.csv"))
    forest_blob_path = os.path.join(base, f"{name}_rf.pkl")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        blob = joblib.load(forest_blob_path)
    model = blob["model"] if isinstance(blob, dict) else blob

    feature_names = [str(c) for c in tr.columns[:-1]]
    X_train = tr.iloc[:, :-1].to_numpy(dtype=float)
    X_test = te.iloc[:, :-1].to_numpy(dtype=float)
    lm = recover_label_map(model, X_train, tr.iloc[:, -1].tolist())
    y_train = np.array([lm[v] for v in tr.iloc[:, -1]], dtype=np.int64)
    y_test = np.array([lm[v] for v in te.iloc[:, -1]], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = float((model.predict(X_test) == y_test).mean())
    return Dataset(
        name=name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        label_map=lm,
        sk_test_acc=acc,
        stored_test_acc=float(blob.get("test_acc", float("nan"))) if isinstance(blob, dict) else float("nan"),
    )


def list_datasets(root: str = "datasets") -> list[str]:
    """List all dataset names in root that have a saved random forest model."""
    out = []
    if not os.path.exists(root):
        return out
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, f"{d}_rf.pkl")):
            out.append(d)
    return out
