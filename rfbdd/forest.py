from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass
class Tree:
    """
    Represents a single decision tree.

    The tree is stored as a collection of nodes. For each node ``i``:

    - ``feature[i]`` specifies which feature is tested;
    - ``threshold[i]`` specifies the threshold of the test
      ``x[feature[i]] <= threshold[i]``;
    - ``left[i]`` is the next node when the test is true (``<=``);
    - ``right[i]`` is the next node when the test is false (``>``);
    - ``label[i]`` specifies the class returned when the node is a leaf.

    An internal node performs a test, while a leaf returns a class.

    For leaves:
    - ``feature`` is ``-1``;
    - ``threshold`` is ``nan``;
    - ``label`` contains the predicted class.

    For internal nodes:
    - ``feature`` and ``threshold`` define the test;
    - ``label`` is ``-1``.

    ``root`` is the index of the node where the tree starts, normally ``0``.
    """

    # represent tree as an array. 
    # each position in the array, represent a node in the tree.

    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    root: int = 0

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the tree."""
        return int(self.feature.shape[0])

    def is_leaf(self, node: int) -> bool:
        """Check if a node is a leaf."""
        return self.left[node] < 0

    def leaves(self) -> list[int]:
        """Return the indices of the leaves in the tree."""
        return [i for i in range(self.n_nodes) if self.is_leaf(i)]

    def internal_nodes(self) -> list[int]:
        """Return the indices of the internal nodes in the tree."""
        return [i for i in range(self.n_nodes) if not self.is_leaf(i)]

    def decide(self, x: np.ndarray) -> int:
        """Class index (leaf) reached by the real-valued sample ``x`` (the ``<=`` rule)."""
        node = self.root
        while not self.is_leaf(node):
            node = self.left[node] if x[self.feature[node]] <= self.threshold[node] else self.right[node]
        return int(self.label[node])

    def decide_cells(self, cells: np.ndarray, taus: dict[int, int]) -> int:
        """
        Predicts the class of a <b>discretized sample</b>.

        It needs the ``cells`` array, which contains the discretized values of the features, and the ``taus`` dictionary, which maps each node to its corresponding threshold index.

        Returns the class label stored in the reached leaf.
        """
        node = self.root
        while not self.is_leaf(node):
            node = self.left[node] if cells[self.feature[node]] <= taus[node] else self.right[node]
        return int(self.label[node])


    def paths(self) -> list[tuple[list[tuple[int, int, int]], int]]:
        """
        It takes the tree and returns all root-to-leaf paths, indicating for each path which nodes are visited, which features are tested, and which branch is taken.
        """
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
    """Represents a random forest as a collection of decision trees.

    The forest is composed of ``D`` trees, denoted by
    ``F = {T_0, ..., T_{D-1}}``.

    Attributes
    ----------
    trees
        List of decision trees contained in the forest.
    n_features
        Number of input features used by the trees.
    n_classes
        Number of possible output classes.
    classes
        Array containing the actual class labels.
    name
        Name of the dataset or forest.
    meta
        Additional metadata associated with the forest.

    The forest uses hard majority voting: each tree produces one class label,
    and the class receiving the most votes is the forest prediction.
    """

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
        """Number of trees in the forest."""
        return len(self.trees)

    @property
    def W(self) -> int:  
        """Minimum number of trees needed to win a majority vote."""
        return self.D // 2 + 1

    def prefix(self, k: int) -> "Forest":
        """
        Returns a new forest containing only the first ``k`` trees.

        If ``k`` is smaller than 1 or larger than the number of trees,
        it is adjusted to the valid range.
        """
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
        """Returns the number of votes received by each class for sample ``x``."""
        v = np.zeros(self.n_classes, dtype=np.int64)
        for t in self.trees:
            v[t.decide(x)] += 1
        return v

    def predict_majority(self, X: np.ndarray) -> np.ndarray:
        """Returns the forest prediction for each sample using majority voting."""
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            out[i] = int(np.argmax(self.votes(X[i])))
        return out

    def certified_label(self, x: np.ndarray) -> int:
        """
        Returns the class that is guaranteed to win (certified) for sample ``x``,
        or ``-1`` if no class can be guaranteed to win.
        """
        lc = np.zeros(self.n_classes, dtype=np.int64)
        D = self.D
        for t_idx, tree in enumerate(self.trees):
            s = tree.decide(x)
            R_t = D - 1 - t_idx
            clinch = all(lc[s] >= lc[l] + R_t for l in range(self.n_classes) if l != s)
            if clinch:
                return s
            lc[s] += 1
        return -1

    def certified_label_cells(self, cells: np.ndarray, taus: list[dict[int, int]]) -> int:
        """
        Returns the class that is guaranteed to win for the discretized sample,
        or ``-1`` if no class can be guaranteed to win.
        The ``cells`` array contains the discretized values of the features, and the ``taus`` list contains the threshold indices for each tree.
        """
        lc = np.zeros(self.n_classes, dtype=np.int64)
        D = self.D
        for t_idx, tree in enumerate(self.trees):
            s = tree.decide_cells(cells, taus[t_idx])
            R_t = D - 1 - t_idx
            if all(lc[s] >= lc[l] + R_t for l in range(self.n_classes) if l != s):
                return s
            lc[s] += 1
        return -1


def _sk_tree_to_tree(sk_tree) -> Tree:
    """
    Converts a scikit-learn decision tree into the project's ``Tree`` format.

    It copies the tree structure and converts leaf predictions into class labels.
    """
    t = sk_tree.tree_
    n = t.node_count
    feature = np.asarray(t.feature, dtype=np.int64).copy()
    threshold = np.asarray(t.threshold, dtype=np.float64).copy()
    left = np.asarray(t.children_left, dtype=np.int64).copy()
    right = np.asarray(t.children_right, dtype=np.int64).copy()
    value = np.asarray(t.value)          # (n_nodes, 1, n_classes)
    label = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if left[i] < 0:
            label[i] = int(np.argmax(value[i, 0]))
            feature[i] = -1
            threshold[i] = np.nan
    return Tree(feature=feature, threshold=threshold, left=left, right=right, label=label)



def load_forest(path: str, max_trees: int | None = None) -> Forest:
    """
    Loads a saved scikit-learn random forest and converts it into a ``Forest``.

    It loads the model, converts each tree using ``_sk_tree_to_tree``, and stores
    the forest information needed by the rest of the project.
    ``max_trees`` can be used to load only the first ``k`` trees.
    """
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
    """
    Recovers the mapping between the dataset labels and the model class indices.

    This is needed because the labels in the dataset may use a different order
    from the class indices used by the scikit-learn model.
    """


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
    """Load ``<name>_train.csv`` / ``<name>_test.csv`` with recovered labels."""
    

    base = os.path.join(root, name)
    tr = pd.read_csv(os.path.join(base, f"{name}_train.csv"))
    te = pd.read_csv(os.path.join(base, f"{name}_test.csv"))
    forest_blob_path = os.path.join(base, f"{name}_rf.pkl")

    import joblib

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
    """All dataset names under ``root`` that ship a forest pickle."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, f"{d}_rf.pkl")):
            out.append(d)
    return out