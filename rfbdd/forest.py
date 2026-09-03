from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field
import numpy as np


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