from __future__ import annotations
import os
import warnings
from dataclasses import dataclass, field
import numpy as np



class Tree:
    """A single decision tree in the ``(nu, lambda_E)``.

    Attributes
    ----------
    feature, threshold
        ``nu``: for an internal node ``i``, ``feature[i]`` is the tested feature
        and ``threshold[i]`` the real threshold of the ``<=`` test.  Both are
        ``-1`` / ``nan`` on leaves.
    left, right
        Children.  ``left`` is the **true** branch (``<=``, ``d_i = 1`` in
        Definition 6), ``right`` the **false** branch (``>``, ``d_i = 0``).
    label
        ``lambda_E``: the class index of a leaf, ``-1`` on internal nodes.
    """

    