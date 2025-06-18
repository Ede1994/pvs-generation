#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module generates maximally perceptually distinct colors based on the CIELAB color space.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from skimage.color import rgb2lab
from matplotlib.colors import to_rgb


# ***************************************************************************
# * Function
# ***************************************************************************
def distinguishable_colors(n_colors=int,
                           bg='white',
                           lab_func=None
                           ) -> np.ndarray:
    r"""
    Generate maximally perceptually distinct colors.

    Parameters:
    ----------
    n_colors : int
        Number of distinct colors to generate.
    bg : str, tuple, list, or np.ndarray, default='white'
        Background color(s) to ensure distinctness from. Can be a single color or a list of colors.
    lab_func : callable, optional
        Function to convert RGB colors to CIELAB. If None, uses skimage's rgb2lab.
    
    Returns:
    -------
    np.ndarray
        Array of shape (n_colors, 3) containing RGB colors in the range [0, 1].
    """
    if lab_func is None:
        lab_func = lambda rgb: rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    bg_rgb = _parse_bg(bg)
    bg_lab = lab_func(bg_rgb)

    # Generate grid of RGB colors
    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.column_stack((R.ravel(), G.ravel(), B.ravel()))

    if n_colors > len(rgb) // 3:
        raise ValueError("Too many colors requested; perceptual distinctiveness will be lost.")

    lab = lab_func(rgb)

    # Compute initial distance from background colors
    min_dist2 = np.full(len(rgb), np.inf)
    for bgl in bg_lab[:-1]:
        dist2 = np.sum((lab - bgl)**2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)

    colors = []
    last_lab = bg_lab[-1]

    for _ in range(n_colors):
        dist2 = np.sum((lab - last_lab) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        idx = np.argmax(min_dist2)
        colors.append(rgb[idx])
        last_lab = lab[idx]

    return np.array(colors)


def _parse_bg(bg) -> np.ndarray:
    r"""
    Parse background color(s) into RGB float array.

    Parameters:
    ----------
    bg : str, tuple, list, or np.ndarray
        Background color(s) to parse. Can be a single color or a list of colors.
    
    Returns:
    -------
    np.ndarray
        Array of RGB colors in the range [0, 1].
    """
    if isinstance(bg, (str, tuple, list, np.ndarray)):
        if isinstance(bg, (str, tuple)) or np.array(bg).ndim == 1:
            bg_list = [bg]
        else:
            bg_list = bg
        parsed = np.array([to_rgb(c) for c in bg_list])
        return parsed
    else:
        raise ValueError("Invalid background color specification.")
