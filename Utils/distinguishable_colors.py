import numpy as np
from skimage.color import rgb2lab
from matplotlib.colors import to_rgb

def distinguishable_colors(n_colors, bg='white', lab_func=None):
    """
    Generate maximally perceptually distinct colors.

    Parameters:
    - n_colors: int, number of distinct colors to generate
    - bg: background color(s) (RGB tuple, hex string, color name, or list of these)
    - lab_func: optional custom RGB to Lab function

    Returns:
    - colors: (n_colors x 3) ndarray of RGB values
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


def _parse_bg(bg):
    """Parse background color(s) into RGB float array."""
    if isinstance(bg, (str, tuple, list, np.ndarray)):
        if isinstance(bg, (str, tuple)) or np.array(bg).ndim == 1:
            bg_list = [bg]
        else:
            bg_list = bg
        parsed = np.array([to_rgb(c) for c in bg_list])
        return parsed
    else:
        raise ValueError("Invalid background color specification.")
