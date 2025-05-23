import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Sequence, Optional, Tuple

def blockproc3(
    im: np.ndarray,
    blksz: Sequence[int],
    fun: Callable[[np.ndarray], np.ndarray],
    border: Optional[Sequence[int]] = None,
    numworkers: int = 1
    ) -> np.ndarray:
    r"""
    Apply block processing to a 2D or 3D image volume, with optional borders
    and parallelism.

    Parameters
    ----------
    im : ndarray
        Input image, 2D or 3D array.
    blksz : sequence of int
        Block size in each dimension (len=2 or 3).
    fun : callable
        Function to apply to each block. Takes an array of shape
        block_size + 2*border and returns an array of the same shape.
    border : sequence of int, optional
        Border size in each dimension (len=2 or 3). Default is zeros.
    numworkers : int, default=1
        If >1, use that many processes for parallel block processing.

    Returns
    -------
    ndarray
        Processed image, same shape as im.
    """
    # Normalize dimensions to 3D
    orig_ndim = im.ndim
    if orig_ndim not in (2, 3):
        raise ValueError("Input image must be 2D or 3D")
    # Make a 3D view for uniform processing
    if orig_ndim == 2:
        im3 = im[:, :, np.newaxis]
    else:
        im3 = im
    D, H, W = im3.shape

    # Normalize block size and border
    b = list(blksz) + [1]*(3 - len(blksz))
    bx, by, bz = b[:3]
    if border is None:
        border = [0,0,0]
    bo = list(border) + [0]*(3 - len(border))
    ox, oy, oz = bo[:3]

    # Compute start (inclusive) and end (exclusive) indices for blocks
    xs = list(range(0, D, bx))
    ys = list(range(0, H, by))
    zs = list(range(0, W, bz))
    xe = [min(x + bx, D) for x in xs]
    ye = [min(y + by, H) for y in ys]
    ze = [min(z + bz, W) for z in zs]

    # Preallocate output
    im2_3 = np.empty_like(im3)

    # Prepare tasks: list of (ix, iy, iz, slices_without_border, slices_with_border)
    tasks = []
    for ix, x0 in enumerate(xs):
        x1 = xe[ix]
        xb0 = max(0, x0 - ox)
        xb1 = min(D, x1 + ox)
        for iy, y0 in enumerate(ys):
            y1 = ye[iy]
            yb0 = max(0, y0 - oy)
            yb1 = min(H, y1 + oy)
            for iz, z0 in enumerate(zs):
                z1 = ze[iz]
                zb0 = max(0, z0 - oz)
                zb1 = min(W, z1 + oz)

                # slices for border-extended block
                blk_ext = (slice(xb0, xb1), slice(yb0, yb1), slice(zb0, zb1))
                # slices for cropping back
                crop_x0 = x0 - xb0
                crop_x1 = crop_x0 + (x1 - x0)
                crop_y0 = y0 - yb0
                crop_y1 = crop_y0 + (y1 - y0)
                crop_z0 = z0 - zb0
                crop_z1 = crop_z0 + (z1 - z0)
                crop_slices = (slice(crop_x0, crop_x1),
                               slice(crop_y0, crop_y1),
                               slice(crop_z0, crop_z1))

                tasks.append((
                    (ix, iy, iz),
                    blk_ext,
                    (slice(x0, x1), slice(y0, y1), slice(z0, z1)),
                    crop_slices
                ))

    def _process(task):
        (ix, iy, iz), ext_slc, out_slc, crop_slc = task
        block = im3[ext_slc]
        processed = fun(block)
        # ensure output shape matches extended block shape
        if processed.shape != block.shape:
            raise ValueError(f"Block at {ix,iy,iz}: fun output shape {processed.shape} "
                             f"!= block shape {block.shape}")
        # crop to original block size
        cropped = processed[crop_slc]
        return (ix, iy, iz, cropped, out_slc)

    # Single- or multi-worker processing
    if numworkers > 1:
        with ProcessPoolExecutor(max_workers=numworkers) as execr:
            futures = [execr.submit(_process, t) for t in tasks]
            for fut in as_completed(futures):
                ix, iy, iz, cropped, out_slc = fut.result()
                im2_3[out_slc] = cropped
    else:
        for t in tasks:
            ix, iy, iz, cropped, out_slc = _process(t)
            im2_3[out_slc] = cropped

    # Return in original dimensionality
    if orig_ndim == 2:
        return im2_3[:, :, 0]
    return im2_3
