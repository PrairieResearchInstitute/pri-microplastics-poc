#!/usr/bin/env python

"""
detect_particles_fluorescence.py

Proof-of-concept detection of fluorescent microplastic particles in microscope
images using open-source Python tools.

This script is designed for fluorescence microscope images where particles appear
brighter than the local background. The method is inspired by source detection in
astronomical imaging: estimate the background, subtract it, estimate the noise
and detect objects above a user-defined sigma threshold.

Black regions are treated as no-data / outside the valid sample area. This avoids
spurious detections caused by sharp transitions near black image borders.

===============================================================================
BACKGROUND-SIGMA: WHAT IT MEANS
===============================================================================

The parameter:

    --background-sigma

controls the spatial scale used to estimate the smooth background illumination.

Internally, the script computes:

    background = gaussian(image, sigma=background_sigma)

where sigma is measured in pixels.

Conceptually, the algorithm separates the image into:

    image = smooth background + compact bright structures

The smooth background is estimated using a Gaussian blur. The blurred image
captures large-scale illumination gradients and slowly varying structure, while
small compact particles remain relatively unchanged.

The residual image is then:

    residual = image - smooth_background

This residual image is what is used for particle detection.

-------------------------------------------------------------------------------
WHY THIS PARAMETER IS IMPORTANT
-------------------------------------------------------------------------------

background_sigma determines what the algorithm considers to be background versus
particle signal.

If background_sigma is too small:

  - the smoothing follows the particles themselves
  - part of the particle signal gets subtracted away
  - faint particles may disappear

If background_sigma is too large:

  - large-scale gradients may not be fully removed
  - illumination structure may remain in the residual image
  - false detections may increase

A useful rule of thumb is:

    background_sigma approximately 3 to 10 times the particle size

For example, if typical particle diameters are around 5 to 10 pixels, reasonable
starting values might be:

    --background-sigma 20
    --background-sigma 25
    --background-sigma 40

This is analogous to astronomical source detection, where a smooth sky
background is estimated and subtracted before detecting stars or galaxies.

===============================================================================
"""

import argparse
from pathlib import Path
import time

import imageio.v3 as iio
import numpy as np
import pandas as pd

from skimage import io, color, measure, morphology, segmentation
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def now():
    return time.perf_counter()


def print_step_timing(step_name, start_time, enabled=True):
    if not enabled:
        return

    dt = now() - start_time
    print(f"[DONE] {step_name:<35s} {dt:8.3f} s", flush=True)


def read_rgb(path):
    """Read an image and return an RGB uint8 array."""
    img = io.imread(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        mx = np.max(img)
        if mx > 0:
            img = (255 * (img.astype(np.float32) / mx)).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def robust_sigma(x):
    """
    Estimate noise using the median absolute deviation.

    This is more robust than standard deviation when real bright particles
    are present.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def make_preview(image, preview_scale):
    """
    Downsample an image for faster writing of visual products.

    This does not affect full-resolution detection or catalog measurements.
    """
    if preview_scale >= 1.0:
        return image

    ny = int(image.shape[0] * preview_scale)
    nx = int(image.shape[1] * preview_scale)

    return resize(
        image,
        (ny, nx),
        anti_aliasing=True,
        preserve_range=True,
    )


def build_valid_data_mask(gray, nodata_threshold=0.02, nodata_dilation=5):
    """
    Build a valid-data mask from the grayscale image.

    Pixels darker than nodata_threshold are treated as no-data.
    """
    invalid = gray <= nodata_threshold

    if nodata_dilation > 0:
        invalid = morphology.binary_dilation(
            invalid,
            morphology.disk(nodata_dilation),
        )

    return ~invalid


def remove_edge_detections(mask, border_px):
    """Remove detections near the image edge."""
    if border_px <= 0:
        return mask

    mask = mask.copy()
    mask[:border_px, :] = False
    mask[-border_px:, :] = False
    mask[:, :border_px] = False
    mask[:, -border_px:] = False

    return mask


def get_overlay_color(color_name):
    """Return an RGB color tuple for overlay rendering."""
    colors = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
    }

    if color_name not in colors:
        raise ValueError(f"Unknown overlay color: {color_name}")

    return colors[color_name]


def detect_fluorescent_particles(
    input_image,
    outdir="out_fluorescence",
    sigma_threshold=5.0,
    min_area=10,
    background_sigma=25.0,
    closing_radius=1,
    nodata_threshold=0.02,
    nodata_dilation=5,
    edge_margin=0,
    overlay_color="green",
    preview_scale=1.0,
    write_mask=True,
    write_labels=True,
    write_residual=True,
    write_valid_mask=False,
    timing=False,
):
    """Detect bright fluorescent particle candidates in a microscope image."""

    t_total = now()

    in_path = Path(input_image)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem

    f_catalog = outdir / f"{base}_particles_catalog.csv"
    f_overlay = outdir / f"{base}_particles_overlay.jpg"
    f_mask = outdir / f"{base}_particles_mask.png"
    f_labels = outdir / f"{base}_particles_labels.tif"
    f_residual = outdir / f"{base}_residual.jpg"
    f_valid = outdir / f"{base}_valid_data_mask.png"

    # 1) Read image.
    t0 = now()
    rgb = read_rgb(str(in_path))
    rgb_f = rgb.astype(np.float32) / 255.0
    print_step_timing("read image", t0, timing)

    # 2) Convert RGB image to grayscale.
    #
    # Fluorescence detection is based primarily on brightness, not color.
    t0 = now()
    gray = color.rgb2gray(rgb_f)
    print_step_timing("convert to grayscale", t0, timing)

    # 3) Build valid-data mask.
    #
    # Black pixels are treated as no-data and excluded from both background/noise
    # estimation and final detection.
    t0 = now()
    valid = build_valid_data_mask(
        gray,
        nodata_threshold=nodata_threshold,
        nodata_dilation=nodata_dilation,
    )
    print_step_timing("build valid-data mask", t0, timing)

    # 4) Estimate the smooth illumination background.
    #
    # background_sigma controls the spatial scale of the smoothing:
    #
    #   small sigma  -> background follows local structure closely
    #   large sigma  -> background varies more smoothly
    #
    # The goal is to model large-scale illumination structure without modeling
    # compact fluorescent particles themselves.
    #
    # This is analogous to smooth sky-background estimation in astronomy.
    #
    # Invalid black pixels are replaced by the median valid intensity before
    # smoothing so they do not bias the background estimate.
    t0 = now()
    valid_median = float(np.median(gray[valid])) if np.any(valid) else 0.0
    gray_for_background = gray.copy()
    gray_for_background[~valid] = valid_median

    background = gaussian(
        gray_for_background,
        sigma=background_sigma,
        preserve_range=True,
    )

    residual = gray - background
    residual[~valid] = 0.0
    print_step_timing("estimate/subtract background", t0, timing)

    # 5) Estimate robust noise and threshold, using only valid pixels.
    #
    # This is analogous to source detection in astronomy:
    # detect pixels that are N sigma above the background.
    t0 = now()
    residual_valid = residual[valid]

    if residual_valid.size == 0:
        raise ValueError("No valid pixels found. Try lowering --nodata-threshold.")

    med = np.median(residual_valid)
    sig = robust_sigma(residual_valid)
    threshold = med + sigma_threshold * sig
    mask = (residual > threshold) & valid
    print_step_timing("threshold residual image", t0, timing)

    # 6) Optional edge clipping.
    t0 = now()
    mask = remove_edge_detections(mask, edge_margin)
    print_step_timing("remove edge detections", t0, timing)

    # 7) Morphological cleanup.
    #
    # Closing can connect nearby pixels from the same particle.
    # remove_small_objects filters out tiny detections/noise.
    t0 = now()
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing_radius))

    mask = mask & valid
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    print_step_timing("clean mask", t0, timing)

    # 8) Label connected components and measure image properties.
    t0 = now()
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=gray)
    print_step_timing("measure regions", t0, timing)

    # 9) Build and write catalog.
    t0 = now()
    rows = []
    for p in props:
        y, x = p.centroid
        minr, minc, maxr, maxc = p.bbox

        rows.append({
            "id": int(p.label),
            "x_centroid": float(x),
            "y_centroid": float(y),
            "area_px": int(p.area),
            "equiv_diameter_px": float(p.equivalent_diameter),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "bbox_min_row": int(minr),
            "bbox_min_col": int(minc),
            "bbox_max_row": int(maxr),
            "bbox_max_col": int(maxc),
            "mean_intensity": float(p.mean_intensity),
            "max_intensity": float(p.max_intensity),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("area_px", ascending=False)

    df.to_csv(f_catalog, index=False)
    print_step_timing("write catalog CSV", t0, timing)

    # 10) Write overlay image.
    #
    # The overlay is for visual inspection. It can be downsampled with
    # --preview-scale to make writing faster.
    t0 = now()
    overlay = segmentation.mark_boundaries(
        rgb_f,
        labels,
        color=get_overlay_color(overlay_color),
        mode="thick",
    )

    overlay_out = make_preview(overlay, preview_scale)

    iio.imwrite(
        f_overlay,
        (overlay_out * 255).astype(np.uint8),
        quality=90,
    )
    print_step_timing("write overlay JPG", t0, timing)

    # 11) Optional binary detection mask.
    if write_mask:
        t0 = now()
        iio.imwrite(
            f_mask,
            (mask.astype(np.uint8) * 255),
            compress_level=1,
        )
        print_step_timing("write mask PNG", t0, timing)

    # 12) Optional label image.
    if write_labels:
        t0 = now()
        iio.imwrite(
            f_labels,
            labels.astype(np.uint16),
        )
        print_step_timing("write labels TIFF", t0, timing)

    # 13) Optional residual image.
    #
    # This is useful for QA because it shows the background-subtracted image used
    # for detection.
    if write_residual:
        t0 = now()
        residual_show = rescale_intensity(
            residual,
            in_range="image",
            out_range=(0, 1),
        )

        residual_out = make_preview(residual_show, preview_scale)

        iio.imwrite(
            f_residual,
            (residual_out * 255).astype(np.uint8),
            quality=90,
        )
        print_step_timing("write residual JPG", t0, timing)

    # 14) Optional valid-data mask for QA.
    if write_valid_mask:
        t0 = now()
        iio.imwrite(
            f_valid,
            (valid.astype(np.uint8) * 255),
            compress_level=1,
        )
        print_step_timing("write valid-data mask PNG", t0, timing)

    total = now() - t_total

    print(f"Input: {in_path}")
    print(f"Output dir: {outdir.resolve()}")
    print(f"Detections: {int(labels.max())}")
    print(f"Threshold used: median + {sigma_threshold} sigma = {threshold:.6f}")
    print(f"Background sigma: {background_sigma} px")
    print(f"No-data threshold: {nodata_threshold}")
    print(f"No-data dilation: {nodata_dilation} px")
    print(f"Valid pixels: {int(valid.sum())} / {valid.size}")
    print(f"Edge margin: {edge_margin} px")
    print(f"Overlay color: {overlay_color}")
    print(f"Preview scale: {preview_scale}")
    print("Key files:")
    print(f"  {f_catalog.name}")
    print(f"  {f_overlay.name}")

    if write_mask:
        print(f"  {f_mask.name}")
    if write_labels:
        print(f"  {f_labels.name}")
    if write_residual:
        print(f"  {f_residual.name}")
    if write_valid_mask:
        print(f"  {f_valid.name}")

    print(f"Total runtime: {total:.3f} s")

    return {
        "catalog": f_catalog,
        "overlay": f_overlay,
        "mask": f_mask if write_mask else None,
        "labels": f_labels if write_labels else None,
        "residual": f_residual if write_residual else None,
        "valid_mask": f_valid if write_valid_mask else None,
        "detections": int(labels.max()),
        "dataframe": df,
        "threshold": threshold,
        "background_sigma": background_sigma,
        "nodata_threshold": nodata_threshold,
        "nodata_dilation": nodata_dilation,
        "edge_margin": edge_margin,
        "overlay_color": overlay_color,
        "total_time": total,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("input_image", help="Input fluorescence microscope image")
    ap.add_argument("--outdir", default="out_fluorescence", help="Output directory")

    ap.add_argument("--sigma-threshold", type=float, default=5.0,
                    help="Detection threshold in sigma above local background")
    ap.add_argument("--min-area", type=int, default=10,
                    help="Minimum detected object area in pixels")
    ap.add_argument("--background-sigma", type=float, default=25.0,
                    help="Gaussian sigma in pixels used to estimate the smooth background")
    ap.add_argument("--closing-radius", type=int, default=1,
                    help="Morphological closing radius in pixels")

    ap.add_argument("--nodata-threshold", type=float, default=0.02,
                    help="Pixels at or below this grayscale value are treated as no-data")
    ap.add_argument("--nodata-dilation", type=int, default=5,
                    help="Expand no-data regions by this many pixels")

    ap.add_argument("--edge-margin", type=int, default=0,
                    help="Ignore detections within this many pixels of the image edge")

    ap.add_argument("--overlay-color", default="green",
                    choices=["red", "green", "yellow", "cyan", "magenta"],
                    help="Color used for detected particles in the overlay")

    ap.add_argument("--preview-scale", type=float, default=1.0,
                    help="Scale factor for overlay/residual preview images")

    ap.add_argument("--no-mask", action="store_true",
                    help="Do not write the binary mask image")
    ap.add_argument("--no-labels", action="store_true",
                    help="Do not write the label image")
    ap.add_argument("--no-residual", action="store_true",
                    help="Do not write the residual image")
    ap.add_argument("--write-valid-mask", action="store_true",
                    help="Write the valid-data mask image for QA")

    ap.add_argument("--timing", action="store_true",
                    help="Print timing information in real time")

    args = ap.parse_args()

    if args.preview_scale <= 0 or args.preview_scale > 1:
        raise ValueError("--preview-scale must be > 0 and <= 1")

    detect_fluorescent_particles(
        input_image=args.input_image,
        outdir=args.outdir,
        sigma_threshold=args.sigma_threshold,
        min_area=args.min_area,
        background_sigma=args.background_sigma,
        closing_radius=args.closing_radius,
        nodata_threshold=args.nodata_threshold,
        nodata_dilation=args.nodata_dilation,
        edge_margin=args.edge_margin,
        overlay_color=args.overlay_color,
        preview_scale=args.preview_scale,
        write_mask=not args.no_mask,
        write_labels=not args.no_labels,
        write_residual=not args.no_residual,
        write_valid_mask=args.write_valid_mask,
        timing=args.timing,
    )


if __name__ == "__main__":
    main()
