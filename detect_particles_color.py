#!/usr/bin/env python
"""
detect_particles_color.py

Particle detection proof-of-concept using scikit-image.

Steps:
1) Read image
2) Estimate background color in Lab (k-means on downsampled image)
3) Compute per-pixel Lab distance (ΔE-like) to background
4) Build ROI mask (mustard region, ignore edge ring)
5) Threshold ΔE to detect particles + morphology cleanup
6) Label objects, measure region properties, write catalog + images

Outputs (basename comes from input file):
- [basename]_particles_catalog.csv
- [basename]_particles_overlay.png
- [basename]_particles_mask.png
- [basename]_particles_labels.tif


===============================================================================
BACKGROUND AND METHODOLOGY
===============================================================================

This script detects small plastic particles in microscope images using
classical image analysis techniques (no machine learning or AI models).

The core idea is simple:
  - The background has a fairly uniform mustard-like color
  - Plastic particles can have many colors, but are different from the background
  - We detect pixels whose color is sufficiently different from the background

The method relies on three main concepts:
  1) Background color estimation in Lab color space
  2) A Region Of Interest (ROI) mask
  3) Delta_E (color distance) thresholding

Each concept is explained below.

-------------------------------------------------------------------------------
1) BACKGROUND COLOR IN LAB SPACE (K-MEANS ON DOWNSAMPLED IMAGE)
-------------------------------------------------------------------------------

Why Lab color space?

Microscope images are stored in RGB (Red, Green, Blue), which is convenient
for display but not ideal for measuring color differences. In RGB:
  - Brightness and color are mixed together
  - Equal numeric differences do not correspond to equal perceived differences

Instead, the image is converted to Lab color space:
  - L : lightness (brightness)
  - a : green <-> red axis
  - b : blue <-> yellow axis

Lab space is approximately perceptually uniform, meaning that Euclidean
distance in Lab space corresponds well to how different two colors look
to the human eye.

What is the "background color"?

The background is assumed to occupy most of the image area and to have a
relatively uniform color (mustard-like). Rather than hard-coding a specific
color value, the script estimates the background color automatically.

How background estimation works:

  1) Convert the image to Lab color space
  2) Downsample the image to a smaller size (e.g. 400 x 400 pixels)
  3) Collect all pixel colors in Lab space
  4) Run k-means clustering (typically k = 3)
  5) Select the cluster with the largest number of pixels
  6) The center of that cluster is taken as the background color

Why downsample the image?

Downsampling is done only for background estimation and has several benefits:
  - Greatly reduces computation time
  - Makes k-means more stable
  - Does not affect accuracy, since background color varies slowly
  - We only care about color statistics, not spatial detail

The full-resolution image is used for all later steps.

-------------------------------------------------------------------------------
2) ROI MASK (REGION OF INTEREST)
-------------------------------------------------------------------------------

Why an ROI mask is needed:

Microscope images often include areas outside the actual sample:
  - Gray or black borders
  - Illumination falloff
  - Mount edges or background outside the sample

These regions can cause false detections if included in the analysis.

The ROI mask restricts detection to the actual sample region.

How the ROI mask is built:

  1) Identify pixels close to the background color
     (small Delta_E values)
  2) Label connected components of those pixels
  3) Keep the largest connected component
     (assumed to be the sample area)
  4) Fill holes inside that region
  5) Remove a small margin near the boundary
     (to avoid edge artifacts)

The result is a boolean mask:
  - True  = inside the sample
  - False = outside the sample

All particle detection is restricted to this region.

-------------------------------------------------------------------------------
3) DELTA_E (COLOR DISTANCE FROM BACKGROUND)
-------------------------------------------------------------------------------

What is Delta_E?

Delta_E (often written as ΔE) is a standard color-science metric that measures
the difference between two colors in Lab space.

In this script, Delta_E is computed as the Euclidean distance:

  Delta_E = sqrt( (L - L0)^2 + (a - a0)^2 + (b - b0)^2 )

where:
  (L0, a0, b0) is the estimated background color
  (L, a, b)   is the pixel color

Interpretation:
  - Small Delta_E : pixel color is similar to background
  - Large Delta_E : pixel color is different from background

  ΔE ~ 1      : difference is barely noticeable (even side-by-side)
  ΔE ~ 2–3    : noticeable if you look carefully
  ΔE ~ 4–5    : clearly different colors
  ΔE ~ 7–10   : very different
  ΔE > 15     : radically different colors

Why Delta_E works well here:

Plastic particles can have many colors and shapes, but they all differ
chromatically from the mustard background. By measuring Delta_E:
  - No prior knowledge of particle color is required
  - Detection reduces to a single, interpretable threshold

Thresholding:

Pixels are classified as particle candidates if:

  Delta_E > deltae_threshold

Lower threshold:
  - More sensitive
  - More detections (and possibly more noise)

Higher threshold:
  - More conservative
  - Fewer detections

-------------------------------------------------------------------------------
SUMMARY
-------------------------------------------------------------------------------

The full pipeline is:

  1) Convert image to Lab color space
  2) Estimate background color using k-means clustering
  3) Compute Delta_E for every pixel
  4) Build an ROI mask from background pixels
  5) Threshold Delta_E to find non-background pixels
  6) Clean up with morphological operations
  7) Label connected components
  8) Measure particle properties
  9) Save catalog and visualization images

All steps are deterministic, interpretable, and implemented using
fully open-source Python libraries.
===============================================================================

"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from skimage import io, color, measure, morphology, segmentation
from skimage.transform import resize

from scipy.ndimage import binary_fill_holes, distance_transform_edt
from sklearn.cluster import KMeans


def now() -> float:
    return time.perf_counter()


def print_step_timing(step_name: str, start_time: float, enabled: bool = True) -> None:
    if not enabled:
        return
    dt = now() - start_time
    print(f"[DONE] {step_name:<35s} {dt:8.3f} s", flush=True)


def read_rgb(path: str) -> np.ndarray:
    img = io.imread(path)

    # Ensure RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Ensure uint8
    if img.dtype != np.uint8:
        mx = np.max(img)
        if mx > 0:
            img = (255 * (img.astype(np.float32) / mx)).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def make_preview(image: np.ndarray, preview_scale: float) -> np.ndarray:
    """
    Downsample an image for faster writing of visual preview products.

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


def get_overlay_color(color_name: str) -> tuple[float, float, float]:
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


def estimate_background_lab(rgb_u8: np.ndarray, n_clusters: int = 3, downsample_to: int = 400) -> np.ndarray:
    """
    Estimate dominant (background) Lab color via k-means on a downsampled image.
    Assumes background is the most common color cluster after removing border/glare.
    """
    small = resize(
        rgb_u8,
        (downsample_to, downsample_to),
        anti_aliasing=True,
        preserve_range=False
    ).astype(np.float32)

    lab = color.rgb2lab(small)  # expects float image in [0,1]
    flat_lab = lab.reshape(-1, 3)

    flat_rgb = small.reshape(-1, 3)
    brightness = flat_rgb.mean(axis=1)

    # Exclude very dark border and near-white glare
    keep = (brightness > 0.10) & (brightness < 0.98)
    X = flat_lab[keep]

    if X.shape[0] < 100:
        # Fallback: if mask goes wrong, use global median Lab
        return np.median(flat_lab, axis=0)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    counts = np.bincount(labels)

    # Most frequent cluster center = background
    return centers[np.argmax(counts)]


def build_roi_mask(dE: np.ndarray, bg_tol: float = 8.0, edge_margin_px: int = 30) -> np.ndarray:
    """
    Build a mask that roughly captures the mustard sample area and removes the edge ring.
    - mustard pixels: dE < bg_tol
    - keep largest connected component
    - fill holes
    - shrink away boundary by edge_margin_px (via distance transform)
    """
    mustard = dE < bg_tol
    lab = measure.label(mustard)

    if lab.max() == 0:
        return np.ones(dE.shape, dtype=bool)

    regions = measure.regionprops(lab)
    largest = max(regions, key=lambda r: r.area)
    roi0 = (lab == largest.label)

    roi0 = binary_fill_holes(roi0)

    if edge_margin_px <= 0:
        return roi0.astype(bool)

    dist = distance_transform_edt(roi0)
    roi = dist > edge_margin_px

    # If the margin nukes the ROI, fall back
    if roi.sum() < 1000:
        roi = roi0.astype(bool)

    return roi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_image", help="Input microscope image")
    ap.add_argument("--outdir", default="out_color", help="Output directory")

    ap.add_argument("--deltae-threshold", type=float, default=5.0,
                    help="Lab-distance from background (higher -> fewer detections)")
    ap.add_argument("--auto-deltae", type=float, default=None,
                    help="Automatically set deltaE threshold to mean_bg + Nsigma*std_bg "
                    "using background pixels inside ROI. Example: --auto-deltae 5")
    ap.add_argument("--bg-tol", type=float, default=8.0,
                    help="Tolerance for defining mustard/background (used for ROI)")
    ap.add_argument("--edge-margin", type=int, default=30,
                    help="Ignore this many pixels from ROI boundary")
    ap.add_argument("--min-area", type=int, default=10,
                    help="Minimum object area (pixels) to keep")
    ap.add_argument("--closing-radius", type=int, default=1,
                    help="Disk radius for binary closing after thresholding")
    ap.add_argument("--overlay-color", default="red",
                    choices=["red", "green", "yellow", "cyan", "magenta"],
                    help="Color used for detected particles in the overlay")
    ap.add_argument("--preview-scale", type=float, default=1.0,
                    help="Scale factor for overlay preview images")

    ap.add_argument("--timing", action="store_true",
                    help="Print timing for each major step and total runtime")

    args = ap.parse_args()

    if args.timing and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)

    if args.preview_scale <= 0 or args.preview_scale > 1:
        raise ValueError("--preview-scale must be > 0 and <= 1")

    t_total0 = now()

    in_path = Path(args.input_image)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem
    f_catalog = outdir / f"{base}_particles_catalog.csv"
    f_labels = outdir / f"{base}_particles_labels.tif"
    f_overlay = outdir / f"{base}_particles_overlay.png"
    f_mask = outdir / f"{base}_particles_mask.png"

    # 1) Read
    t0 = now()
    rgb = read_rgb(str(in_path))
    print_step_timing("read image", t0, args.timing)

    rgb_f = rgb.astype(np.float32) / 255.0

    # 2) Background estimate
    t0 = now()
    bg_lab = estimate_background_lab(rgb, n_clusters=3, downsample_to=400)
    print_step_timing("estimate background Lab", t0, args.timing)

    # 3) Compute dE
    t0 = now()
    lab_full = color.rgb2lab(rgb_f)
    dE = np.linalg.norm(lab_full - bg_lab, axis=2)
    print_step_timing("compute dE", t0, args.timing)

    # 4) ROI
    t0 = now()
    roi = build_roi_mask(dE, bg_tol=args.bg_tol, edge_margin_px=args.edge_margin)
    print_step_timing("compute ROI", t0, args.timing)

    # 5) Mask + cleanup
    t0 = now()
    mask = roi & (dE > args.deltae_threshold)

    if args.closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(args.closing_radius))

    mask = morphology.remove_small_objects(mask, min_size=args.min_area)
    print_step_timing("compute mask", t0, args.timing)

    # 6) Regionprops
    t0 = now()
    labels = measure.label(mask)
    props = measure.regionprops(labels, intensity_image=dE)
    print_step_timing("measure regionprops", t0, args.timing)

    # 7) Catalog (CSV)
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
            "mean_deltaE": float(p.mean_intensity),
            "max_deltaE": float(p.max_intensity),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("area_px", ascending=False)

    df.to_csv(f_catalog, index=False)
    print_step_timing("write catalog CSV", t0, args.timing)

    # 8) Write labels TIFF
    t0 = now()
    io.imsave(f_labels, labels.astype(np.uint16), check_contrast=False)
    print_step_timing("write labels TIFF", t0, args.timing)

    # 9) Write mask PNG
    t0 = now()
    io.imsave(f_mask, (mask.astype(np.uint8) * 255), check_contrast=False)
    print_step_timing("write mask PNG", t0, args.timing)

    # 10) Write overlay PNG
    t0 = now()
    overlay = segmentation.mark_boundaries(
        rgb_f,
        labels,
        color=get_overlay_color(args.overlay_color),
        mode="thick")
    overlay_out = make_preview(overlay, args.preview_scale)
    io.imsave(
        f_overlay,
        (overlay_out * 255).astype(np.uint8),
        check_contrast=False)
    print_step_timing("write overlay PNG", t0, args.timing)

    total_time = now() - t_total0

    n_det = int(labels.max())
    print(f"Input: {in_path}")
    print(f"Output dir: {outdir.resolve()}")
    print(f"Detections: {n_det}")
    print(f"Overlay color: {args.overlay_color}")
    print(f"Preview scale: {args.preview_scale}")
    print("Key files:")
    print(f"  {f_catalog.name}")
    print(f"  {f_overlay.name}")
    print(f"  {f_mask.name}")
    print(f"  {f_labels.name}")

    print(f"Total runtime: {total_time:.3f} s")


if __name__ == "__main__":
    main()
