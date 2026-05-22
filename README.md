# Microplastic Particle Detection (Proof of Concept)

This repository contains two open-source Python scripts for detecting
microplastic particles in microscope images with classical image analysis.

- `detect_particles_color.py`: color-based detection for bright-field images
- `detect_particles_fluorescence.py`: brightness-based detection for
  fluorescence images

Both pipelines are deterministic, interpretable, and easy to tune.

## Requirements

The scripts use standard scientific Python packages:

- `numpy`
- `pandas`
- `scikit-image`
- `scipy`
- `scikit-learn`
- `imageio`

## `detect_particles_color.py`

Color-based particle detection for images where particles differ in color from a
mostly uniform background.

### Method

1. Convert the image to Lab color space.
2. Estimate the background color with k-means on a downsampled image.
3. Compute per-pixel color distance from the background.
4. Build a region-of-interest mask for the sample area.
5. Threshold, clean, and label detected particles.

### Key options

- `--deltae-threshold`: color-distance threshold for detection
- `--auto-deltae`: optional automatic threshold scaling
- `--bg-tol`: tolerance used to identify background-like pixels
- `--edge-margin`: excludes detections near the sample boundary
- `--min-area`: removes very small detections
- `--closing-radius`: morphology cleanup strength
- `--overlay-color`: overlay color for visualization
- `--preview-scale`: downsamples the overlay preview only

### Outputs

Written to `out_color/` by default:

- `[input]_particles_catalog.csv`
- `[input]_particles_overlay.png`
- `[input]_particles_mask.png`
- `[input]_particles_labels.tif`

### Example

```bash
python detect_particles_color.py \
  data/Bright_Field/1985_02_020226.jpeg \
  --deltae-threshold 5.0 \
  --min-area 3
```

## `detect_particles_fluorescence.py`

Brightness-based detection for fluorescence images where particles appear
brighter than the local background.

### Method

1. Convert the image to grayscale intensity.
2. Mask black or no-data regions.
3. Estimate the smooth background with a Gaussian blur.
4. Subtract the background to form a residual image.
5. Estimate noise robustly and detect bright pixels above a sigma threshold.
6. Clean, label, and measure connected components.

### Key options

- `--sigma-threshold`: detection threshold in units of background noise
- `--background-sigma`: smoothing scale for background estimation
- `--nodata-threshold`: defines black or invalid pixels
- `--nodata-dilation`: expands the invalid-data mask
- `--edge-margin`: removes detections near image edges
- `--min-area`: removes very small detections
- `--overlay-color`: overlay color for visualization
- `--preview-scale`: downsamples preview products only

### Outputs

Written to `out_fluorescence/` by default:

- `[input]_particles_catalog.csv`
- `[input]_particles_overlay.jpg`
- `[input]_particles_mask.png` unless `--no-mask`
- `[input]_particles_labels.tif` unless `--no-labels`
- `[input]_residual.jpg` unless `--no-residual`
- `[input]_valid_data_mask.png` if `--write-valid-mask`

### Example

```bash
python detect_particles_fluorescence.py \
  data/Fluorescence/sample.png \
  --sigma-threshold 5.0 \
  --background-sigma 25 \
  --min-area 10
```

## Notes

- Use `detect_particles_color.py` when the particle signal is mainly color
  contrast against a stable background.
- Use `detect_particles_fluorescence.py` when the signal is mainly brightness
  and illumination varies across the field.
