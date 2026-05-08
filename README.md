# Microplastic Particle Detection (Proof of Concept)

This repository contains proof-of-concept scripts for detecting small plastic
particles in microscope images using classical image analysis techniques,
without machine learning or proprietary software.

The repository currently contains two complementary pipelines:

1. `detect_particles.py`
   Detects particles based on color differences from the background.

2. `detect_particles_fluorescence.py`
   Detects fluorescent particles based on brightness relative to the local
   background.

Both approaches are:

- fully open source
- deterministic
- interpretable
- easy to tune
- scientifically reproducible

---

# Color-Based Particle Detection (`detect_particles.py`)

## Conceptual Overview

The key observation behind this method is simple:

- The background has a relatively uniform mustard-like color
- Plastic particles can be many different colors
- Particles are detected as regions whose color differs sufficiently from the background

The method relies on three main ideas:

1. Background color estimation in Lab color space
2. Construction of a Region of Interest (ROI) mask
3. Pixel-wise color distance (Delta_E) thresholding

Each is described below.

---

## 1. Background Color in Lab Space

### Why Lab instead of RGB?

Microscope images are usually stored in RGB (Red, Green, Blue). While RGB is
useful for display, it is not ideal for measuring color differences:

- Brightness and color are mixed together
- Equal numeric differences do not correspond to equal perceived differences

The script converts the image to Lab color space, which separates brightness
from color:

- L: lightness
- a: green ↔ red axis
- b: blue ↔ yellow axis

Lab space is approximately perceptually uniform, meaning Euclidean distance in
Lab space corresponds well to how different two colors appear to humans.

---

### How the background color is estimated

Rather than hard-coding a specific mustard color, the background color is
estimated automatically.

The procedure is:

1. Convert the image to Lab color space
2. Downsample the image to a fixed size (e.g. 400 × 400 pixels)
3. Collect all pixel colors in Lab space
4. Apply k-means clustering (typically with 3 clusters)
5. Identify the cluster with the largest number of pixels
6. Use that cluster center as the background color

The assumption is that the background occupies the largest fraction of the image.

---

### Why downsample the image?

Downsampling is used only for background estimation and is safe because:

- The background color varies slowly in space
- We care about color statistics, not fine spatial detail
- Tens of thousands of pixels are still available after downsampling

Downsampling:

- Greatly reduces computation time
- Improves clustering stability
- Does not affect detection accuracy

All subsequent analysis uses the full-resolution image.

---

## 2. ROI Mask (Region of Interest)

### Why an ROI mask is needed

Raw microscope images often include areas outside the actual sample, such as:

- Borders or mounts
- Illumination falloff
- Non-sample background

Including these areas can lead to false detections.

The ROI mask restricts analysis to the actual sample region.

---

### How the ROI mask is built

1. Identify pixels close to the background color (small Delta_E)
2. Label connected components of those pixels
3. Keep the largest connected component (assumed to be the sample)
4. Fill holes inside the region
5. Remove a small margin near the boundary to avoid edge artifacts

The result is a boolean mask:

- `True` inside the sample
- `False` outside the sample

All particle detection is performed inside this mask.

---

## 3. Delta_E (Color Distance from Background)

### What is Delta_E?

Delta_E (ΔE) is a standard metric in color science that measures the difference
between two colors in Lab space.

In this script, Delta_E is computed as the Euclidean distance:

```text
Delta_E = sqrt( (L - L0)^2 + (a - a0)^2 + (b - b0)^2 )
```

where:

- `(L0, a0, b0)` is the estimated background color
- `(L, a, b)` is the pixel color

Interpretation:

- Small Delta_E: pixel color is similar to background
- Large Delta_E: pixel color is different from background

Typical interpretation:

- ΔE ~ 1: barely noticeable
- ΔE ~ 2–3: slightly noticeable
- ΔE ~ 4–5: clearly different
- ΔE ~ 7–10: very different
- ΔE > 15: radically different

---

### Why Delta_E works well here

Plastic particles can have many colors and shapes, but they all differ
chromatically from the mustard background.

By measuring Delta_E:

- No prior knowledge of particle color is required
- Detection reduces to a single interpretable threshold

Pixels are classified as candidate particles if:

```text
Delta_E > deltae_threshold
```

Lower threshold:

- More sensitive
- More detections
- More noise

Higher threshold:

- More conservative
- Fewer detections

---

## Outputs

The script writes:

- `[input]_particles_catalog.csv`
- `[input]_particles_overlay.png`
- `[input]_particles_mask.png`
- `[input]_particles_labels.tif`

---

## Example Command

```bash
python detect_particles.py \
  data/Bright_Field/1985_02_020226.jpeg \
  --deltae-threshold 5.0 \
  --min-area 3
```

This command will:

- Read the input image `Sample2_010626.png`
- Detect pixels whose color differs from the estimated background by more than a Delta_E of 5.0
- Keep detected regions with an area of at least 3 pixels
- Write output products into the default output directory (`out_particles/`)

The outputs include:

- a CSV catalog of detected particles
- a binary mask image
- a labeled segmentation image
- an overlay image showing detected particles

Thresholds such as `--deltae-threshold` and `--min-area` can be adjusted depending on image quality, illumination conditions and desired sensitivity.

---

# Fluorescence-Based Particle Detection (`detect_particles_fluorescence.py`)

In addition to the color-based segmentation approach implemented in
`detect_particles.py`, this repository also includes a second proof-of-concept
pipeline for detecting fluorescent particles using image brightness rather than
color differences.

This method is intended for fluorescence microscope images where particles appear
brighter than the local background.

The algorithm is conceptually similar to how compact sources (stars or galaxies)
are detected in astronomical images:

- estimate the smooth background
- subtract the background
- estimate the noise level
- detect statistically significant bright sources

This approach works well when:

- particles fluoresce strongly
- illumination varies smoothly
- the background is not uniform in color
- color information is less important than brightness

The implementation is fully open source and uses:

- numpy
- scikit-image
- scipy
- pandas
- imageio

No machine learning or proprietary software is required.

---

## Conceptual Overview

The fluorescence detection pipeline works as follows:

1. Read the microscope image
2. Convert the image to grayscale intensity
3. Identify valid image regions (exclude black/no-data areas)
4. Estimate the smooth illumination background
5. Subtract the background
6. Estimate the image noise level
7. Detect statistically significant bright pixels
8. Clean the binary mask using morphology
9. Label connected regions
10. Measure particle properties
11. Write catalogs and visualization products

---

## 1. Grayscale Intensity Image

Unlike the color-based approach in `detect_particles.py`, fluorescence detection
is based primarily on image brightness.

The RGB image is converted to a grayscale intensity image:

```python
gray = color.rgb2gray(rgb_f)
```

This collapses the RGB channels into a single image where:

- bright pixels correspond to stronger fluorescence
- dark pixels correspond to weaker fluorescence

The grayscale image becomes the basis for all subsequent analysis.

---

## 2. Valid Data Mask (No-Data Rejection)

Microscope fluorescence images often contain:

- black borders
- vignetted regions
- areas outside the microscope field
- stitching artifacts
- empty detector regions

These regions are not scientifically meaningful and should not be interpreted as
real image background.

The script therefore constructs a valid-data mask.

Pixels below a threshold are treated as no-data:

```python
valid = gray > nodata_threshold
```

Typical values:

```bash
--nodata-threshold 0.02
```

where grayscale intensities are normalized to the range `[0,1]`.

---

### Why This Matters

Without a valid-data mask:

- sharp black edges generate artificial gradients
- background subtraction becomes unstable
- spurious detections appear near image boundaries

The valid-data mask ensures that:

- background estimation uses only real image pixels
- noise estimation ignores invalid regions
- detections occur only inside scientifically valid areas

---

### No-Data Dilation

The script can optionally expand the no-data region slightly:

```bash
--nodata-dilation 5
```

This is useful because:

- interpolation artifacts often exist near sharp black boundaries
- edge ringing can create false detections
- invalid borders are rarely perfectly sharp

---

## 3. Smooth Background Estimation

Real fluorescence microscope images rarely have perfectly uniform illumination.

Common large-scale structures include:

- illumination gradients
- vignetting
- optical falloff
- detector sensitivity variation
- broad fluorescent background

To detect particles robustly, the script estimates a smooth background image.

This is done using a Gaussian filter:

```python
background = gaussian(gray, sigma=background_sigma)
```

The parameter controlling the smoothing scale is:

```bash
--background-sigma
```

This is one of the most important parameters in the pipeline.

---

## 4. What `background_sigma` Means

`background_sigma` controls the spatial scale used to estimate the smooth
illumination background.

Conceptually, the image is separated into:

```text
image = smooth background + compact bright structures
```

The Gaussian blur captures:

- large-scale smooth structure
- illumination gradients
- broad background variations

while leaving:

- compact fluorescent particles
- small sharp structures

relatively unchanged.

The residual image is then:

```text
residual = image - smooth_background
```

This residual image is what is actually used for detection.

---

### Why This Parameter Is Important

`background_sigma` determines what the algorithm considers:

- background
- versus particle signal

If `background_sigma` is too small:

- the smoothing follows the particles themselves
- particle signal gets subtracted away
- faint particles may disappear

If `background_sigma` is too large:

- illumination gradients may remain
- residual background structure increases
- false detections may occur

---

### Rule of Thumb

A useful heuristic is:

```text
background_sigma ≈ 3–10 × particle size
```

Typical starting values:

```bash
--background-sigma 20
--background-sigma 25
--background-sigma 40
```

---

## 5. Background Subtraction

After estimating the smooth background, the script computes:

```python
residual = gray - background
```

The residual image contains:

- compact bright structures
- small-scale fluorescence features
- reduced large-scale illumination structure

Bright particles become significantly easier to detect after subtraction.

---

## 6. Robust Noise Estimation

The residual image still contains:

- noise
- detector fluctuations
- illumination residuals
- real fluorescent particles

The script estimates the noise level robustly using the Median Absolute
Deviation (MAD):

```python
sigma = 1.4826 * median(abs(x - median(x)))
```

This is preferred over ordinary standard deviation because:

- real bright particles bias the standard deviation upward
- MAD is much more stable in the presence of outliers

Only valid pixels are used in the noise estimate.

---

## 7. Sigma Threshold Detection

Pixels are classified as candidate particles if:

```text
residual > median + sigma_threshold × sigma
```

controlled by:

```bash
--sigma-threshold
```

Example:

```bash
--sigma-threshold 5
```

means:

- detect pixels at least 5 sigma above the local background

---

### Interpreting Sigma Thresholds

Lower thresholds:

```bash
--sigma-threshold 3
```

- more sensitive
- detects fainter particles
- increases false positives

Higher thresholds:

```bash
--sigma-threshold 7
```

- more conservative
- fewer false positives
- may miss faint particles

Typical useful values:

```bash
--sigma-threshold 4
--sigma-threshold 5
--sigma-threshold 6
```

---

## 8. Morphological Cleanup

The threshold operation produces a binary mask.

The script then applies:

- binary closing
- small-object removal

to improve segmentation quality.

---

### Binary Closing

Controlled by:

```bash
--closing-radius
```

This operation:

- connects nearby bright pixels
- fills tiny gaps
- produces smoother particle masks

---

### Minimum Area Filtering

Controlled by:

```bash
--min-area
```

Objects smaller than this size are removed.

This suppresses:

- isolated hot pixels
- tiny noise detections
- compression artifacts

Example:

```bash
--min-area 10
```

keeps only objects with at least 10 pixels.

---

## 9. Connected Component Labeling

After cleanup, connected pixels are grouped into individual objects:

```python
labels = measure.label(mask)
```

Each detected particle receives:

- a unique integer ID
- a separate measured region

---

## 10. Particle Measurements

For each detected object, the script measures:

- centroid position
- area
- equivalent diameter
- eccentricity
- solidity
- bounding box
- mean intensity
- maximum intensity

The measurements are written to:

```text
[input]_particles_catalog.csv
```

At present, measurements are reported in pixels.

If microscope calibration metadata becomes available, the pipeline can easily be
extended to compute:

- particle diameter in micrometers
- particle area in square micrometers
- size distributions in physical units

---

## 11. Output Products

### Particle Catalog

```text
[input]_particles_catalog.csv
```

Contains:

- one row per detected object
- measured particle properties

---

### Overlay Image

```text
[input]_particles_overlay.jpg
```

Shows:

- original microscope image
- detected particles highlighted in color

Overlay colors are configurable:

```bash
--overlay-color green
--overlay-color yellow
```

---

### Binary Detection Mask

```text
[input]_particles_mask.png
```

Binary image:

- white = detected pixels
- black = background

---

### Label Image

```text
[input]_particles_labels.tif
```

Each detected object has:

- a unique integer label

Useful for:

- quantitative segmentation
- region analysis
- downstream workflows

---

### Residual Image

```text
[input]_residual.jpg
```

Shows:

- the background-subtracted image
- what the detection algorithm actually sees

This is extremely useful for:

- debugging
- parameter tuning
- understanding failures

---

### Valid Data Mask

Optional QA product:

```text
[input]_valid_data_mask.png
```

Shows:

- valid scientific image regions
- excluded no-data regions

Generated using:

```bash
--write-valid-mask
```

---

## Performance Considerations

Large microscope images can be tens of megapixels.

Writing full-resolution PNG images can become slow.

The script therefore:

- writes preview products as JPEG by default
- optionally downsamples preview products

using:

```bash
--preview-scale 0.5
```

This:

- speeds up output generation
- reduces file size
- does not affect scientific measurements

All measurements remain full resolution.

---

## Example Command

```bash
python detect_particles_fluorescence.py \
  data/Fluorescense/1985_02_022626_fluorescense.jpeg \
  --sigma-threshold 5 \
  --background-sigma 25 \
  --min-area 10 \
  --nodata-threshold 0.02 \
  --nodata-dilation 5 \
  --overlay-color green \
  --preview-scale 0.5 \
  --write-valid-mask \
  --timing
```

---

## Summary

The fluorescence pipeline follows a classical scientific image-analysis approach:

1. Build a valid-data mask
2. Estimate the smooth background
3. Subtract the background
4. Estimate image noise
5. Detect statistically significant bright sources
6. Clean the segmentation mask
7. Measure object properties
8. Write reproducible outputs

The method is:

- deterministic
- interpretable
- fully open source
- easy to tune
- easy to validate scientifically

No proprietary software or machine learning models are required.
