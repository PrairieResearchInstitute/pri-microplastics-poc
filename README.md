# Microplastic Particle Detection (Proof of Concept)

This repository contains a proof-of-concept script for detecting small plastic
particles in microscope images using **classical image analysis**, without
machine learning or proprietary software.

The approach is fully open source, interpretable, and designed to be easy to
explain to non–computer-vision specialists.

## Conceptual Overview

The key observation behind this method is simple:

- The background has a relatively uniform mustard-like color
- Plastic particles can be many different colors
- Particles are detected as regions whose color differs sufficiently from the background

The method relies on three main ideas:

1. Background color estimation in **Lab color space**
2. Construction of a **Region of Interest (ROI) mask**
3. Pixel-wise color distance (**Delta_E**) thresholding

Each is described below.

## 1. Background Color in Lab Space

### Why Lab instead of RGB?

Microscope images are usually stored in RGB (Red, Green, Blue). While RGB is
useful for display, it is not ideal for measuring color differences:

- Brightness and color are mixed together
- Equal numeric differences do not correspond to equal perceived differences

The script converts the image to **Lab color space**, which separates brightness
from color:

- **L**: lightness
- **a**: green ↔ red axis
- **b**: blue ↔ yellow axis

Lab space is approximately perceptually uniform, meaning Euclidean distance in
Lab space corresponds well to how different two colors appear to humans.

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

## 2. ROI Mask (Region of Interest)

### Why an ROI mask is needed

Raw microscope images often include areas outside the actual sample, such as:

- Borders or mounts
- Illumination falloff
- Non-sample background

Including these areas can lead to false detections.

The ROI mask restricts analysis to the actual sample region.

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


## 3. Delta_E (Color Distance from Background)

### What is Delta_E?

Delta_E (ΔE) is a standard metric in color science that measures the difference
between two colors in Lab space.

In this script, Delta_E is computed as the Euclidean distance:


## 4. Example Usage

Below is a simple example showing how to run the particle detection script on one of the sample microscope images.

```bash
python detect_particles.py data/Sample2_010626.png \
  --deltae-threshold 5.0 \
  --min-area 3
```

This command will:
	- Read the input image `Sample2_010626.png`
	- Detect pixels whose color differs from the estimated background by more than a Delta_E of 5.0
	- Keep detected regions with an area of at least 3 pixels
	- Write the output files to the default output directory (`out_particles/`), including:
	  - a CSV catalog of detected particles
	  - a binary mask image
	  - a labeled image where each particle has a unique ID
	  - an overlay image highlighting detected particles on top of the original image

Thresholds such as `--deltae-threshold` and `--min-area` can be adjusted depending on image quality, illumination conditions, and the desired sensitivity of the detection.
