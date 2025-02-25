# Paint-by-Numbers SVG Generator

This repository contains a complete Streamlit app that converts an image into a paint-by-numbers SVG template. The pipeline includes:

1. K-means clustering for color quantization.
2. Facet building via connected components.
3. Small facet pruning based on detail level.
4. Border detection and segmentation.
5. Label placement at facet centroids.
6. SVG generation with polygon outlines and labels.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/paintbynumbers-app.git
   cd paintbynumbers-app
