import cv2
import numpy as np
from PIL import Image
import svgwrite

def quantize_image(image_bgr, k=10):
    """Perform K-means clustering to reduce the image to k colors."""
    h, w = image_bgr.shape[:2]
    data = image_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape((h, w, 3))
    return quantized, centers

def build_facets(quantized_img):
    """
    Build facets by grouping connected pixels of the same color.
    Returns:
      label_img: 2D array with facet labels
      facets: dict mapping label -> list of (y,x) coordinates
    """
    h, w = quantized_img.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    label_img = -1 * np.ones((h, w), dtype=int)
    facets = {}
    label = 0

    def neighbors(y, x):
        for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx

    for i in range(h):
        for j in range(w):
            if not visited[i, j]:
                stack = [(i, j)]
                visited[i, j] = True
                facet_pixels = [(i, j)]
                label_img[i, j] = label
                orig_color = quantized_img[i, j]
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in neighbors(cy, cx):
                        if not visited[ny, nx] and np.array_equal(quantized_img[ny, nx], orig_color):
                            visited[ny, nx] = True
                            label_img[ny, nx] = label
                            stack.append((ny, nx))
                            facet_pixels.append((ny, nx))
                facets[label] = facet_pixels
                label += 1
    return label_img, facets

def prune_facets(label_img, facets, min_size):
    """Merge small facets into neighbors by simply discarding small ones.
       Returns the set of facet labels that are large enough."""
    keep = {label for label, pixels in facets.items() if len(pixels) >= min_size}
    # For pixels in small facets, assign them to a neighboring facet (simple approach)
    h, w = label_img.shape
    for i in range(h):
        for j in range(w):
            if label_img[i, j] not in keep:
                # Look at 4-neighbors and choose the first facet that is kept
                for ny, nx in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ny < h and 0 <= nx < w and label_img[ny, nx] in keep:
                        label_img[i, j] = label_img[ny, nx]
                        break
    return keep

def get_contour(label_img, facet_label):
    """Return the contour points for a given facet label using cv2.findContours."""
    mask = (label_img == facet_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return approx.reshape(-1, 2)  # (x,y) coordinates
    return None

def compute_centroid(points):
    pts = np.array(points)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    return (cx, cy)

def generate_svg(facets, label_img, keep_labels, svg_size=(800,800)):
    """Generate an SVG string with polygon outlines for each facet and a centered label."""
    dwg = svgwrite.Drawing(size=svg_size)
    dwg.add(dwg.rect(insert=(0,0), size=svg_size, fill='white'))
    for label in keep_labels:
        contour = get_contour(label_img, label)
        if contour is None or len(contour) < 3:
            continue
        points = [(int(x), int(y)) for (x, y) in contour]
        dwg.add(dwg.polygon(points=points, fill='none', stroke='black', stroke_width=1))
        centroid = compute_centroid(points)
        dwg.add(dwg.text(str(label+1), insert=centroid, fill='black', font_size="10px", text_anchor="middle"))
    return dwg.tostring()

def generate_paint_by_numbers(
    input_image_path,
    output_path="output.svg",
    resize_width=1024,
    resize_height=1024,
    num_colors=10,
    min_facet_size=50
):
    """
    Complete pipeline:
      1. Read and resize image.
      2. K-means clustering.
      3. Facet building.
      4. Small facet pruning.
      5. Border detection and segmentation.
      6. Label placement.
      7. SVG generation.
    """
    # Load image using OpenCV (read in BGR)
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {input_image_path}")
    image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    
    # Step 1 & 2: K-means clustering
    quantized, centers = quantize_image(image, k=num_colors)
    
    # Step 3: Facet building
    label_img, facets = build_facets(quantized)
    
    # Step 4: Small facet pruning
    keep_labels = prune_facets(label_img, facets, min_facet_size)
    
    # Steps 5 & 6: Border detection is implicit in contour extraction and label placement in SVG generation
    svg_str = generate_svg(facets, label_img, keep_labels, svg_size=(resize_width, resize_height))
    
    # Step 7: Write SVG to output_path
    with open(output_path, "w") as f:
        f.write(svg_str)
    
    print(f"Paint-by-numbers SVG generated and saved to {output_path}")
