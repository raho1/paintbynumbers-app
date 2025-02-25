import streamlit as st
import os
import tempfile
from PIL import Image
from paintbynumbersgenerator import generate_paint_by_numbers

st.set_page_config(page_title="Paint-by-Numbers SVG Generator", page_icon="🎨", layout="wide")
st.title("🎨 Paint-by-Numbers SVG Generator")

st.write("""
Upload an image, select the number of colors and detail level, and generate a paint-by-numbers SVG template.
""")

# Sidebar parameters
resize_width = st.sidebar.number_input("Resize width (px)", min_value=256, max_value=4096, value=800)
resize_height = st.sidebar.number_input("Resize height (px)", min_value=256, max_value=4096, value=800)
num_colors = st.sidebar.slider("Number of Colors", min_value=2, max_value=64, value=10)

detail_level = st.sidebar.selectbox("Image Detail Level", options=["High", "Medium", "Low"], index=1)
def detail_to_min_size(detail):
    if detail == "High":
        return 20
    elif detail == "Medium":
        return 50
    elif detail == "Low":
        return 100
    return 50

min_facet_size = detail_to_min_size(detail_level)

uploaded_file = st.sidebar.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display original
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original image", use_container_width=True)

    if st.button("Generate Paint-by-Numbers SVG"):
        st.info("Processing image...")

        # --- WRITE UPLOADED FILE TO A STABLE TEMP PATH ---
        tempdir = tempfile.gettempdir()
        input_path = os.path.join(tempdir, "upload_image.png")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        # Output path for SVG
        output_path = os.path.join(tempdir, "paint_by_numbers.svg")

        # Run pipeline
        generate_paint_by_numbers(
            input_image_path=input_path,
            output_path=output_path,
            resize_width=resize_width,
            resize_height=resize_height,
            num_colors=num_colors,
            min_facet_size=min_facet_size
        )

        # Provide download
        with open(output_path, "rb") as f:
            svg_data = f.read()
        st.download_button("Download SVG", data=svg_data, file_name="paint_by_numbers.svg", mime="image/svg+xml")

        st.success("SVG generation complete!")
