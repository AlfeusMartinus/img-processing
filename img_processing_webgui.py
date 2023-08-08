import os

import cv2
import numpy as np
import streamlit as st
from natsort import natsorted
from skimage import filters

st.set_page_config(page_title="Image-Comparison Example", layout="centered")

st.title("Image Processing Functions")

# Let the user select the image they want to use
image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg", "tif", "tiff"])
if image is None: st.info("Please upload an image file."); st.stop()
if image.size > 5000000: st.info("Please upload an image file smaller than 5MB."); st.stop()

image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
# Delete any existing image.png file and altered.png file
try: os.remove("image.png")
except: pass
try: os.remove("altered.png")
except: pass

with st.form(key='my_form', clear_on_submit=False):
    st.subheader("Settings")
    # Add a toggle button for including the before/after image
    before_after = st.checkbox("Before/After slider tool", value=True)
    col1, col2 = st.columns(2)
    top_crop = col1.number_input("Top Crop %", 0, 100, 0)
    bottom_crop = col2.number_input("Bottom Crop %", 0, 100, 0)
    col1, col2 = st.columns(2)
    left_crop = col1.number_input("Left Crop", 0, 100, 0)
    right_crop = col2.number_input("Right Crop", 0, 100, 0)

    # Resize the image to a user selected size
    resize = st.slider('Resize', 0, 100, 100)

    # Give the user some options for altering the image with multiselect
    img_processing_operations = natsorted(["AWB", "Blur", "Grayscale", 
                                "Denoise", "Contrast", "Brightness",
                                "Saturation", "Sharpness", "CLAHE", 
                                "AWB_custom", "Gamma", "Remove-Background",
                                "Tile-Plakakia","Sobel_Vertical","Sobel_Horizontal"])
    options = st.multiselect("Alter the image...", img_processing_operations)

    if "AWB_custom" in options:
        # Perform custom white balance i.e. the user can select the factor for each channel
        col1, col2 = st.columns(2)
        with col1: awb_b = st.slider('AWB B', 0.0, 10.0, 3.0)
        with col2: awb_g = st.slider('AWB G', 0.0, 10.0, 3.0)
    if "Blur" in options:
        # Blur the image using a Gaussian filter and a slider for the sigma value
        sigmablur = st.slider('Blur Sigma', 0.0, 10.0, 3.0)
    if "Denoise" in options:
        col1, col2, col3 = st.columns(3)
        with col1: diameter = st.slider('Denoise Diameter', 0, 10, 3)
        with col2: sigmaColor = st.slider('Denoise Sigma Color', 0, 10, 3)
        with col3: sigmaSpace = st.slider('Denoise Sigma Space', 0, 10, 3)
    if "Sharpness" in options:
        sigmaSharpness = st.slider('Sharpness Sigma', 0.0, 10.0, 3.0)
    if "Brightness" in options:
        brightness = st.slider('Brightness', 0.0, 10.0, 3.0)
    if "Gamma" in options:
        gamma = st.slider('Gamma', 0.0, 10.0, 3.0)
    if "Saturation" in options:
        saturation = st.slider('Saturation', 0.0, 10.0, 3.0)
    if "CLAHE" in options:
        col1, col2 = st.columns(2)
        with col1:
            clipLimit = st.slider('CLAHE Clip Limit', 0.0, 10.0, 3.0)
        with col2:
            tileGridSize = st.slider('CLAHE Tile Grid Size', 0, 10, 3)
    if "Remove-Background" in options:
        # Let the user select alpha matting foreground and background threshold values
        col1, col2, col3 = st.columns(3)
        with col1: alpha_matting = st.checkbox("Alpha Matting")
        with col2: bg_threshold = st.slider('Background Threshold', 0, 255, 10)
        with col3: fg_threshold = st.slider('Foreground Threshold', 0, 255, 240)
    if "Tile-Plakakia" in options:
        # Let the user select the tile_size and the step_size
        col1, col2 = st.columns(2)
        with col1: tile_size = st.slider('Tile Size', 0, 100, 50)
        with col2: step_size = st.slider('Step Size', 0, 100, 50)

    # Add a submit button
    submit_button = st.form_submit_button(label='Initialize / Update Image')

# Explain each of the image processing functions
with st.expander("ℹ️ Glossary 📖"):
    st.markdown("""
    - **AWB** - Automatic White Balance (https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption)
    - **AWB_custom** - Custom White Balance
    - **Blur** - Blur the image using a Gaussian filter (https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
    - **Brightness** - Increase or decrease the image brightness (https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    - **CLAHE** - Contrast Limited Adaptive Histogram Equalization (https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
    - **Contrast** - Increase or decrease the image contrast (https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    - **Denoise** - Denoise the image using a bilateral filter (https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise.html)
    - **Gamma** - Gamma Correction (https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    - **Grayscale** - Convert the image to grayscale (https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html)
    - **Saturation** - Increase or decrease the image saturation (https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    - **Sharpness** - Increase or decrease the image sharpness (https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html)
    - **Remove-Background** - Remove the background from the image (https://pypi.org/project/rembg/)
    - **Sobel_Vertical** - Sobel Vertical Edge Detection (https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)
    - **Sobel_Horizontal** - Sobel Horizontal Edge Detection (https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)
    - **Tile-Plakakia** - Tile the image into smaller images (https://www.github.com/kalfasyan/plakakia)
    """)


if submit_button:
    # Crop the image
    image = image[:, int(image.shape[1] * (left_crop / 100.0)):int(image.shape[1] * (1 - (right_crop / 100.0))), :]
    # Crop the image
    image = image[int(image.shape[0] * (top_crop / 100.0)):int(image.shape[0] * (1 - (bottom_crop / 100.0))), :, :]
    image = cv2.resize(image, (0, 0), fx=(resize / 100.0), fy=(resize / 100.0))
    cv2.imwrite("image.png", image)

    if "AWB" in options:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(image[:, :, 1])
        avg_b = np.average(image[:, :, 2])
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                l, a, b = image[x, y, :]
                # fix for CV correction
                l *= 100 / 255.0
                image[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
                image[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        # Explanation of the above code:
        # We are using the "Gray World" assumption method to perform white balancing
        # The idea is that in an image the average value for each channel is the same
        # So we calculate the average channel value and then adjust each channel
        # based on the average value and the difference from the theoretical average value of 128
        # We multiply the average value by 100/255 to normalize the value to 0-100
        # We then multiply the average value by 1.1 to increase the color balance
        # Finally we convert the image back to BGR from LAB
        # The above method works well for most images but can sometimes overcorrect
        # For this reason we have also included a custom white balance option
        # The custom white balance option allows the user to select the factor for each channel
        # The custom white balance option is also useful for images that are already well balanced
        # Resouces:
        # https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
        # https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
        # https://stackoverflow.com/questions/36982736/opencv-white-balancing-with-grayworld-assumption
        # https://stackoverflow.com/questions/3490727/what-are-some-simple-methods-for-balancing-the-colors-of-an-image
    if "AWB_custom" in options:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(image[:, :, 1])
        avg_b = np.average(image[:, :, 2])
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                l, a, b = image[x, y, :]
                # fix for CV correction
                l *= 100 / 255.0
                image[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * awb_b)
                image[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * awb_g)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        
    if "Blur" in options:
        image = cv2.GaussianBlur(image, (0, 0), sigmablur)
    if "Grayscale" in options:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if "Denoise" in options:
        image = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    if "Contrast" in options:
        # Change contrast using a slider
        contrast = st.slider('Contrast', 0.0, 10.0, 3.0)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    if "Brightness" in options:
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    if "Gamma" in options:
        # Perform gamma correction with a slider
        image = np.power(image / 255.0, gamma)
        image = cv2.convertScaleAbs(image, alpha=255.0)
    if "Saturation" in options:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=saturation, beta=0)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    if "Sharpness" in options:
        # sharpen the colors image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.GaussianBlur(image, (0, 0), sigmaSharpness)
    if "CLAHE" in options:
        # convert from BGR to LAB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # split on 3 different channels
        l, a, b = cv2.split(image)
        # apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
        l = clahe.apply(l)
        # merge channels
        image = cv2.merge((l, a, b))
        # convert from LAB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    if "Remove-Background" in options:
        # Use the rembg library to remove the background
        import rembg

        # use a slider to adjust the alpha matting
        image = rembg.remove(image, alpha_matting=alpha_matting, alpha_matting_foreground_threshold=fg_threshold, alpha_matting_background_threshold=bg_threshold)
    if "Tile-Plakakia" in options:
        from plakakia.utils_tiling import tile_image
        tiles, coordinates = tile_image(image, tile_size, step_size)

        # Draw a rectangle around each tile and make it smaller by 1 pixel to see the grid
        tile_size = int(tile_size)
        step_size = int(step_size)
        for i in range(len(coordinates)):
            x1, y1, x2, y2 = coordinates[i]
            # Make the color the inverse of the average color of the tile
            inv_color = (255 - int(np.average(tiles[i, :, :, 0])), 255 - int(np.average(tiles[i, :, :, 1])), 255 - int(np.average(tiles[i, :, :, 2])))
            # Make the rectangle 1 pixel smaller to see the grid
            cv2.rectangle(image, (x1, y1), (x2-1, y2-1), inv_color, 1)
        st.write("Image size: ", image.shape)
    if "Sobel_Vertical" in options:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = filters.sobel_v(image)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
    if "Sobel_Horizontal" in options:
        if len(image.shape) == 3:        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = filters.sobel_h(image)
        image = image / np.max(image)
        image = (image * 255).astype(np.uint8)


    if image is not None:
        cv2.imwrite("altered.png", image) 
    else:
        st.error("Please initialize")    

    if image is not None:
        if before_after:
            from streamlit_image_comparison import image_comparison
            st.header("Before and after")
            image_comparison(img1="image.png", img2="altered.png") 
        else:
            st.header("Processed image")
            st.image(image, caption=f"", use_column_width=True, channels="BGR")

    else: 
        st.stop()
