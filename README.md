# Image Processing Functions

This Streamlit application provides various image processing functions that allow you to alter and visualize images. The application uses Python, OpenCV, and other libraries to perform image processing operations.

## Getting Started

To run this application locally, follow these steps:

1. Make sure you have Python installed on your machine.

2. Install the required Python packages using the following command:

   ```bash
   pip install streamlit opencv-python numpy natsort scipy skimage rembg streamlit_image_comparison    
3. Download the code from this repository or copy the code provided below.

4. Open a terminal window and navigate to the directory where the code is located.

5. Run the Streamlit app using the following command: 

   ```bash
   streamlit run img_processing_webgui.py
6. The Streamlit app will open in a web browser, allowing you to upload an image and adjust various image processing settings.

## Features

This Streamlit application offers the following features for image processing:

- **Load and Visualize Images:** Upload an image using the file uploader and visualize it in the Streamlit app.

- **Adjust Cropping:** Customize cropping settings for different portions of the image, including the top, bottom, left, and right edges.

- **Resize Images:** Use the provided slider to resize the image according to your preferences.

- **Apply Various Image Processing Operations:** Choose from a range of image processing operations to alter the appearance of the image:

  - **Automatic White Balance (AWB):** Perform automatic white balance correction based on the "Gray World" assumption.

  - **Custom White Balance (AWB_custom):** Apply custom white balance by adjusting factors for each color channel.

  - **Blur:** Apply Gaussian blurring to the image, with adjustable sigma values.

  - **Grayscale:** Convert the image to grayscale and then back to color.

  - **Denoise:** Apply bilateral filtering for denoising, with adjustable parameters.

  - **Contrast:** Adjust the contrast of the image using a slider.

  - **Brightness:** Modify the brightness of the image using a slider.

  - **Gamma Correction:** Apply gamma correction to adjust image contrast and brightness.

  - **Saturation:** Alter the saturation of the image using a slider.

  - **Sharpness:** Enhance image sharpness using convolutional filters and Gaussian blurring.

  - **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Apply adaptive histogram equalization for enhanced contrast.

  - **Background Removal:** Use the "rembg" library to remove the background from the image.

  - **Image Tiling:** Divide the image into smaller tiles, optionally with a grid drawn around each tile.

  - **Sobel Edge Detection (Vertical and Horizontal):** Apply Sobel edge detection in both vertical and horizontal directions.

- **Toggle Before/After Images:** Utilize the "Before/After slider tool" to toggle between the original and processed images.

- **Visualize Processed Images:** Display the processed image in the Streamlit app, either as a standalone image or in comparison mode.

## Usage

1. **Upload Image:** Use the file uploader to load an image.

2. **Crop:** Adjust cropping to keep desired parts of the image.

3. **Resize:** Change image size using the slider.

4. **Process:** Select operations and adjust settings as needed.

5. **Apply:** Click "Initialize" to apply settings.

6. **View:** Compare processed image to original using "Before/After" tool.

7. **Experiment:** Try different settings for desired effects.