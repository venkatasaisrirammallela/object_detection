# object_detection

# YOLOv4-Tiny Object Detection with Streamlit

## Overview

This project is a Streamlit web application that utilizes the YOLOv4-Tiny model for real-time object detection. The application allows users to upload images, processes them using the YOLOv4-Tiny model, and displays the results with detected objects highlighted by bounding boxes. Users can also download the processed images with annotations.

## Features

- **Image Upload**: Users can upload images in JPG, JPEG, or PNG formats.
- **Object Detection**: Detects multiple objects within the image using the YOLOv4-Tiny model.
- **Bounding Boxes**: Draws bounding boxes around detected objects.
- **Labels**: Adds labels to the bounding boxes indicating the class of detected objects.
- **Download Image**: Provides an option to download the processed image with annotations.

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python libraries:
  - `streamlit`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  
You can install the required libraries using pip:

```bash
pip install streamlit opencv-python numpy Pillow
YOLOv4-Tiny Files
You need to download the following files for YOLOv4-Tiny:

YOLOv4-Tiny configuration file (yolov4-tiny.cfg)
YOLOv4-Tiny weights file (yolov4-tiny.weights)
COCO class labels file (coco.names)
Place these files in the project directory.

Usage
Save the Script: Save the provided Python script in a file named app.py.

Run the Streamlit App: Open a terminal or command prompt and navigate to the directory containing app.py. Run the following command:

bash
Copy code
streamlit run app.py
Access the Application: Open a web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).

Upload an Image: Use the file uploader to select an image for processing.

View Results: The processed image will be displayed with bounding boxes and labels. You can download the processed image using the download button.

Code Overview
Imports: Includes necessary libraries for Streamlit, OpenCV, NumPy, and PIL.
Model Loading: Loads YOLOv4-Tiny model and weights.
Image Processing: Handles image upload, preprocessing, object detection, and bounding box drawing.
UI Components: Implements Streamlit components for file upload, image display, and image download.
Example

Image processed with YOLOv4-Tiny.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
YOLOv4-Tiny model and COCO dataset
Streamlit for building interactive web applications
OpenCV for computer vision tasks
PIL for image processing
Contact
For any questions or issues, please open an issue on the GitHub repository.

vbnet
Copy code

### Notes:

- Replace `path/to/example_image.jpg` with the path to an example image in your repository or provide a relevant image URL.
- Update the `GitHub repository` link with the actual URL to your repository.
- Make sure the `LICENSE` file is present in your repository if you mention it in the README.

Feel free to adjust the README content based on any additional details specific to your project or preferences!






