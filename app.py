import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Load YOLOv4-tiny model and weights
net = cv2.dnn.readNet(r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.cfg', r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.weights')

# Load COCO class labels
with open(r'C:\Users\venka\OneDrive\Desktop\object detection\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit App Titleimport streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Load YOLOv4-tiny model and weights
net = cv2.dnn.readNet(r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.cfg', r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.weights')

# Load COCO class labels
with open(r'C:\Users\venka\OneDrive\Desktop\object detection\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit App Title
st.title("Optimized Object Detection with YOLOv4-Tiny")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Detect objects
    outs = net.forward(output_layers)
    
    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Lowered confidence threshold
    conf_threshold = 0.1
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []

    # Draw bounding boxes and labels on the image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indexes:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}"
        color = colors[class_ids[i]]
        
        # Draw semi-transparent rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Add small text label
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Convert image back to RGB (from BGR) for display with PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    
    # Display the image
    st.image(img_pil, caption='Processed Image.', use_column_width=True)
    
    # Save image to a buffer for downloading
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    # Provide download link
    st.download_button(
        label="Download image",
        data=byte_im,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )

# Load YOLOv4-tiny model and weights
net = cv2.dnn.readNet(r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.cfg', r'C:\Users\venka\OneDrive\Desktop\object detection\yolov4-tiny.weights')

# Load COCO class labels
with open(r'C:\Users\venka\OneDrive\Desktop\object detection\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit App Title
st.title("Optimized Object Detection with YOLOv4-Tiny")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Detect objects
    outs = net.forward(output_layers)
    
    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Lowered confidence threshold
    conf_threshold = 0.1
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []

    # Draw bounding boxes and labels on the image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in indexes:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}"
        color = colors[class_ids[i]]
        
        # Draw semi-transparent rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Add small text label
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Convert image back to RGB (from BGR) for display with PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    
    # Display the image
    st.image(img_pil, caption='Processed Image.', use_column_width=True)
    
    # Save image to a buffer for downloading
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    # Provide download link
    st.download_button(
        label="Download image",
        data=byte_im,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )

st.title("Optimized Object Detection with YOLOv4-Tiny")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Detect objects
    outs = net.forward(output_layers)
    
    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Lowered confidence threshold
    conf_threshold = 0.1
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Draw bounding boxes and labels on the image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}"  # Removed confidence score
            color = colors[class_ids[i]]
            
            # Draw semi-transparent rectangle
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Add small text label
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Convert image back to RGB (from BGR) for display with PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    
    # Display the image
    st.image(img_pil, caption='Processed Image.', use_column_width=True)
    
    # Save image to a buffer for downloading
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    
    # Provide download link
    st.download_button(
        label="Download image",
        data=byte_im,
        file_name="processed_image.jpg",
        mime="image/jpeg"
    )