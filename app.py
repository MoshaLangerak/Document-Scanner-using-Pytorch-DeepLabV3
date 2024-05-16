# Install CPU version of torch and torchvision on streamlit cloud
import os
import cv2
import sys
import time
import subprocess
import numpy as np
import streamlit as st
import torch
from torchvision.datasets.utils import download_file_from_google_drive

# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")):
    print("Downloading Deeplabv3 with MobilenetV3-Large backbone...")
    download_file_from_google_drive(file_id=r"1ROtCvke02aFT6wnK-DTAIKP5-8ppXE2a", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C049.pth")


if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    print("Downloading Deeplabv3 with ResNet-50 backbone...")
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")
# ------------------------------------------------------------


from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from utility_functions import traditional_scan, deep_learning_scan, manual_scan, images_to_pdf_download_link

@st.cache_resource
def load_model_DL_MBV3(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model


@st.cache_resource
def load_model_DL_R50(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)

    model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model


def main(input_files, procedure, image_size=384):
    output_images = []
    for i, input_file in enumerate(input_files):
        
        file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)  # Read bytes
        image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]  # Decode and convert to RGB
        output_image = None

        st.title(f"Document {i + 1}")
        st.write("Input image size:", image.shape)

        if procedure == "Manual":
            output_image = manual_scan(og_image=image, key=i)

            # check if the output image is not None
            if output_image is not None:
                output_images.append(output_image)
   
        else:
            col1, col2 = st.columns((1, 1))

            with col1:
                st.header("Original")
                st.image(image, channels="RGB", use_column_width=True)

            with col2:
                st.header("Scanned")

                if procedure == "Traditional":
                    output_image = traditional_scan(og_image=image)
                else:
                    model = model_mbv3 if model_selected == "MobilenetV3-Large" else model_r50
                    output_image = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)

                st.image(output_image, channels="RGB", use_column_width=True)
            output_images.append(output_image)
            
    # convert the list of images to pdf
    if len(output_images) > 0:
        st.markdown(images_to_pdf_download_link(output_images), unsafe_allow_html=True)


IMAGE_SIZE = 384
model_mbv3 = load_model_DL_MBV3(img_size=IMAGE_SIZE)
model_r50 = load_model_DL_R50(img_size=IMAGE_SIZE)

st.title("Document Scanner")

procedure_selected = st.radio("Select Scanning Procedure:", ("Traditional", "Deep Learning", "Manual"), index=1, horizontal=True)

if procedure_selected == "Deep Learning":
    model_selected = st.radio("Select Document Segmentation Backbone Model:", ("MobilenetV3-Large", "ResNet-50"), horizontal=True)


file_upload = st.file_uploader("Upload Document Images :", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# check if the file is uploaded
if len(file_upload) > 0:
    _ = main(input_files=file_upload, procedure=procedure_selected, image_size=IMAGE_SIZE)