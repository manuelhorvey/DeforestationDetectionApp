# Deforestation Detection App

This application uses a transformer-based ChangeFormer model to detect deforestation in the Brazilian Amazon using Sentinel-2 satellite imagery. Developed as a final year project, it processes 4-band (RGB + NIR) .tif images from 2020 and 2021 to generate binary change masks and overlay predictions, achieving an F1-score of 0.9886 and IoU of 0.9572 on validation data.

## Overview
- **Model**: Custom ChangeFormer with a VisionTransformer encoder, FeatureDifferenceModule, and DeconvDecoder.
- **Data**: Sentinel-2 Level-2A imagery (10m resolution) and PRODES ground-truth data.
- **Interface**: Built with Gradio for interactive uploads and visualizations.
- **Purpose**: Supports land governance, policy-making, and ecological conservation through scalable deforestation monitoring.

## Features
- Upload two .tif images (2020 and 2021) with 4 bands (B2, B3, B4, B8).
- Outputs: Raw 2021 RGB, overlay with predicted deforestation, binary change mask, and a comment on change percentage.
- Patch-wise processing for large images, with percentile-based normalization and stitching.

## Setup
1. **Prerequisites**:
   - Python 3.8+
   - Required libraries: `torch`, `torchvision`, `timm`, `rasterio`, `numpy`, `pillow`, `gradio`.

2. **Installation**:
   - Clone or download this repository.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Place your pretrained model (`best_model.pth`) in the `models/` folder.

3. **Run Locally**:
   - Launch the app:
     ```bash
     python app.py
     ```
   - Access the interface at `http://localhost:7860`.

4. **Deployed Version**:
   - Check the live app at ***** .

## Usage
- **Input**: Upload two .tif files (e.g., 256x256 patches) containing RGB and NIR bands.
- **Output**:
  - **Raw 2021 RGB**: Normalized base image.
  - **Overlay with Prediction**: Red overlay highlighting deforested areas.
  - **Binary Change Mask**: Black-and-white change map.
  - **Comment**: Auto-generated note on change extent (e.g., "Significant change detected: 5.83%").
- **Notes**: Ensure images are preprocessed (e.g., <20% cloud cover) for best results.

## Project Details
- **Region of Interest**: Top 5 deforested conservation units (e.g., Área de Proteção Ambiental Triunfo do Xingu).
- **Dataset**: 19,560 bitemporal patches (2020–2021), augmented with rotations.
- **Performance**: Validation F1-score: 0.9986, IoU: 0.9972.
- **Future Work**: Multi-year forecasting, web-based alerts, SAR integration.

## Credits
- **Author**: Emmanuel Amey, Sammuel Young Appiah, Asare Prince Owusu, Yaaya Pearl Apenu.
- **References**: Inspired by Alshehri et al. (2024), IEEE Geoscience and Remote Sensing Letters.
