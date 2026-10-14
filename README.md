# AI-Powered Multi-Modal Medical Image Analysis & Report Assistant

## Project Overview
An educational AI application for chest X-ray classification, visual explanations, and downloadable PDF reports.

## Features
- Upload chest X-ray images
- Classify images into BACTERIAL, NORMAL, or VIRAL categories
- Display model output scores
- Generate Grad-CAM heatmaps and overlays
- Generate educational analysis reports
- Download PDF reports with images
- View analysis history during the current session

## Technologies
- Python
- Streamlit
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- OpenCV
- Pillow
- Pandas
- ReportLab

## Model Details
- Architecture: EfficientNetB0
- Input size: 224 x 224
- Output classes: BACTERIAL, NORMAL, VIRAL
- Preliminary validation accuracy reported during development: approximately 81.11%

## How to Run
1. Install the dependencies using:
   pip install -r requirements.txt

2. Ensure the trained model and class_names.json file are available at the paths configured in app.py.

3. Start the application:
   streamlit run app.py

## Limitations and Responsible Use
This is an educational prototype, not a clinically validated medical device.
Predictions must not be used for diagnosis or treatment decisions.
Grad-CAM visualizations show regions influencing model output and are not medical findings.

## Author
Shravani Gudelli
