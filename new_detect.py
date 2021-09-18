from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "-new2.png", required=True,
    help="path to input image")
ap.add_argument("-f", "--face", type=str,
    default="MaskDetector.model",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="MaskDetector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
