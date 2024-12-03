import sys
import os
import cv2
sys.path.insert(0, 'test_data')
from retinaface_model1 import FaceExtractionModel

# image = cv2.imread("test_image_2.jpg")
extraction = FaceExtractionModel("test_data", "test_output", 0, 1)
extraction.extract_faces()