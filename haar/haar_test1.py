import sys
import os
import cv2
sys.path.insert(0, 'test_data')
from haar_model1 import FaceExtractionModel

extraction = FaceExtractionModel("test_data", 1.05, 21, (10,10))
extraction.extract_faces()

