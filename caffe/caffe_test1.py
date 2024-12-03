# import both classes
# call extract frame in a loop, store
# pass the extracted frame to face extraction model
# output faces into another dir
import sys
import os
from caffe_model1 import FaceExtractionModel

face_extraction = FaceExtractionModel(
        prototxt_path="deploy.prototxt", 
        caffe_model_path="res10_300x300_ssd_iter_140000.caffemodel", 
        input_directory="test_data", 
        output_directory="test_output"
    )

face_extraction.extract_faces(first_conf=0.128, second_conf=0.18, two_pass = True)