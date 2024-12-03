import os
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt
import cv2
import numpy as np

class FaceExtractionModel:
    """
    A class for extracting faces from images using a pre-trained Caffe model.

    Attributes:
        opencv_dnn_model (cv2.dnn_Net): The OpenCV DNN model loaded from the provided prototxt and Caffe model files.

    Methods:
        __init__(prototxt_path, caffe_model_path, input_directory, output_directory):
            Initializes the FaceExtractionModel with the given paths to the prototxt and Caffe model files, 
            as well as the input and output directories.
        cv_dnn_detect_faces(image, min_confidence, display=True):
            Detects faces in the given image using the OpenCV DNN model with the specified minimum confidence level.
        two_pass_face_detection(image, first_conf, second_conf, im):
            Detects faces in the given image using two passes with different confidence levels.
        extract_faces():
            Runs the face extraction process on all images in the input directory.
    """
    
    def __init__(self, prototxt_path, caffe_model_path, input_directory, output_directory):
        """
        Initializes the FaceExtractionModel with the given paths to the prototxt and Caffe model files, 
        as well as the input and output directories.
            
        Args:
            prototxt_path (str): The path to the prototxt file for the Caffe model.
            caffe_model_path (str): The path to the Caffe model file.
            input_directory (str): The path to the directory containing input images.
            output_directory (str): The path to the directory where extracted faces will be saved.
            
        Raises:
            FileNotFoundError: If the prototxt or Caffe model file is not found.
        """
    
        # Raise errors if prototxt or Caffe model file not found
        if not os.path.isfile(prototxt_path):
            raise FileNotFoundError(f"Prototxt file not found at {prototxt_path}")
        if not os.path.isfile(caffe_model_path):
            raise FileNotFoundError(f"Caffe model file not found at {caffe_model_path}")
        
        # Load Caffe model
        self.opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt=prototxt_path, caffeModel=caffe_model_path)

        # Initialize input and output directories
        self.input_directory = input_directory
        self.output_directory = output_directory
    
    def cv_dnn_detect_faces(self, image, min_confidence, display = True):
        """
        Detects faces in the given image using the OpenCV DNN model with the specified minimum confidence level,
        outputs image with bounding boxes, model results, a lits of extracted face images, and count of faces detected.

        Args:
            image (numpy.ndarray): The input image in BGR format.
            min_confidence (float): The minimum confidence level for face detection.
            display (bool): Whether to display the input image with detected faces.
        
        Returns:
            output_image (numpy.ndarray): The input image with boxes drawn around detected faces.
            results (numpy.ndarray): The results of the face detection model.
            faces (list): A list of extracted face images.
            faces_count (int): The number of faces detected in the
        """
        
        # Determine height and width of input image
        image_height, image_width, _ = image.shape

        # Copy input image to draw rectangles on
        output_image = image.copy()

        # Preprocess input image
        preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                                mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

        # Run model
        self.opencv_dnn_model.setInput(preprocessed_image)

        results = self.opencv_dnn_model.forward()    

        # Iterate through model results, counting faces above confidence threshold and filling lists of extracted faces

        faces_count = 0
        faces = []
        bboxes = []

        for face in results[0][0]:
            
            face_confidence = face[2]
            
            if face_confidence > min_confidence:
                faces_count += 1

                bbox = face[3:]
                bboxes.append(bbox)
                x1 = int(bbox[0] * image_width)
                y1 = int(bbox[1] * image_height)
                x2 = int(bbox[2] * image_width)
                y2 = int(bbox[3] * image_height)

                cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//400)

                cropped_img = output_image[y1:y2, x1:x2]
                if not (cropped_img is None or cropped_img.size == 0):
                    faces.append(cropped_img)

        # If display flag on, show image with rectangles
        if display:

            plt.figure(figsize=[20,20])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off')
            plt.show()

            # Return input image with boxes, model results, and count of faces
            return output_image, results, bboxes, faces, faces_count

        # Else return input image with boxes, model results, and count of faces
        else:

            return output_image, results, bboxes, faces, faces_count
        

    def two_pass_face_detection(self, image, first_conf, second_conf):

        image_height, image_width, _ = image.shape
        output_image_2 = image.copy()

        # Run first pass of model on image
        output_image, results, bboxes, faces, faces_count = self.cv_dnn_detect_faces(image, first_conf, display=False)

        # Determine number of faces after initial pass
        initial_faces_count = len(faces)

        # Check if no faces found on initial pass
        if initial_faces_count == 0:

            return image, 0

        else:
            final_faces = []
            face_index = 0

            for face in faces:
                
                if self.cv_dnn_detect_faces(face, second_conf, display=False)[4] > 0:
                    
                    # Add face extraction to final_faces
                    final_faces.append(face)
                    bbox = bboxes[face_index]             
                    x1 = int(bbox[0] * image_width)
                    y1 = int(bbox[1] * image_height)
                    x2 = int(bbox[2] * image_width)
                    y2 = int(bbox[3] * image_height)
                    cv2.rectangle(output_image_2, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//500)
                    face_index += 1

            # Determine final number of faces
            final_faces_count = len(final_faces)

            return output_image_2, final_faces_count
        
    def extract_faces(self, first_conf, second_conf = None, two_pass = False):
        
        # Initialize input directory
        image_directory = self.input_directory

        # Define image count variable
        image_count = 0

        output_directory = None
        if two_pass:
            output_directory = self.output_directory + f"-{first_conf}-{second_conf}"
        else:
            output_directory = self.output_directory + f"-{first_conf}"

        os.makedirs(output_directory, exist_ok=True)
        textfile_path = os.path.join(output_directory, "caffe_output.txt")
        f=open(textfile_path, "w")

        for filename in os.listdir(self.input_directory):

            # If file is image:
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
    
                file_path = os.path.join(self.input_directory, filename)
                
                image = cv2.imread(file_path)
                
                if image is None:

                    continue

                image_count +=1

                # use image as input
                output_image = None
                if two_pass:
                    output_image, faces_count = self.two_pass_face_detection(image, first_conf, second_conf)

                else:
                    output_image, results, bboxes, faces, faces_count = self.cv_dnn_detect_faces(image, first_conf, display = False)

                output_path = os.path.join(output_directory, filename)
                cv2.imwrite(output_path, output_image)
                f.write(f"{faces_count} faces detected in {filename}\n")               
        return
        
                