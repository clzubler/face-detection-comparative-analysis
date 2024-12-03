import matplotlib.pyplot as plt
import cv2
import os

class FaceExtractionModel:
    """
    A class for extracting faces from images using Haar cascades.

    Attributes:
        input_directory (str): The path to the directory containing input images.
        scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
        min_neighbors (float): Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        min_size (float): Minimum possible object size.

    Methods:
        __init__(self, input_directory, scale_factor, min_neighbors, min_size):
            Initializes the FaceExtractionModel with the given input directory path,
            scale factor, minimum neighbors, and minimum size.
        detect_faces(self, image, display=True):
            Detects faces in the given image, returns the number of faces detected and the image with bounding boxes, 
            optionally displays the image with detected faces.
        extract_faces(self):
            Detects faces in each image in the input directory, saves images with bounding boxes to output directory,
            prints the number of faces detected in each image to log file.
    """

    def __init__(self, input_directory, scale_factor, min_neighbors, min_size):
        """
        Initializes the FaceExtractionModel with the given input and output directories, scale factor, minimum neighbors, and minimum size.

        Args:
            input_directory (str): The path to the directory containing input images.
            scale_factor (float): Parameter specifying how much the image size is reduced at each image scale.
            min_neighbors (float): Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            min_size (float): Minimum possible object size.
        """
        self.input_directory = input_directory
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, image, display=True):
        """
        Detects faces in the given image, returns the number of faces detected and the image with bounding boxes, 
        optionally displays the image with detected faces.

        Args:
            image (numpy.ndarray): The input image.
            display (bool): Whether to display the input image with detected faces.

        Returns:
            int: The number of faces detected in the image.
            image: The input image with bounding boxes.
        """
        gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image1)

        profile_face_classifier = cv2.CascadeClassifier("haarcascade_profileface.xml")
        frontal_face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        frontal_faces = frontal_face_classifier.detectMultiScale(
            gray_image, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size
        )
        
        profile_faces = profile_face_classifier.detectMultiScale(
            gray_image, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size
        )

        flipped_img = cv2.flip(gray_image, 1)
        flipped_profile_faces = profile_face_classifier.detectMultiScale(
            flipped_img, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size
        )

        for (x, y, w, h) in flipped_profile_faces:
            profile_faces = list(profile_faces)
            profile_faces.append((gray_image.shape[1] - x - w, y, w, h))

        faces = list(frontal_faces) + list(profile_faces)

        count = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
            count += 1

        if display:
            cv2.imshow("Detected faces", image)
            cv2.waitKey(0)

        return count, image

    def extract_faces(self):
        """
        Detects faces in each image in the input directory, saves images with bounding boxes to output directory,
        prints the number of faces detected in each image to log file.
        """
        output_directory = f"output-{self.scale_factor}-{self.min_neighbors}-{self.min_size}"
        os.makedirs(output_directory, exist_ok=True)
        textfile_path = os.path.join(output_directory, "haar_output.txt")
        f=open(textfile_path, "w")
        for filename in os.listdir(self.input_directory):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                file_path = os.path.join(self.input_directory, filename)
                image = cv2.imread(file_path)

                if image is None:
                    continue

                faces_count, output_image = self.detect_faces(image, display=False)

                # Save detections to output directory
                output_filename = f"{filename}"
                output_path = os.path.join(output_directory, output_filename)
                cv2.imwrite(output_path, output_image)
                f.write(f"{faces_count} faces detected in {filename}\n")
        