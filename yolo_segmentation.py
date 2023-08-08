# Import necessary libraries
from ultralytics import YOLO
import numpy as np

# Define a class for YOLO segmentation
class YOLOSegmentation:
    # Initialize the class with the model path
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    # Define a function to detect objects and segmentations in an image
    def detect(self, img):
        # Get the height, width, and channels of the image
        height, width, channels = img.shape
        
        # Use the YOLO model to predict objects and segmentations in the image
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        
        # Check if there are any segmentation masks in the results
        if result.masks:
            # Create an empty list to store the segmentation contours
            segmentation_contours_idx=[]
            
            # Loop through each segmentation mask in the results
            for seg in result.masks.xyn:
                # Scale the segmentation coordinates to match the image size
                seg[:,0]*=width
                seg[:,1]*=height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

            # Convert the bounding boxes, class IDs, and scores to numpy arrays
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
            
            # Return the bounding boxes, class IDs, segmentation contours, and scores
            return bboxes, class_ids, segmentation_contours_idx, scores
        
        # If there are no segmentation masks in the results, return None for all outputs
        return None, None, None, None
