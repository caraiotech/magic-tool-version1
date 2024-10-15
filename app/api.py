"""Implement functions that send the input image to 
the API and receive the initial masks and contours
"""
import requests
#import torch
import pickle


API_BASE_URL = "http://3.7.234.80:8000"

"""
API_BASE_URL = "http://98.70.79.202:800"
"""


def get_initial_masks(image_numpy):
    """
    Get the initial masks of identified chromosomes from the API.
    Args:
        image_numpy (numpy.ndarray): Numpy array of the metaphase image.
    Returns:
        List of initial masks (PyTorch tensors) of identified chromosomes.
        List of bounding boxes of identified chromosomes.
    """
    # Send the image numpy array to the API
    response = requests.post(API_BASE_URL + "/segment", json={"images":image_numpy.tolist(), "confidence": 0.2})
    
    if response.status_code == 200:
        # Unpickle the response content
        response_data = pickle.loads(response.content)
        
        # Get the masks from the unpickled data
        masks_tensors, bboxes = response_data[0], response_data[1]  # Assuming masks are at index 0
        print("Initial masks received from API.")
        return masks_tensors, bboxes
    else:
        print("Error: Unable to get initial masks from API.")
        return []

def get_masks_with_bbox(image_numpy, bboxes):
    """
    Get the masks using the bounding boxes of identified chromosomes from the API.
    Args:
        image_numpy (numpy.ndarray): Numpy array of the metaphase image.
    Returns:
        List of initial masks (PyTorch tensors) of identified chromosomes.
        List of bounding boxes of identified chromosomes.
    """
    # Send the image numpy array to the API
    response = requests.post(API_BASE_URL + "/segment-with-bbox", json={"images":image_numpy.tolist(), "bbox": bboxes})
    
    if response.status_code == 200:
        # Unpickle the response content
        response_data = pickle.loads(response.content)
        
        # Get the masks from the unpickled data
        masks_tensors, bboxes = response_data[0], response_data[1]  # Assuming masks are at index 0
        print("Mask with bbox received from API.")
        return masks_tensors, bboxes
    else:
        print("Error: Unable to get initial masks from API.")
        return []
