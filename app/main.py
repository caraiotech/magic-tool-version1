import tkinter as tk
import requests
import numpy as np
from PIL import Image
import pickle
from gui import display_chromosomes_grid
from utils import masks_to_contours
from api import get_initial_masks
from config import image_path


# API base URL
API_BASE_URL = "http://3.7.234.80:8000"

if __name__ == "__main__":
    root = tk.Tk()
    # Load the image
    image_numpy = Image.open(image_path)  # Load your metaphase image here
    # If image size is greater than 512x512, resize it
    if image_numpy.size[0] > 512:
        image_numpy = image_numpy.resize((512, 512))
    image_numpy = np.array(image_numpy)
    masks, bboxes = get_initial_masks(image_numpy)
    contours = masks_to_contours(masks)
    with open('stored_contours.pkl', 'wb') as f:
        pickle.dump(contours, f)
    with open('stored_bboxes.pkl', 'wb') as f:
        pickle.dump(bboxes, f)
    display_chromosomes_grid(root)
    root.mainloop()
