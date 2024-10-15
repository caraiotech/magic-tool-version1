"""
define utility functions that might be used across the application. This could include functions to convert masks to contours, display images, handle grid layouts, etc.
"""
import pickle
import cv2
import numpy as np
import requests
# import torch
from skimage.morphology import medial_axis

def masks_to_contours(masks_tensors):
    """
    Convert masks (PyTorch tensors) to contours.
    Args:
        masks_tensors (list): List of PyTorch tensors representing masks.
    Returns:
        List of contours representing chromosomes.
    """
    contours_list = []
    for mask_tensor in masks_tensors[0]:
        kernel = np.ones((5,5), np.uint8)
        mask = mask_tensor.cpu().permute(1,2,0).numpy()
        mask = mask[:,:,0].astype(np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        contours_list.append(largest_contour)
    return contours_list

def extract_chromosomes_with_contours(contours, img):
    # Extract chromosomes from the original image using the contours
    extracted_chromosomes = []
    extracted_chromosomes_coords = []

    for contour in contours:
        # Get the bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Save the coordinates of the center of the chromosome
        extracted_chromosomes_coords.append((x + w / 2, y + h / 2))

        # Create a mask for the current contour
        mask = np.zeros(np.array(img).shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Extract the chromosome using the mask, the original image, and the bounding rectangle 
        # Make sure the background is white
        mask_inverse = cv2.bitwise_not(mask)
        # Extract the chromosome using the mask and the bounding rectangle
        chromosome = cv2.bitwise_and(np.array(img), np.array(img), mask=mask)
        # Use the mask inverse on the chromosome to make the background white
        chromosome[mask_inverse == 255] = 255
        # Extract the chromsome proper using the bounding rectangle
        chromosome = chromosome[y:y + h, x:x + w]
        # Rotate the chromosomes
        chromosome = rotate_image(chromosome)
        # Straighten the chromosomes
        chromosome = straighten_chromosome(chromosome)
        # Pad the chromosomes by 5 pixels on each side with (255, 255, 255)
        # chromosome = np.pad(chromosome, ((5, 5), (5, 5), (0, 0)), mode='constant', constant_values=255)

        extracted_chromosomes.append(chromosome)
    return extracted_chromosomes, extracted_chromosomes_coords

def get_region(image, coords):
    # Get the coordinates
    x, y = coords
    # Calculate the region boundaries
    y_start = max(0, y - 100)
    y_end = min(image.shape[0], y + 100)
    x_start = max(0, x - 100)
    x_end = min(image.shape[1], x + 100)
    # Convert the coordinates to integers
    y_start, y_end, x_start, x_end = int(y_start), int(y_end), int(x_start), int(x_end)
    # Get the region
    region = image[y_start:y_end, x_start:x_end]
    
    # Pad the region with zeros if necessary
    if region.shape[0] < 200:
        pad_height = 200 - region.shape[0]
        region = np.pad(region, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
    if region.shape[1] < 200:
        pad_width = 200 - region.shape[1]
        region = np.pad(region, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    
    return region

def rotate_image(image):
    """Rotates the image if the width is greater than the height.

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get the image width and height
    height, weight, _ = image.shape
    # If the width is greater than the height, then rotate the image by 90 degrees
    if weight > height:
        image = np.rot90(image)
    # Return the image
    return image


def straighten_chromosome(image):
    """Straightens a chromosome image by rotating it to align with the horizontal axis.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The straightened image as a NumPy array.
    """
    # Converting the image to grayscale, thresholding, and finding contours
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold_image = cv2.threshold(
        grayscale_image, 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.ones(
        (image.shape[0], image.shape[1], 3), np.uint8) * 255

    for cont in contours:
        # Finding the minimum rectangle around the chromosome
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[2]
        coordinates = box.tolist()

        # Finding the slope between the two bottom points
        sorted_points = sorted(coordinates, key=lambda p: p[1])
        x_1, y_1 = sorted_points[0]
        x_2, y_2 = sorted_points[1]
        slope = 0
        if (x_2 - x_1) != 0:
            slope = (y_2 - y_1) / (x_2 - x_1)

        # Using the slope to determine the tilt direction (tilted left/right)
        if slope > 0:
            rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        elif slope < 0:
            rotation_matrix = cv2.getRotationMatrix2D(
                rect[0], -(90 - angle), 1.0)
        else:
            rotation_matrix = cv2.getRotationMatrix2D(rect[0], 0, 1.0)

        # Rotating the image
        rotated_image = cv2.warpAffine(
            image, rotation_matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
        blank_image = np.ones(
            (image.shape[0], image.shape[1], 3), np.uint8) * 255
        blank_image[:rotated_image.shape[0],
                    :rotated_image.shape[1]] = rotated_image
        _ = cv2.transform(np.array([box]), rotation_matrix)

    return blank_image

def contours_to_bboxes(contours):
    """
    Convert contours to bounding boxes.
    Args:
        contours (list): List of contours.
    Returns:
        List of bounding boxes.
    """
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return bboxes

def get_karyotype(preds):
    """_summary_

    Args:
        preds (_type_): _description_

    Returns:
        _type_: _description_
    """
    rows = 4
    cols = 7
    # A list representation of Karyotype template
    karyotype_template = [
        [1, 2, 3, 0, 0, 4, 5],
        [6, 7, 8, 9, 10, 11, 12],
        [13, 14, 15, 0, 16, 17, 18],
        [19, 20, 0, 21, 22, 23, 24]
    ]
    # Create a numpy array which holds 200,200,3 images in each cell
    karyotype = np.zeros((rows*200, cols*200, 3), np.uint8)
    # Start by filling each row
    for row in range(rows):
        for col in range(cols):
            # Get number of cell row major
            cell_num = karyotype_template[row][col]
            # Create a white empty image of 200,200,3
            image = np.ones((200, 200, 3), np.uint8) * 255
            # Check if there is an image in preds with the key (row+col+1)
            if cell_num in preds:
                image = preds[cell_num]
            # Add black borders to the image
            image = cv2.copyMakeBorder(
                image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Resize the image to 200x200
            image = cv2.resize(image, (200, 200))
            # Write (row+col+1) on the image bottom middle
            if cell_num != 0:
                cell_num = 'X' if cell_num == 23 else cell_num
                cell_num = 'Y' if cell_num == 24 else cell_num
                cv2.putText(image, str(cell_num), (80, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            # Add the image to the karyotype
            karyotype[row*200:(row+1)*200, col*200:(col+1)*200] = image
    return karyotype


def concat_and_pad_images(image_list):
    """_summary_

    Args:
        image_list (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(image_list) == 0:
        # Create a blank white image
        final_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    elif len(image_list) == 1:
        # Pad a single image to 200x200
        image, _ = image_list[0]
        image = pad_image(image, target_shape=(200, 200))
        final_image = image
    else:
        # Concatenate and pad multiple images
        images = [image for image, _ in image_list]
        widths = [image.shape[1] for image in images]
        heights = [image.shape[0] for image in images]
        max_height = max(heights)
        total_width = sum(widths)
        final_image = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        x_offset = 0
        #cv2.resize(im, (max_height, im.shape[1]),interpolation = cv2.INTER_LINEAR)
        images = [pad_image(im, (max_height, im.shape[1])) for im in images]
        #images = [cv2.resize(im, (max_height, im.shape[1]),interpolation = cv2.INTER_LINEAR) for im in images]
        for image in images:
            final_image[:image.shape[0], x_offset:x_offset +image.shape[1], :] = image
            x_offset += image.shape[1]
        # Resize and pad the concatenated image to 200x200
        final_image = pad_image(final_image, target_shape=(200, 200))
    return final_image


def pad_image(image, target_shape):
    """_summary_

    Args:
        image (_type_): _description_
        target_shape (_type_): _description_

    Returns:
        _type_: _description_
    """
    padded_image = np.ones(
        (target_shape[0], target_shape[1], 3), dtype=np.uint8) * 255
    height_pad = (target_shape[0] - image.shape[0]) // 2
    width_pad = (target_shape[1] - image.shape[1]) // 2
    padded_image[height_pad:height_pad + image.shape[0],
                 width_pad:width_pad + image.shape[1], :] = image
    return padded_image


def process_images(dictionary):
    """_summary_

    Args:
        dictionary (_type_): _description_

    Returns:
        _type_: _description_
    """
    processed_dictionary = {}
    for pair_number, image_list in dictionary.items():
        final_image = concat_and_pad_images(image_list)
        processed_dictionary[pair_number] = final_image
    return processed_dictionary

def get_api_out(image_list):
    # Create the input payload
    input_payload = {"images": image_list}
    # URL of your FastAPI endpoint
    api_url = "http://3.7.234.80:8001/classify"
    # Send POST request with JSON payload
    response = requests.post(api_url, json=input_payload)
    # print(response.content)
    unpickled_preds = pickle.loads(response.content)
    print(response.headers)
    return unpickled_preds

# def crop_output(out):
#     """
#     Description : The server returns images which are padded to a high extend that they hinder visibility to fix this
#     cropping each image . This function will throw an error if any image is cropped to an extend that all remains is whitespace
#     Args: Dictionary of images
#     Returns: Dictionary of cropped images
#     """
#     ret_dict=dict()
#     for i in out.keys():
#         h,w,_=out[i].shape
#         off=int(h*0.4)
#         end=off+100
#         if end> h:
#             end=h
#         img=out[i][off:end,off:end]
#         if np.mean(img) == 255:
#             raise Exception('ALL WHITE IMAGE CROP GONE WRONG!')
#         ret_dict[i]=img
#     return ret_dict


def pipeline(path_pkl_file):
    """
    :param path_pkl_file:  Path to the pkl file which is a pickled object of a list of images
    :return: a single image which is the Karyotype
    """
    with open(path_pkl_file, 'rb') as file:
        img_list = pickle.load(file)
    image_data_list = [arr.tolist() for arr in img_list]
    api_out = get_api_out(image_data_list)
    processed_dict = process_images(api_out)
    with open("api_out.pkl", 'wb') as file:
        pickle.dump(processed_dict,file)
    fin_out = get_karyotype(processed_dict)
    return fin_out
