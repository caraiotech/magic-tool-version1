import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Label, Scrollbar, Toplevel
from PIL import Image, ImageTk
import pickle
from utils import masks_to_contours, extract_chromosomes_with_contours, contours_to_bboxes, pipeline
from api import get_masks_with_bbox
from config import image_path


# Global variable to store the contours
stored_contours = []
current_scale_factor = 1.0  # Initial scale factor
region_image_tk = None  # Initial region image


def open_chromosome_image_window(coords, chromosome_image, original_image, all_contours):
    global region_image_tk
    # Open a new top-level window for the chromosome image
    magnified_window = tk.Toplevel()
    magnified_window.title("Chromosome Image")


    # Create a canvas to draw the original image and the contours
    canvas = Canvas(magnified_window, width=original_image.shape[1],
                    height=original_image.shape[0])
    canvas.grid(row=0,column=0,rowspan=11,columnspan=2)
    # outer_canvas.create_window((0, 0), window=canvas, anchor=tk.NW)


    # canvas.pack()

    # Convert the original image to a PIL image
    region_image = Image.fromarray(original_image)
    # Convert the PIL image to a Tkinter-compatible image
    region_image_tk = ImageTk.PhotoImage(region_image)
    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)

    # Draw a point at the center of the chromosome
    x, y = coords
    x, y = int(x), int(y)
    canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")

    drawing = False
    line_points = []

    def save_contours():
        global current_scale_factor
        print("Save", current_scale_factor)
        descale_contour_points()
        # Save the contours
        with open('stored_contours.pkl', 'wb') as f:
            pickle.dump(all_contours, f)

    def start_draw(event):
        global drawing
        drawing = True
        # line_points.clear()
        x = canvas.canvasx(event.x) 
        y = canvas.canvasy(event.y) 
        line_points.append((x,y))

    def continue_draw(event):
        global drawing
        global current_scale_factor
        if drawing:

            # Adjust the event coordinates based on panning and zooming
            x = canvas.canvasx(event.x) 
            y = canvas.canvasy(event.y) 
            line_points.append((x, y))
            # print(x, y)
            canvas.create_line(
                line_points[-2][0], line_points[-2][1], x,y, fill="blue")

    def end_draw(event):
        global drawing
        drawing = False

    def draw_contours():
        return line_points

    def create_initial_contour():
        global region_image_tk
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        contour_colors = ["red", "blue", "green", "orange",
                          "purple"]  # List of colors for contours
        print(len(all_contours))
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # print(flattened_contour)
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            # If it's a point, create an oval
            if (len(flattened_contour) < 4):
                canvas.create_oval(flattened_contour[0] - 5, flattened_contour[1] - 5,flattened_contour[0] + 5, flattened_contour[1] + 5, fill='red')
            else:
                canvas.create_line(flattened_contour, fill=color, width=2)

    def add_contours():
        global region_image_tk

        # Assuming you have a list of line points (drawn by the user) representing the extension
        new_line_points = draw_contours()

        if new_line_points:
            # Convert the drawn line into a new contour
            new_contour = []

            # Find the contour to extend using the drawn line endpoints
            contour_to_extend = None  # Initialize the variable
            left_intersection = None
            right_intersection = None

            # Start from the left opf the new line, and check for the first contour it intersects, and store thr point
            left_contour = None
            right_contour = None
            for point in new_line_points:
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        left_contour = contour
                        left_intersection = point
                        break
                if left_contour is not None:
                    break
            # Start from the right opf the new line, and check for the first contour it intersects, and store thr point
            for point in reversed(new_line_points):
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        right_contour = contour
                        right_intersection = point
                        break
                if right_intersection is not None:
                    break

            # If left and right contours are the same, then the new line is inside the contour
            if np.array_equal(left_contour, right_contour):
                contour_to_extend = left_contour
            else:
                return
            

            # If left and right interstions exist, then extend the contour
            if left_intersection is not None and right_intersection is not None and contour_to_extend is not None:
                # Convert new_line_points to a list of arrays
                new_line_contour = [np.array(new_line_points, dtype=np.int32)]

                # Create a mask using the new line
                mask = np.zeros((region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(mask, new_line_contour, -1, 255, -1)
                # save the mask as image
                cv2.imwrite("mask.png", mask)

                # All the points of the contour to extend inside the mask should be removed from the contour
                # Create a mask using the contour to extend
                contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(
                    contour_mask, [contour_to_extend], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("contour_mask.png", contour_mask)

                # Apply the mask to the contour bitwise or
                contour_masked = cv2.bitwise_or(contour_mask, mask)
                # save the mask as image
                cv2.imwrite("contour_masked.png", contour_masked)

                # Get the boundary of the mask in countour_masked
                contours, _ = cv2.findContours(
                    contour_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Replace the contour to extend with the largest contour
                for idx, contour in enumerate(all_contours):
                    if np.array_equal(contour, contour_to_extend):
                        all_contours[idx] = largest_contour

        # Draw the new contour on the canvas=
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)

    def remove_contours():

        # Assuming you have a list of line points (drawn by the user) representing the extension
        new_line_points = draw_contours()

        if new_line_points:
            # Convert the drawn line into a new contour
            new_contour = []

            # Find the contour to extend using the drawn line endpoints
            contour_to_extend = None  # Initialize the variable
            left_intersection = None
            right_intersection = None

            # Start from the left opf the new line, and check for the first contour it intersects, and store thr point
            left_contour = None
            right_contour = None
            for point in new_line_points:
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        left_contour = contour
                        left_intersection = point
                        break
                if left_contour is not None:
                    break
            # Start from the right opf the new line, and check for the first contour it intersects, and store thr point
            for point in reversed(new_line_points):
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        right_contour = contour
                        right_intersection = point
                        break
                if right_intersection is not None:
                    break

            # If left and right contours are the same, then the new line is inside the contour
            if np.array_equal(left_contour, right_contour):
                contour_to_extend = left_contour
            else:
                return

            # If left and right interstions exist, then extend the contour
            if left_intersection is not None and right_intersection is not None:
                # Convert new_line_points to a list of arrays
                new_line_contour = [np.array(new_line_points, dtype=np.int32)]

                # Create a mask using the new line
                mask = np.zeros((region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(mask, new_line_contour, -1, 255, -1)
                # invert the mask
                mask = cv2.bitwise_not(mask)
                # save the mask as image
                cv2.imwrite("mask.png", mask)

                # All the points of the contour to extend inside the mask should be removed from the contour
                # Create a mask using the contour to extend
                contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(
                    contour_mask, [contour_to_extend], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("contour_mask.png", contour_mask)

                # Apply the mask to the contour bitwise or
                contour_masked = cv2.bitwise_and(contour_mask, mask)
                # save the mask as image
                cv2.imwrite("contour_masked.png", contour_masked)

                # Get the boundary of the mask in countour_masked
                contours, _ = cv2.findContours(
                    contour_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Replace the contour to extend with the largest contour
                for idx, contour in enumerate(all_contours):
                    if np.array_equal(contour, contour_to_extend):
                        all_contours[idx] = largest_contour

        # Draw the new contour on the canvas
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)

    def split_contours():

        # Assuming you have a list of line points (drawn by the user) representing the extension
        new_line_points = draw_contours()

        if new_line_points:
            # Convert the drawn line into a new contour
            new_contour = []

            # Find the contour to extend using the drawn line endpoints
            contour_to_extend = None  # Initialize the variable
            left_intersection = None
            right_intersection = None

            # Start from the left opf the new line, and check for the first contour it intersects, and store thr point
            left_contour = None
            right_contour = None
            for point in new_line_points:
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        left_contour = contour
                        left_intersection = point
                        break
                if left_contour is not None:
                    break
            # Start from the right opf the new line, and check for the first contour it intersects, and store thr point
            for point in reversed(new_line_points):
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        right_contour = contour
                        right_intersection = point
                        break
                if right_intersection is not None:
                    break

            # If left and right contours are the same, then the new line is inside the contour
            if np.array_equal(left_contour, right_contour):
                contour_to_extend = left_contour
            else:
                return

            # If left and right interstions exist, then split the contour
            if left_intersection is not None and right_intersection is not None:
                # Convert new_line_points to a list of arrays
                new_line_contour = [np.array(new_line_points, dtype=np.int32)]

                # Create a mask using the new line
                mask = np.zeros((region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(mask, new_line_contour, -1, 255, -1)
                # invert the mask
                mask = cv2.bitwise_not(mask)
                # save the mask as image
                cv2.imwrite("mask.png", mask)

                # All the points of the contour to extend inside the mask should be removed from the contour
                # Create a mask using the contour to extend
                contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(
                    contour_mask, [contour_to_extend], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("contour_mask.png", contour_mask)

                # Apply the mask to the contour bitwise or
                contour_masked = cv2.bitwise_and(contour_mask, mask)
                # save the mask as image
                cv2.imwrite("contour_masked.png", contour_masked)

                # Get the boundary of the mask in countour_masked
                contours, _ = cv2.findContours(
                    contour_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Replace the contour with the contours found
                for idx, contour in enumerate(all_contours):
                    if np.array_equal(contour, contour_to_extend):
                        all_contours.pop(idx)
                        all_contours.extend(contours)

        # Draw the new contour on the canvas
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)

    def join_contours():

        # Assuming you have a list of line points (drawn by the user) representing the extension
        new_line_points = draw_contours()

        if new_line_points:

            first_contour = None
            # Start from the left of the line and find the first contour it intersects
            for point in new_line_points:
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        first_contour = contour
                        break
                if first_contour is not None:
                    break
            second_contour = None
            # Start from the right of the line and find the first contour it intersects
            for point in reversed(new_line_points):
                for contour in all_contours:
                    if cv2.pointPolygonTest(contour, point, False) == 0:
                        second_contour = contour
                        break
                if second_contour is not None:
                    break
            if (first_contour is not None) and (second_contour is not None) and (not first_contour.shape == second_contour.shape):
                # Create a mask from the new line
                new_line_contour = [np.array(new_line_points, dtype=np.int32)]
                mask = np.zeros((region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(mask, new_line_contour, -1, 255, -1)
                # save the mask as image
                cv2.imwrite("mask.png", mask)
                # Create a mask from the first contour
                first_contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(first_contour_mask, [
                                 first_contour], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("first_contour_mask.png", first_contour_mask)
                # Create a mask from the second contour
                second_contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(
                    second_contour_mask, [second_contour], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("second_contour_mask.png", second_contour_mask)
                # Combine the all the three masks
                combined_mask = cv2.bitwise_or(
                    first_contour_mask, second_contour_mask)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
                # save the mask as image
                cv2.imwrite("combined_mask.png", combined_mask)
                # Get the boundary of the mask in combined_mask
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Replace the contour to extend with the largest contour
                for idx, contour in enumerate(all_contours):
                    if np.array_equal(contour, first_contour):
                        all_contours[idx] = largest_contour
                    elif np.array_equal(contour, second_contour):
                        all_contours.pop(idx)
                        break

        # Draw the new contour on the canvas
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)

    def create_new_contour():
        # Create a new contour
        new_contour = []
        for point in line_points:
            new_contour.append([[point[0], point[1]]])
        # Add the new contour to the list of contours
        all_contours.append(np.array(new_contour, dtype=np.int32))

        # Draw the new contour on the canvas
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)
    
    # Remove contours based on the drawn line
    def delete_contours():
        # Assuming you have a list of line points (drawn by the user) representing the extension
        new_line_points = draw_contours()

        if new_line_points:
            # Get the mask from the drawn line
            new_line_contour = [np.array(new_line_points, dtype=np.int32)]
            mask = np.zeros((region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
            cv2.drawContours(mask, new_line_contour, -1, 255, -1)
            # save the mask as image
            cv2.imwrite("mask.png", mask)

            # Get the contours inside the mask
            contours_inside_mask = []
            for contour in all_contours:
                # Create a mask from the contour
                contour_mask = np.zeros(
                    (region_image_tk.width(), region_image_tk.height()), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                # save the mask as image
                cv2.imwrite("contour_mask.png", contour_mask)
                # Apply the mask to the contour bitwise or
                contour_masked = cv2.bitwise_and(contour_mask, mask)
                # save the mask as image
                cv2.imwrite("contour_masked.png", contour_masked)
                # Get the boundary of the mask in countour_masked
                contours, _ = cv2.findContours(
                    contour_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    contours_inside_mask.append(contour)
                
            # Remove the contours inside the mask
            for contour_to_remove in contours_inside_mask:
                for idx, contour in enumerate(all_contours):
                    if np.array_equal(contour, contour_to_remove):
                        all_contours.pop(idx)
                        break
            
       # Draw the new contour on the canvas=
        canvas.delete("all")
        line_points.clear()
        canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
        # List of colors for contours
        contour_colors = ["red", "blue", "green", "orange", "purple"]
        for i, contour in enumerate(all_contours):
            flattened_contour = [
                coord for point in contour for coord in point[0]]
            # Get a color based on index
            color = contour_colors[i % len(contour_colors)]
            canvas.create_line(flattened_contour, fill=color, width=2)


    def on_canvas_right_click(event):
        canvas.scan_mark(event.x, event.y)

    def on_canvas_right_release(event):
        canvas.scan_dragto(event.x, event.y, gain=1)

    canvas.bind("<Button-3>", on_canvas_right_click)
    canvas.bind("<ButtonRelease-3>", on_canvas_right_release)

    # Scale the contour points based on the current scale factor
    def scale_contour_points():
        global current_scale_factor
        print("Scale", current_scale_factor)
        for contour in all_contours:
            for point in contour:
                point[0][0] = int(point[0][0] * current_scale_factor)
                point[0][1] = int(point[0][1] * current_scale_factor)
    
    def descale_contour_points():
        global current_scale_factor
        print("Descale", current_scale_factor)
        for contour in all_contours:
            for point in contour:
                point[0][0] = int(point[0][0] / current_scale_factor)
                point[0][1] = int(point[0][1] / current_scale_factor)
            



    # Zoom in
    def zoom_in():
        global current_scale_factor
        global region_image_tk
        if current_scale_factor < 2:
            current_scale_factor += 1
            scaled_image = cv2.resize(original_image, None, fx=current_scale_factor, fy=current_scale_factor)
            scaled_region_image = Image.fromarray(scaled_image)
            region_image_tk = ImageTk.PhotoImage(scaled_region_image)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
            canvas.image = region_image_tk
            # Scale the contour points based on the current scale factor
            scale_contour_points()

    # Zoom out
    def zoom_out():
        global current_scale_factor
        global region_image_tk
        if current_scale_factor > 1:
            # De scale the contour points based on the current scale factor
            descale_contour_points()
            current_scale_factor -= 1
            scaled_image = cv2.resize(original_image, None, fx=current_scale_factor, fy=current_scale_factor)
            scaled_region_image = Image.fromarray(scaled_image)
            region_image_tk = ImageTk.PhotoImage(scaled_region_image)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)
            canvas.image = region_image_tk
            




    # Bind the mouse events to functions
    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<B1-Motion>", continue_draw)
    canvas.bind("<ButtonRelease-1>", end_draw)
    magnified_window.grid_columnconfigure(1, minsize=100)
    # Create a button to zoom in
    zoom_in_button = Button(
        magnified_window, text="Zoom In",height=3, width=15, command=zoom_in)
    zoom_in_button.grid(row=0,column=2,columnspan=2)
    # Create a button to zoom out
    zoom_out_button = Button(
        magnified_window, text="Zoom Out",height=3, width=15, command=zoom_out)
    zoom_out_button.grid(row=1,column=2,columnspan=2)

    # Create a button to create a new contour
    new_contour_button = Button(
        magnified_window, text="New Contour",height=3, width=15, command=create_new_contour)
    new_contour_button.grid(row=2, column=2,columnspan=2)
    # Initial contour button: Draw the initial contour
    initial_contour_button = Button(
        magnified_window, text="Initial Contour",height=3, width=15, command=create_initial_contour)
    initial_contour_button.grid(row=3, column=2,columnspan=2)
    # Add contour button: Add a new contour
    add_contour_button = Button(
        magnified_window, text="Add Contour",height=3, width=15, command=add_contours)
    add_contour_button.grid(row=4, column=2,columnspan=2)
    # Remove contour button: Remove a contour
    remove_contour_button = Button(
        magnified_window, text="Remove Contour",height=3, width=15, command=remove_contours)
    remove_contour_button.grid(row=5, column=2,columnspan=2)
    # Delete contour button: Delete a contour
    delete_contour_button = Button(
        magnified_window, text="Delete Contour", height=3, width=15,command=delete_contours)
    delete_contour_button.grid(row=6, column=2,columnspan=2)
    # Split contour button: Split a contour
    split_contour_button = Button(
        magnified_window, text="Split Contour", height=3, width=15,command=split_contours)
    split_contour_button.grid(row=7, column=2,columnspan=2)
    # Join contour button: Join two contours
    join_contour_button = Button(
        magnified_window, text="Join Contour",height=3, width=15, command=join_contours)
    join_contour_button.grid(row=8, column=2,columnspan=2)
    # Save contours button: Save the contours
    save_contours_button = Button(
        magnified_window, text="Save Contours", height=3, width=15,command=save_contours)
    save_contours_button.grid(row=9, column=2,columnspan=2)

    def close_chromosome_window():
        # Destroy the chromosome window
        magnified_window.destroy()

    # Add a button to close the chromosome window
    close_button = Button(magnified_window, text="Done",height=5, width=10,command=close_chromosome_window)
    close_button.grid(row=0,column=4,rowspan=8)

    magnified_window.mainloop()


def call_sam(root, image_numpy):
    # Load the stored contours
    global stored_contours
    # Call the SAM algorithm
    bbox = contours_to_bboxes(stored_contours)
    masks, _ = get_masks_with_bbox(image_numpy, bbox)
    stored_contours = masks_to_contours(masks)
    # Write the contours to a file
    with open('stored_contours.pkl', 'wb') as f:
        pickle.dump(stored_contours, f)
    # Remove the previous grid
    for widget in root.winfo_children():
        widget.destroy()
    # Display the grid with scaled chromosomes
    display_chromosomes_grid(root)

def reset_chromosomes(root):
    # Load the stored contours
    global stored_contours
    with open('stored_contours.pkl', 'rb') as f:
        stored_contours = pickle.load(f)

    # Remove the previous grid
    for widget in root.winfo_children():
        widget.destroy()

    # Display the grid with scaled chromosomes
    display_chromosomes_grid(root)

# Display top level window
def show_karyotype_window(chromosomes):
    karyotype_window = tk.Toplevel()
    karyotype_window.title("Karyotype Image")

    pkl_path = "chromosomes.pkl"
    image = pipeline(pkl_path)

    # Create a canvas to draw the original image and the contours
    canvas = Canvas(karyotype_window, width=image.shape[1],height=image.shape[0])
    canvas.grid(row=0, column=0)
    # outer_canvas.create_window((0, 0), window=canvas, anchor=tk.NW)


    # canvas.pack()

    # Convert the original image to a PIL image
    region_image = Image.fromarray(image)
    # Convert the PIL image to a Tkinter-compatible image
    region_image_tk = ImageTk.PhotoImage(region_image)
    # Add the image to the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=region_image_tk)

    def close_chromosome_window():
        # Destroy the chromosome window
        karyotype_window.destroy()

    # Add a button to close the chromosome window
    close_button = Button(karyotype_window, text="Done",
                          command=close_chromosome_window)
    close_button.grid(row=1,column=1)

    karyotype_window.mainloop()
def padded_chromosome(chromosome,dim):
    h,w,_=chromosome.shape
    print(h,w)
    h_add=dim[0]-h
    w_add=dim[1]-w
    top=h_add//2
    bottom=h_add//2
    left=w_add//2
    right=w_add//2
    white = [255,255,255]
    print(f"DIMS: {top},{bottom},{right},{left}")
    padded_img = cv2.copyMakeBorder(chromosome, top, bottom, left, right,cv2.BORDER_CONSTANT,value=white)
    return padded_img
def calculate_dimentions(list_chromosomes):
    max_h=0
    max_w=0
    for chr in list_chromosomes:
        h,w,_=chr.shape
        if(h>max_h):
            max_h=h
        if (w>max_w):
            max_w=w
    return (max_h,max_w)
def display_chromosomes_grid(root):
    root.title("Chromosomes Grid")

    # Load the image
    # Load your metaphase image here
    image_numpy = Image.open(image_path)
    # If image is greater tahn 512x512, resize it
    if image_numpy.size[0] > 512:
        image_numpy = image_numpy.resize((512, 512))
    image_numpy = np.array(image_numpy)

    # Load the stored contours
    global stored_contours
    with open('stored_contours.pkl', 'rb') as f:
        stored_contours = pickle.load(f)

    # Extract chromosomes from the original image using the contours
    chromosomes, chromosomes_coords = extract_chromosomes_with_contours(
        stored_contours, image_numpy)
    
    # Save these chromosomes in a pickle file
    with open('chromosomes.pkl', 'wb') as f:
        pickle.dump(chromosomes, f)

    # Create a grid layout
    # When a chromosome is clicked, open a new window with the chromosome image
    # Open a new window without destroying the grid window
    #
    for i, chromosome in enumerate(chromosomes):
        # Resize the chromosome image 1.5x
        #scaled_chromosome = cv2.resize(chromosome, None, fx=2, fy=2)
        dim = calculate_dimentions(chromosomes)
        pad_chromosome = padded_chromosome(chromosome, dim)
        #scaled_chromosome = cv2.resize(pad_chromosome, None, fx=2, fy=2)
        # Convert the chromosome to a PIL image
        chromosome_image = Image.fromarray(pad_chromosome)
        # Convert the PIL image to a Tkinter-compatible image

        chromosome_image_tk = ImageTk.PhotoImage(chromosome_image)

        # Add the image to the grid layout
        label = Label(root, image=chromosome_image_tk)
        # Arrange in 4 rows, 5 columns
        label.grid(row=i // 10, column=i % 10)
        # Bind the click event to a function

        label.bind("<Button-1>", lambda e, coords=chromosomes_coords[i],img=pad_chromosome, original_image=image_numpy, contours=stored_contours: open_chromosome_image_window(coords, img, original_image, contours))
        # Keep a reference to the image so that the image is not garbage collected
        label.image = chromosome_image_tk
    # Display the number of the chromosome
    label = Label(root, text=str(i+1))
    label.grid(row = 1, column = 11)



    # Add the Reset button
    reset_button = Button(root, text="Reset",command=lambda: reset_chromosomes(root))
    # Place it at the bottom of the grid
    reset_button.grid(row=0, column=11)
    # Add call SAM button
    call_sam_button = Button(root, text="Call SAM",command=lambda: call_sam(root, image_numpy))
    # Place it at the bottom of the grid
    call_sam_button.grid(row=2, column=11)
    # Add call to show_karyotype
    karyotype_button = Button(root, text="Classify",
                              command=lambda: show_karyotype_window(chromosomes))
    karyotype_button.grid(row=3,column=11)

    root.mainloop()
