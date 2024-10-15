# Chromosome Segmentation and Manipulation Application

The Chromosome Segmentation and Manipulation Application is a graphical user interface (GUI) tool that allows users to segment chromosomes in a metaphase image, manipulate their contours, and visualize the results in real time.
Getting Started

Follow these instructions to set up and run the application on your local machine.
Prerequisites

    Python 3.x
    Virtual environment (recommended)

Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/chromosome-segmentation-app.git

Navigate to the project directory:

bash

cd chromosome-segmentation-app

Install dependencies:

bash

    pip install -r requirements.txt

Usage

    Run the application:

    bash

    python run.py

    The application window will open, displaying the initial metaphase image.

    Click the "Load Image" button to load a metaphase image.

    Click the "Segment Chromosomes" button to segment the chromosomes in the image using an API. The segmented chromosomes will be displayed in a grid.

    Click on a segmented chromosome in the grid to view a magnified image of the selected chromosome.

    Use the chromosome manipulation tool to modify the contours of the selected chromosome.

    Click the "Apply Changes" button to update the grid with the modified chromosomes.

    You can reset the grid to the original segmented chromosomes using the "Reset" button.

    To exit the application, close the main window.

Project Structure

The project is organized as follows:

    images/: Directory to store example images.
    app/: Main application package.
        main.py: Main application logic.
        api.py: API interaction functions.
        utils.py: Utility functions.
    run.py: Script to run the application.

    Okay great I got the basics workingf
Now I have to make a fulkly flegded GUI python app with Tkinter for segmentatio of chromosome and it's manipulation with our tool.
The flow is this
* We get an image of metpahse of chromosomes from the user
* Display the image, and call our api (api is already implemenetd)
* The API will return the initial masks of identified chromosomes
* Convert these masks to a list of contours 
* Now display a grid consisting of the segmented chromosomes (segment using the contours list)
* When a user clicks on a chromosomes image in the grid, it should open a new window with a magnified image of the metpahse focused on the clicked chromosomes poistion
* Now the user can use the tool to manipulate the contours, once he is done, he will hit the ok button, and we will now show him a grid of the updated images based on the changes (if any) that are made

How do I get started with this undertaking
Let's start step by step
Let's create a project folder, think of the structure, think of what functions will be used, how to handle the flow of events and so on
Keep this (the *)instructions in mind when later we write the code part 