# Medical Image Identification System

This project is a medical image identification system built using PyTorch and a ResNet18 model to classify images of various body parts. The system uses a Tkinter-based graphical user interface (GUI) to allow users to import, train, and test models for different body parts, such as brain, lungs, knee, and hand.

## Features
- Add training images for body parts.
- Train a model using a custom dataset.
- Test the model with new medical images.
- Display model predictions and accuracy.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Task2-Yolo-Ai_Detector/medical-image-identification.git
    ```

2. Navigate to the project directory:
    ```bash
    cd AI Detector
    ```

3. Install the dependencies:
    ```bash
    pip install torch torchvision Pillow numpy
    ```

## Usage

1. Run the app:
    ```bash
    main.py
    ```

2. Use the GUI to:
   - Add training images for specific body parts.
   - Train the model and test it with new images.

## Training Model

Ensure you have collected images for each body part (brain, lungs, knee, hand) to train the model. Images can be added through the GUI.


