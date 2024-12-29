import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
from gui_7 import combined_image
import cv2

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Load YOLOv8 model from the .pt file
strained_model = YOLO('best (21).pt')

# Global variable to store the selected image path
file_path_1 = None
file_path_2 = None
image1_references = None
image2_references = None

# Function to select an image and display it in the GUI
def select_image_1():
    global file_path_1,image1_references
    # Open file dialog to select an image
    file_path_1 = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if not file_path_1:
        print("No image selected.")
        return

    # Load the image using PIL
    img = Image.open(file_path_1)
    
    # Resize the image to fit the GUI window (if necessary)
    img = img.resize((319, 160), Image.LANCZOS)
    
    # Convert the image to a format Tkinter can use
    image1_references = ImageTk.PhotoImage(img)
    
    # Display the image in the label
    image_label.config(image=image1_references)
    image_label.image = image1_references  # Keep a reference to avoid garbage collection

    canvas.create_image(
        640,  # X center of the rectangle
        160,  # Y center of the rectangle
        image=image1_references,
        anchor="center"  # Anchor the image to the center of the coordinates
    )

    print('selected')
# Function to run YOLO model on the selected image

## backimage
def select_image_2():
    global file_path_2,image2_references
    # Open file dialog to select an image
    file_path_2 = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if not file_path_2:
        print("No image selected.")
        return

    # Load the image using PIL
    img = Image.open(file_path_2)
    
    # Resize the image to fit the GUI window (if necessary)
    img = img.resize((319, 160), Image.LANCZOS)
    
    # Convert the image to a format Tkinter can use
    image2_references = ImageTk.PhotoImage(img)
    
    # Display the image in the label
    image_label.config(image=image2_references)
    image_label.image = image2_references  # Keep a reference to avoid garbage collection

    canvas.create_image(
        640,  # X center of the rectangle
        371,  # Y center of the rectangle
        image=image2_references,
        anchor="center"  # Anchor the image to the center of the coordinates
    )

    print('selected')
def draw_bounding_boxes(image_path, results, class_colors):
    # Load the image
    image = Image.open(image_path).convert("RGBA")  # Use RGBA for transparency
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes with colors
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

            # Draw filled rectangle for the bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], fill=class_colors[class_id])

    return image

def crop_image(image, crop_area):
    """
    Crop the image based on the given crop area.
    crop_area is a tuple (x_min, y_min, x_max, y_max)
    """
    return image.crop(crop_area)

def run_yolo():
    global file_path_1, file_path_2
    
    # Check if the first image is selected
    if not file_path_1:
        print("No first image selected. Please select the first image.")
        return

    # Check if the second image is selected
    if not file_path_2:
        print("No second image selected. Please select the second image.")
        return

    # Define class colors and names
    class_colors = {
        1: "#FCFCFC",  # No Egg - white
        2: "#D4D70E",  # Strained - Yellow
        3: "#25DA58",  # Uncracked - Green
        4: "#D12B2B"   # Cracked - red
    }
    class_names = {
        1: "No Egg",
        2: "Strained",
        3: "Uncracked",
        4: "Cracked"
    }
    
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Run YOLO on the first image (with bounding box size calculation)
    print("Running YOLO on the first image...")
    results1 = strained_model(file_path_1)

    # Extract class names, confidence scores, and bounding box sizes for the first image
    print("YOLO results for Image 1:")
    for result in results1:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            class_id = int(box.cls)  # Get class index (integer)
            confidence = box.conf.item()  # Get confidence score

            # Get bounding box coordinates (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

            # Calculate bounding box width and height
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            print(f"Image 1 - Detected class ID: {class_id}, Confidence: {confidence}")
            print(f"Bounding Box Size - Width: {bbox_width}, Height: {bbox_height}")

    # Run YOLO on the second image (without bounding box size calculation)
    print("Running YOLO on the second image...")
    results2 = strained_model(file_path_2)

    # Extract class names and confidence scores for the second image
    print("YOLO results for Image 2:")
    for result in results2:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            class_id = int(box.cls)  # Get class index (integer)
            confidence = box.conf.item()  # Get confidence score

            print(f"Image 2 - Detected class ID: {class_id}, Confidence: {confidence}")

    # Create new images with bounding boxes for both images
    colored_image1 = draw_bounding_boxes(file_path_1, results1, class_colors)
    colored_image2 = draw_bounding_boxes(file_path_2, results2, class_colors)

    # Define cropping areas (customize these values as needed)
    crop_area1 = (580, 100, 1450, 950)  # Example crop area for the first image (x_min, y_min, x_max, y_max)
    crop_area2 = (540, 95, 1450, 980)  # Example crop area for the second image (x_min, y_min, x_max, y_max)

    # Crop the images
    cropped_image1 = crop_image(colored_image1, crop_area1)
    cropped_image2 = crop_image(colored_image2, crop_area2)
    
    cropped_image1 = cropped_image1.convert("RGB")
    cropped_image2 = cropped_image2.convert("RGB")

    # Save the cropped images to the output folder
    cropped_image1_path = os.path.join(output_folder, "cropped_image_1.jpg")
    cropped_image2_path = os.path.join(output_folder, "cropped_image_2.jpg")
    print(cropped_image2_path)
    cropped_image1.save(cropped_image1_path)  # Save the first cropped image
    cropped_image2.save(cropped_image2_path)  # Save the second cropped image
    
    # Display the cropped images
    # cropped_image1.show()  # Display the first cropped image
    # cropped_image2.show()  # Display the second cropped image
    cropp1 = cv2.imread(cropped_image1_path)
    cropp2 = cv2.imread(cropped_image2_path)
    output_image = combined_image(cropp1, cropp2)  # Combine and display the cropped images in a single window
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    pil_output_image = Image.fromarray(output_image)

                # Resize the PIL image to fit the canvas (optional)
    resized_pil_image = pil_output_image.resize((423, 375), Image.LANCZOS)

                # Convert the resized PIL image to Tkinter format
    tk_image = ImageTk.PhotoImage(resized_pil_image)

                # Assuming you have a Tkinter canvas to display this image
    image_label.config(image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

    canvas.create_image(240, 263, image=tk_image, anchor="center")
    print("Both images processed and cropped successfully.")

window = Tk()

window.geometry("982x559")
window.configure(bg = "#434A6D")

image_label = tk.Label(window)
image_label.pack()

canvas = Canvas(
    window,
    bg = "#434A6D",
    height = 559,
    width = 982,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    499.0,
    40.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    906.0,
    327.0,
    image=image_image_2
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    904.0,
    477.0,
    image=image_image_3
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command= select_image_1,
    relief="flat"
)
button_1.place(
    x=205.0,
    y=505.0,
    width=91.0,
    height=34.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command= select_image_2,
    relief="flat"
)
button_2.place(
    x=333.0,
    y=505.0,
    width=91.0,
    height=34.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command= run_yolo,
    relief="flat"
)
button_3.place(
    x=461.0,
    y=505.0,
    width=88.0,
    height=34.0
)


canvas.create_rectangle(
    37.0,
    77.0,
    452.0,
    451.0,
    fill="#D9D9D9",
    outline="")

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    639.0,
    254.0,
    image=image_image_4
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    639.0,
    469.0,
    image=image_image_5
)
window.resizable(False, False)
window.mainloop()
