import cv2
import numpy as np

# Load the two input images
# image1 = cv2.imread(r'C:\Users\DNR\Documents\vscodepython\python_gui\New folder\build\output\cropped_image_1.jpg')  # First image
# image2 = cv2.imread(r'C:\Users\DNR\Documents\vscodepython\python_gui\New folder\build\output\cropped_image_2.jpg')  # Second image
def combined_image(image1, image2):
    
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # Now you can access the shape
    height1, width1, _ = image1_np.shape
    height2, width2, _ = image2_np.shape
    # Check if the images are loaded successfully
    if image1 is None:
        print("Error: Image 1 could not be loaded. Please check the file path.")
    else:
        print("Image 1 loaded successfully.")

    if image2 is None:
        print("Error: Image 2 could not be loaded. Please check the file path.")
    else:
        print("Image 2 loaded successfully.")

    # Resize both images to the same dimensions if they are loaded
    if image1 is not None and image2 is not None:
        height1, width1, _ = image1.shape
        height2, width2, _ = image2.shape

        # Find the minimum height and width to resize both images to the same dimensions
        min_height = min(height1, height2)
        min_width = min(width1, width2)

        # Resize both images
        image1_resized = cv2.resize(image1, (min_width, min_height))
        image2_resized = cv2.resize(image2, (min_width, min_height))

        # Set up the grid size
        grid_size = 4  # 4x4 grid
        grid_h = min_height // grid_size
        grid_w = min_width // grid_size

        # Create an empty image for output
        output_image = np.ones((min_height, min_width, 3), dtype=np.uint8) * 255  # Start with a white image

        # Function to determine if the average color falls within a specific range
        def is_color_in_range(color, lower_bound, upper_bound):
            return np.all(color >= lower_bound) and np.all(color <= upper_bound)

        # Define color ranges for red, green, white, and yellow
        red_lower_bound = np.array([0, 0, 100])
        red_upper_bound = np.array([100, 100, 255])
        green_lower_bound = np.array([0, 100, 0])
        green_upper_bound = np.array([100, 255, 100])
        white_lower_bound = np.array([200, 200, 200])
        white_upper_bound = np.array([255, 255, 255])
        yellow_lower_bound = np.array([0, 100, 100])
        yellow_upper_bound = np.array([100, 255, 255])

        # Function to determine the output color and label based on the rules
        def get_output_color_and_label(color1, color2):
            if is_color_in_range(color1, white_lower_bound, white_upper_bound) or is_color_in_range(color2, white_lower_bound, white_upper_bound):
                return np.array([255, 255, 255]), "medium"  # White if either square is white
            if is_color_in_range(color1, red_lower_bound, red_upper_bound) and is_color_in_range(color2, red_lower_bound, red_upper_bound):
                return np.array([0, 0, 255]), "medium"  # Red if both squares are red
            if is_color_in_range(color1, green_lower_bound, green_upper_bound) and is_color_in_range(color2, green_lower_bound, green_upper_bound):
                return np.array([0, 255, 0]), "medium"  # Green if both squares are green
            if is_color_in_range(color1, yellow_lower_bound, yellow_upper_bound) and is_color_in_range(color2, yellow_lower_bound, yellow_upper_bound):
                return np.array([0, 255, 255]), "medium"  # Yellow if both squares are yellow
            if (is_color_in_range(color1, red_lower_bound, red_upper_bound) and is_color_in_range(color2, green_lower_bound, green_upper_bound)) or \
            (is_color_in_range(color2, red_lower_bound, red_upper_bound) and is_color_in_range(color1, green_lower_bound, green_upper_bound)):
                return np.array([0, 0, 255]), "medium"  # Red for red and green
            if (is_color_in_range(color1, red_lower_bound, red_upper_bound) and is_color_in_range(color2, yellow_lower_bound, yellow_upper_bound)) or \
            (is_color_in_range(color2, red_lower_bound, red_upper_bound) and is_color_in_range(color1, yellow_lower_bound, yellow_upper_bound)):
                return np.array([0, 0, 255]), "medium"  # Red for red and yellow
            if (is_color_in_range(color1, yellow_lower_bound, yellow_upper_bound) and is_color_in_range(color2, green_lower_bound, green_upper_bound)) or \
            (is_color_in_range(color2, yellow_lower_bound, yellow_upper_bound) and is_color_in_range(color1, green_lower_bound, green_upper_bound)):
                return np.array([0, 255, 255]), "medium"  # Yellow for yellow and green
            return np.array([255, 255, 255]), "medium"  # Default to white if none match

        # Iterate over each grid square and apply the rules
        for row in range(grid_size):
            for col in range(grid_size):
                # Get the coordinates for the current grid square
                y_start = row * grid_h
                y_end = (row + 1) * grid_h
                x_start = col * grid_w
                x_end = (col + 1) * grid_w

                # Extract the grid square from both images
                square1 = image1_resized[y_start:y_end, x_start:x_end]
                square2 = image2_resized[y_start:y_end, x_start:x_end]

                # Calculate the average color of each square to determine its dominant color
                avg_color1 = np.mean(square1, axis=(0, 1)).astype(int)
                avg_color2 = np.mean(square2, axis=(0, 1)).astype(int)

                # Determine the output color and label based on the rules
                output_color, label = get_output_color_and_label(avg_color1, avg_color2)

                # Set the corresponding grid square in the output image to the determined color
                output_image[y_start:y_end, x_start:x_end] = output_color

                # Draw a border around each grid square
                cv2.rectangle(output_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 2)  # Black border

                # Add label text to the grid square
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 0, 0)  # Black color for text
                thickness = 1
                text = label

                # Get the size of the text box
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                # Calculate the position to center the text
                text_x = x_start + (grid_w - text_width) // 2
                text_y = y_start + (grid_h + text_height) // 2

                cv2.putText(output_image, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    return output_image
combined_image()