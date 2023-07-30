import cv2

def display_image_with_bboxes(image_path, bbox_file_path):
    # Read the image
    image = cv2.imread(image_path)

    # Read the bounding box coordinates from the text file
    with open(bbox_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        class_index, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert YOLO-format coordinates to pixel coordinates
        image_height, image_width, _ = image.shape
        x_center = int(x_center * image_width)
        y_center = int(y_center * image_height)
        box_width = int(width * image_width)
        box_height = int(height * image_height)

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = x_center - box_width // 2
        y1 = y_center - box_height // 2
        x2 = x_center + box_width // 2
        y2 = y_center + box_height // 2

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green color
        thickness = 2
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = '/home/roshan/Code/Mamiya/simple_classifier/object_detection/data/generated/images/00000.jpg'
bbox_file_path = '/home/roshan/Code/Mamiya/simple_classifier/object_detection/data/generated/labels/00000.txt'
display_image_with_bboxes(image_path, bbox_file_path)
