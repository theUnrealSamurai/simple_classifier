import os, random, cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


def get_noisy_image(length, width, mode):
    """
    Generate a noisy image of the specified dimensions and color mode.
    Parameters:
        length (int): The length of the image.
        width (int): The width of the image.
        mode (str): The color mode, either 'RGB' or 'BW' (Black and White).
    Returns:
        Image: The generated noisy image.
    """
    if mode not in ['RGB', 'BW']:
        raise ValueError("Invalid mode. Mode must be 'RGB' or 'BW'.")

    # Generate random pixel values in the range [0, 255] based on the color mode
    if mode == 'RGB':
        pixels = np.random.randint(0, 256, size=(length, width, 3), dtype=np.uint8)
    else:  # mode == 'BW'
        pixels = np.random.randint(0, 256, size=(length, width), dtype=np.uint8)

    return pixels


def get_slot_machine_images(path):
    images = glob(os.path.join(path, '*/*.png'))
    img_path = random.choice(images)
    img = Image.open(img_path)
    img = img.convert("RGB")
    return np.array(img)



def place_image_inside_background(background, foreground):
    # Get the size of the background and foreground images
    bg_height, bg_width, _ = background.shape
    fg_height, fg_width, _ = foreground.shape

    # Randomly choose the position to place the foreground image inside the background image
    x_pos = random.randint(0, bg_width - fg_width)
    y_pos = random.randint(0, bg_height - fg_height)

    # Place the foreground image inside the background image
    background[y_pos:y_pos + fg_height, x_pos:x_pos + fg_width] = foreground

    # Calculate the bounding box coordinates in YOLO format
    x_center = (x_pos + x_pos + fg_width) / 2 / bg_width
    y_center = (y_pos + y_pos + fg_height) / 2 / bg_height
    width = fg_width / bg_width
    height = fg_height / bg_height

    # Return the new image with the bounding box coordinates
    return background, [x_center, y_center, width, height]


def is_overlap(box1, box2):
    print(box1, box2)
    x_center1, y_center1, width1, height1 = box1
    x_center2, y_center2, width2, height2 = box2

    left1 = x_center1 - width1 / 2
    right1 = x_center1 + width1 / 2
    top1 = y_center1 - height1 / 2
    bottom1 = y_center1 + height1 / 2

    left2 = x_center2 - width2 / 2
    right2 = x_center2 + width2 / 2
    top2 = y_center2 - height2 / 2
    bottom2 = y_center2 + height2 / 2

    x_overlap = max(0, min(right1, right2) - max(left1, left2))
    y_overlap = max(0, min(bottom1, bottom2) - max(top1, top2))

    intersection_area = x_overlap * y_overlap

    area1 = width1 * height1
    area2 = width2 * height2

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area

    return iou > 0





def check_overlap(boxes):
    for i, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1

        for j, box2 in enumerate(boxes[i+1:]):
            x2, y2, w2, h2 = box2

            # Convert the bounding box coordinates to (x_min, y_min, x_max, y_max) in pixel values
            img_width, img_height = 1024, 1024
            x_min1 = (x1 - w1 / 2) * img_width
            y_min1 = (y1 - h1 / 2) * img_height
            x_max1 = (x1 + w1 / 2) * img_width
            y_max1 = (y1 + h1 / 2) * img_height

            x_min2 = (x2 - w2 / 2) * img_width
            y_min2 = (y2 - h2 / 2) * img_height
            x_max2 = (x2 + w2 / 2) * img_width
            y_max2 = (y2 + h2 / 2) * img_height

            # Check for overlap
            if x_min1 < x_max2 and x_max1 > x_min2 and y_min1 < y_max2 and y_max1 > y_min2:
                return True

    return False


class DataGenerator(Dataset):
    def __init__(self, n):
        """n is the number of images to generate"""
        self.n = n

    def __len__(self):
        return len(self.n)
    
    def __getitem__(self, index):
        noise = get_noisy_image(1024, 1024, 'RGB')
        labels = []
        for i in range(9):
            slot_image  = get_slot_machine_images('/home/roshan/Code/Mamiya/simple_classifier/object_detection/data/raw_images')
            image, label = place_image_inside_background(noise, slot_image)
            noise = image
            labels.append(label)

        return image, labels
    
dset = DataGenerator(3)

x = dset[0]

print(x[0].shape)
print(x[1])
# print(is_overlap(x[1][0], x[1][1]))
print(check_overlap(x[1]))
plt.imshow(x[0])
plt.show()
