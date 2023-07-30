import os, random, argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


classes = {
    'bar': 0,
    'bell': 1,
    'cherry': 2,
    'clown': 3,
    'elephant': 4,
    'grape': 5,
    'z': 6,
}


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
    return np.array(img), classes[img_path.split('/')[-2]]



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


def check_overlap(boxes):
    for i, box1 in enumerate(boxes):
        class_id, x1, y1, w1, h1 = box1

        for j, box2 in enumerate(boxes[i+1:]):
            class_id, x2, y2, w2, h2 = box2

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
    def __init__(self, n, img_size):
        """n is the number of images to generate"""
        self.n = n
        self.IMG_SIZE = img_size
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        while True:
            noise = get_noisy_image(self.IMG_SIZE, self.IMG_SIZE, 'RGB')
            labels = []
            for i in range(9):
                slot_image, id  = get_slot_machine_images('/home/roshan/Code/Mamiya/simple_classifier/object_detection/data/raw_images')
                image, label = place_image_inside_background(noise, slot_image)
                noise = image
                # labels.append([id].extend(label))
                label.insert(0, id)
                labels.append(label)
            if not check_overlap(labels):
                break

        return image, labels
    

def save_as_rgb_jpeg(image_array, file_path):
    image = Image.fromarray(np.uint8(image_array))
    image.save(file_path)

def write_list_to_txt(my_list, file_path):
    with open(file_path, "w") as file:
        lines = []
        for i in my_list:
            line = " ".join([str(j) for j in i]) + "\n"
            lines.append(line)
        file.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset and save images and labels.")
    parser.add_argument("--base_path", type=str, default="/home/roshan/Code/Mamiya/simple_classifier/object_detection/data/generated", help="Base path to save images and labels")
    parser.add_argument("--img_size", type=int, default=640, help="Image size (both height and width)")
    parser.add_argument("--n", type=int, default=10, help="Number of images to generate")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.base_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.base_path, "labels"), exist_ok=True)

    dset = DataGenerator(n=args.n, img_size=args.img_size)
    for i in tqdm(range(args.n)):
        img, labels = dset[i]
        img_path = os.path.join(args.base_path, f"images/{str(i).zfill(5)}.jpg")
        label_path = os.path.join(args.base_path, f"labels/{str(i).zfill(5)}.txt")
        save_as_rgb_jpeg(img, img_path)
        write_list_to_txt(labels, label_path)
