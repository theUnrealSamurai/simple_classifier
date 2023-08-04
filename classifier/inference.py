import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import time
import sys



#loading the Tflite converted model here.
interpreter = tf.lite.Interpreter(model_path='./model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def split_image(input_path, output_paths):
    """
    This function take a image and splits it into 3 pices. The split is done
    based on the lenght of the image. and is dynamic.
    Input path is a single string for the file.
    Output path is a list of paths, the lenght should be equal to number of
    splits
    """


    piece_count = len(output_paths)  # Number of output pieces
    image = cv2.imread(input_path)  # Read the input image

    # Check if the image and output paths are valid
    if image is None or len(output_paths) != piece_count:
        print("Invalid image or output paths")
        return

    # Get the dimensions of the image
    image_height, image_width, _ = image.shape

    # Calculate the width and height of each piece
    piece_width = image_width // piece_count
    piece_height = image_height

    # Split the image into horizontal pieces and save each piece
    for i in range(piece_count):
        start_x = i * piece_width
        end_x = (i + 1) * piece_width
        piece = image[:, start_x:end_x]
        cv2.imwrite(output_paths[i], piece)

    print("Image split into {} pieces successfully.".format(piece_count))





def classify_image(img_path):
    """ This function takes one single input image and classifies it into it's
    respective class"""

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    print('loaded img', img)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet.preprocess_input(x)

    interpreter.set_tensor(input_details[0]['index'], x)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    class_names = ['bar', 'bell', 'cherry', 'clown', 'elephant', 'grape', 'z']

    class_indices = np.argmax(output, axis=1)
    class_labels = [class_names[i] for i in class_indices]

    probabilities = np.max(output, axis=1)

    predicted_labels = list(zip(class_labels, probabilities))
    return {"accuracy": predicted_labels[0][1], "class_index": int(class_indices), "class": predicted_labels[0][0]}





def predict(img_path):
    """ This function takes in the test image, Passes it through the split_image
    function and the splits it into 3 pieces.
    once split, it takes each of the image and then it classifies it using the
    classify function.
    Once classified it draws the conculsion if all the given images are same
    or not.
    """

    image = mpimg.imread(img_path)
    plt.imshow(image)
    plt.show()
    output_paths = ['/tmp/split1.png', '/tmp/split2.png', '/tmp/split3.png']
    split_image(img_path, output_paths)
    class_index = []
    for img in output_paths:
        out = classify_image(img)
        class_index.append(out['class_index'])
        print(out)

    if class_index[0] == class_index[1] and class_index[1] == class_index[2]:
        print("\n\n The icons in the row are the same.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
    else:
        a = time.time()
        image_path = sys.argv[1]
        print(predict(image_path))
        print(f"ran in {time.time()-a} seconds")








