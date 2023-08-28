from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# 画像情報が格納されているフォルダへのパスを指定
data_path = './data/train_data'

# 画像の情報、及び機械学習のパラメータ情報の定義
image_width, image_height = 224, 224
batch_size = 32
num_classes = 2 # 二値分類を指定

# ImageDataGeneratorを使用して、画像処理を施し、データをトレーニングする準備をする。
data_generator = ImageDataGenerator(rescale=1.0/255.0)
tg = data_generator.flow_from_directory(data_path,target_size=(image_width, image_height),batch_size=batch_size,class_mode='categorical')

# モデルを構築
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# モデルをコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# モデルをトレーニング
model.fit(tg, epochs=6)

# モデルを物理ファイルに保存
model.save('./model.h5')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'sl_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
