import coremltools as ct
from tensorflow import keras 

# Kerasのモデルを読み込む
model = keras.models.load_model('./model.h5')

# KerasモデルをCoreMLモデルに変換
mlmodel = ct.convert(model, inputs=[ct.ImageType(scale=1/127, shape=[1,224,224,3])])

# 変換されたモデルを物理ファイルに保存
mlmodel.save("Test.mlmodel")





