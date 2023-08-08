import coremltools as ct
from tensorflow import keras 
model = keras.models.load_model('./sl_model.h5')
print(model)

mlmodel = ct.convert(model, inputs=[ct.ImageType(scale=1/127, shape=[1,224,224,3])])
print(mlmodel)

mlmodel.save("Test.mlmodel")