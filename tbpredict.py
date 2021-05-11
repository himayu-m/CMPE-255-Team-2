import sys
import numpy as np
from PIL import Image
import requests
import io
import tensorflow as tf

model_path1 = './denseNet201v2.h5'
model1 = tf.keras.models.load_model(model_path1)

model_path2 = './InceptionV3.h5'
model2 = tf.keras.models.load_model(model_path2)

model_path3 = './VGG19.h5'
model3 = tf.keras.models.load_model(model_path3)

model_path4 = './InceptionResNetV2.h5'
model4 = tf.keras.models.load_model(model_path4)

path = sys.argv[1]

#using images from internet
def preprocess_image_url(image_path, desired_size=224):
    response = requests.get(image_path)
    image_bytes = io.BytesIO(response.content)
    im = Image.open(image_bytes)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    return im

#using system path
def preprocess_image_sys(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    return im

x = np.empty((1, 224, 224, 3), dtype=np.uint8)
if path.startswith('http'):
  x[0, :, :, :] = preprocess_image_url(path)
else:
  x[0, :, :, :] = preprocess_image_sys(path)


y1 = model1.predict(x) > 0.5
y1 = y1.astype(int)
y1 = y1[0][0]

y2 = model2.predict(x) > 0.5
y2 = y2.astype(int)
y2 = y2[0][0]

y3 = model3.predict(x) > 0.5
y3 = y3.astype(int)
y3 = y3[0][0]

y4 = model4.predict(x) > 0.5
y4 = y4.astype(int)
y4 = y4[0][0]

##print(y1)
##print(y2)
##print(y3)
##print(y4)

w = [0.28,0.28,0.18,0.26] 
final = y1*w[0] + y2*w[1] + y3*w[2] + y4*w[3] 

pred = 0
if (final>0.56):
  pred = 1
else:
  pred = 0

if (pred==0):
    print("Normal")
else:
    print("Tuberculosis")



