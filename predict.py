import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("model/cassava_inception.h5")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


img_path = "pics/gomen.jpeg"
img_array = preprocess_image(img_path)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
confidence = np.max(predictions)
if confidence < 0.7:
    print(f"Unknown image, not in dataset with confidence {confidence}")
else:
    print(f"Predicted class: {predicted_class[0]} with confidence {confidence}")

# print(f"Predicted class: {predicted_class[0]}")
