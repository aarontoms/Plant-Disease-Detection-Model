import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("model/tomato_model.h5")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


img_path = "pics/bacterial-leaf-spot-on-tomato-Rutgers.jpg"
img_array = preprocess_image(img_path)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
confidence = np.max(predictions)
if confidence < 0.7:  # Adjust as needed
    print(f"Unknown image, not in dataset with confidence {confidence}")
else:
    print(f"Predicted class: {predicted_class[0]} with confidence {confidence}")

# print(f"Predicted class: {predicted_class[0]}")

{
    "disease": "Healthy Tomato Leaf",
    "symptoms": "No visible signs of disease; leaves are green, firm, and show no signs of damage or discoloration.",
    "causes": "Proper growing conditions, including adequate sunlight, water, nutrients, and pest/disease control.",
    "longTermSteps": "Maintain healthy soil through regular fertilization and amendment. Practice crop rotation to prevent soilborne diseases. Choose disease-resistant tomato varieties. Ensure proper spacing for good air circulation.",
    "shortTermSteps": "Monitor plants regularly for any signs of disease or pest infestation. Provide adequate watering and fertilization as needed. Prune away any damaged or dead leaves.",
}
