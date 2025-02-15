import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = np.max(output)
    return predicted_class, confidence

img_path = "pics/108936_01.jpg"
img_array = preprocess_image(img_path)

class_names = {0: 'Bacterial_spot', 1: 'Early_blight', 2: 'Late_blight', 3: 'Septoria_leaf_spot', 4: 'healthy'}

keras_model = load_model("model/tomato_keras_inception.tflite")
h5_model = load_model("model/tomato_h5_inception.tflite")

pred_keras, conf_keras = predict(keras_model, img_array)
pred_h5, conf_h5 = predict(h5_model, img_array)

if conf_keras < 0.7 or conf_h5 < 0.7:
    print(f"Unknown image. Click again. Keras Confidence: {conf_keras:.2f}, H5 Confidence: {conf_h5:.2f}")
elif conf_keras > conf_h5:
    print(f"Final Prediction (Keras Model): Class {class_names[pred_keras]}, Confidence {conf_keras:.2f}")
else:
    print(f"Final Prediction (H5 Model): Class {class_names[pred_h5]}, Confidence {conf_h5:.2f}")

print(f"\nKeras Model: Class {class_names[pred_keras]}, Confidence {conf_keras:.2f}")
print(f"H5 Model: Class {class_names[pred_h5]}, Confidence {conf_h5:.2f}")


# h5_model = load_model("model/tomato_keras_inception.tflite")
# pred_h5, conf_h5 = predict(h5_model, img_array)
# print(f"Final Prediction (H5 Model): Class {class_names[pred_h5]}, Confidence {conf_h5:.2f}")