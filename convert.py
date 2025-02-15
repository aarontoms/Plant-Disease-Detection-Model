import tensorflow as tf

model = tf.keras.models.load_model("model/tomato_inception.keras")  
converter = tf.lite.TFLiteConverter.from_keras_model(model)  
tflite_model = converter.convert()  

with open("model/tomato_keras_inception.tflite", "wb") as f:  
    f.write(tflite_model)
