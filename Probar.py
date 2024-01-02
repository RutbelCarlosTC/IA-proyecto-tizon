from keras.models import load_model

# Cargar el modelo
model = load_model('potatoes.h5')

import numpy as np
from keras.preprocessing import image

# Cargar el conjunto de datos de entrenamiento para obtener class_names
# Suponiendo que ya tienes train_ds como tu conjunto de datos de entrenamiento
class_names =  ['Potato___Early_blight', 'Potato___Late_blight','Potato___healthy']

# Ruta de la nueva imagen que quieres predecir
nueva_imagen = 'test_imagenes/healthy_6e.jpg' 
# Cargar la imagen y preprocesarla
img = image.load_img(nueva_imagen, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalizar

# Realizar la predicción
predictions = model.predict(img_array)
class_index = np.argmax(predictions[0])
predicted_class = class_names[class_index]
confidence = predictions[0][class_index] * 100

print(f'Predicción: {predicted_class} con una confianza del {confidence:.2f}%')