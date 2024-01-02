import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
import numpy as np
import mahotas as mt
import joblib

# Cargar los datos desde el archivo CSV
data = pd.read_csv('SVM/dataset.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data[['mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', 'contrast', 'correlation', 'inverse_difference_moments', 'entropy']]
y = data['type']

# Codificar las etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
# Entrenar el modelo en el conjunto de entrenamiento
svm_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test)

#print ("yepred[0]", y_pred[0])
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

## Guardando modelo:
joblib.dump(svm_model, 'SVM/modelo_svm.pkl')

print("COMPROBANDO CON UNA IMAGEN NUEVA")

def extract_features(main_img):
    #main_img = cv2.imread(imgpath)    
    #preprocesamiento
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
    
    #Caractersticas de color
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]
    blue_channel[blue_channel == 255] = 0
    green_channel[green_channel == 255] = 0
    red_channel[red_channel == 255] = 0
    
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
    
    #Caracteristicas de Textura
    textures = mt.features.haralick(gs)
    ht_mean = textures.mean(axis=0)
    contrast = ht_mean[1]
    correlation = ht_mean[2]
    inverse_diff_moments = ht_mean[4]
    entropy = ht_mean[8]
      
    vector = [red_mean,green_mean,blue_mean,red_std,green_std,blue_std,\
                contrast,correlation,inverse_diff_moments,entropy
                ]
    return vector

img_test = "PlantVillage/Potato___Late_blight/2f419afc-a232-42c7-b0ff-92d928647a0f___RS_LB 4917.JPG"
main_img = cv2.imread(img_test) 
resized_img = cv2.resize(main_img, (255,255))

# Define los nombres de características para tus datos de entrada
feature_names = ['mean_r', 'mean_g', 'mean_b', 'stddev_r', 'stddev_g', 'stddev_b', 'contrast', 'correlation', 'inverse_difference_moments', 'entropy']

# Asigna los nombres de características a tus datos de entrada
feat = extract_features(main_img)
feat_test = pd.DataFrame([feat], columns=feature_names)

pred = svm_model.predict(feat_test)
print(pred)
etiqueta_predicha = label_encoder.inverse_transform(pred)
print("Etiqueta predicha:", etiqueta_predicha)
