import os
import cv2
import pandas as pd
import numpy as np
import mahotas as mt

# Define las carpetas donde se encuentran las imágenes
folders = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]
names = ['mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b', \
             'contrast','correlation','inverse_difference_moments','entropy','type']

# Lista para almacenar los datos de las imágenes
data = pd.DataFrame([], columns=names)

def extract_features(imgpath):
    main_img = cv2.imread(imgpath)    
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

def create_dataset(folder):
    folder_path = "PlantVillage/" + folder
    img_files = os.listdir(folder_path)

    df = pd.DataFrame([], columns=names)
    for file in img_files:
        imgpath = folder_path + "/" + file
        vector = extract_features(imgpath)
        type = folder
        vector.append(type)
        df_temp = pd.DataFrame([vector],columns=names)
        df = pd.concat([df,df_temp],ignore_index=True)
        print(file)
    return df

# Recorre las carpetas y extrae las características de las imágenes
for folder in folders:
    data = pd.concat([data, create_dataset(folder)],ignore_index=True)

# Guarda el DataFrame en un archivo CSV
data.to_csv('SVM/dataset.csv')

