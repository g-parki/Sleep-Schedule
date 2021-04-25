import pandas as pd
import main

import cv2

import os   

with open('TokensAndResponses/devicelist.json', 'r') as f:
    print('hi')

df = pd.read_csv('data.csv')

#Get sample dataframes of each value type
unique_values = df['Value'].unique().tolist()
min_available = df['Value'].value_counts().min()
print(df['Value'].value_counts())

sampledDFs = [df.loc[df['Value'] == value].sample(n= min_available)
                for value in unique_values]

#Combine and scramble
combinedDF = pd.concat(sampledDFs)
combinedDF = combinedDF.sample(frac=1).reset_index(drop=True)
'''
import numpy as np
import sleepmultithread
import os
import cv2
from tensorflow.keras.models import load_model

model_name = sleepmultithread.get_recent_model()
model = load_model(model_name)


picture = os.listdir('Training Data/Resized')[0]
picture_path = f'Training Data//Resized//{picture}'

image = cv2.imread(picture_path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('window', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

array = image/255


array = array.reshape(1, 135, 240, 1)

prediction = model(array, training= False)
print(prediction)

#Show sample

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sample_size = 50
sample_to_display = combinedDF.sample(n=sample_size)
print(sample_to_display)


plt.figure(figsize=(25,12))
for i in range(sample_size):
    path = sample_to_display.iloc[i]['ResizedPath']
    value = sample_to_display.iloc[i]['Value']

    image = mpimg.imread(path)
    plt.subplot(5,10,i+1)
    plt.imshow(image)
    plt.title(str(value))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    
plt.show()
'''
