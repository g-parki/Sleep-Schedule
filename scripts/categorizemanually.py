import pandas as pd
from cv2 import cv2
import csv
import generateHTML

df = pd.read_csv('tobeprocessed.csv')

#Loop through images and categorize
# 0 means empty
# 1 means awake
# 2 means asleep
for i in range(len(df.index)):
    path = df.iloc[i]['FilePath']

    img = cv2.imread(path)

    cv2.putText(img = img,
                text = 'CATEGORIZE THIS IMAGE',
                org = (0, 100),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color = (255, 255, 255),
                thickness = 2)

    cv2.imshow('Categorize This Image', img)

    keypress = cv2.waitKey(0) & 0xFF

    if keypress == ord('0'):
        df.at[i, 'Value'] = 0
        print('Empty')
    elif keypress == ord('1'):
        df.at[i, 'Value'] = 1
        print('Awake')
        
    elif keypress == ord('2'):
        df.at[i, 'Value'] = 2
        print('Asleep')
    
    cv2.destroyAllWindows()

#Append new data to data csv
with open('data.csv', 'a', newline='') as f:
    df.to_csv(f, index=False, header=False)

#Clear tobeprocessed csv
with open('tobeprocessed.csv', "w+", newline='') as f:
    csv.writer(f).writerow(['FilePath','ResizedPath','Value'])

generateHTML.updateHTML()