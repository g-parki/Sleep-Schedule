import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    all_lines = [x for x in reader]
    tail = all_lines[-50:]

    plt.figure(figsize=(25,12))
    for i, (original_img_path, resized_img_path, value) in enumerate(tail):
        image = mpimg.imread(resized_img_path)
        plt.subplot(5,10,i+1)
        plt.imshow(image, cmap = 'gray')
        plt.title(value)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    
    plt.show() 
    
               