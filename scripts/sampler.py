#Ensure project folder is added to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).parent))

#Normal imports
from streamer import Streamer
from cv2 import cv2
#need to ensure database imports follow same convention as other scripts
#https://stackoverflow.com/questions/37908767/table-roles-users-is-already-defined-for-this-metadata-instance/49490203
from scripts import datamodels

def reading_image_saver(frame_obj):
    """Saves MyFrame object to readings originals & resized directories"""
    OUTPUT_DIRECTORY_ORIGINALS = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep_Schedule\\scripts\\static\\ReadingImagesOriginals'
    OUTPUT_DIRECTORY_RESIZED = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep_Schedule\\scripts\\static\\ReadingImagesResized'
    

    output_path_originals = f'{OUTPUT_DIRECTORY_ORIGINALS}\\{frame_obj.filename}'
    cv2.imwrite(output_path_originals, frame_obj.original)

    output_path_resized = f'{OUTPUT_DIRECTORY_RESIZED}\\{frame_obj.filename}'
    cv2.imwrite(output_path_resized, frame_obj.smallsize)

    return output_path_originals, output_path_resized

def add_to_database(empty_value, baby_value, orig_path, resized_path):
    """Adds single data point to database"""
    if empty_value > baby_value:
        value = 'Empty'
    else:
        value = 'Baby'

    datapoint = datamodels.DataPoint(
        value= value,
        baby_reading = baby_value,
        empty_reading = empty_value,
        image_orig_path = orig_path,
        image_resized_path = resized_path
    )

    datamodels.commit_item(datapoint)
    return None

def create_sleep_session():
    return None

def main():
    sample_size = 30
    samples = Streamer().get_sample(sample_size)

    ave_empty_values = sum([frame.empty_prediction for frame in samples])/sample_size
    ave_baby_values = sum([frame.baby_prediction for frame in samples])/sample_size

    orig_path, resized_path = reading_image_saver(samples[-1])

    add_to_database(ave_empty_values, ave_baby_values, orig_path, resized_path)

if __name__ == '__main__':
    main()