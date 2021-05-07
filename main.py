import requests
import json
from cv2 import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import time
import csv
import pytz
from threading import Thread, Event, active_count, enumerate, currentThread
from queue import Queue
import generateHTML
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import shareglobals

# Restrict TensorFlow to only use the first GPU
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass 


def get_recent_model():
    """Returns path of highest named model in 'models' directory"""

    current_dir = os.curdir
    model_name = max(os.listdir('models'))
    model_path = os.path.join(current_dir, 'models', model_name)
    print(f'Model at {model_path}')
    return model_path


def predictor(frame, model):
    """Runs prediction of MyFrame.smallsize, returns MyFrame object with prediction attributes"""
    
    array = frame.smallsize
    array = array/255
    array = array.reshape(1, 135, 240, 1)

    #Make prediction
    #prediction_list = model.predict(array).tolist()[0]
    prediction_list = model(array, training= False).numpy().tolist()[0]
    frame.empty_prediction = prediction_list[0]
    frame.baby_prediction = prediction_list[1]

    if frame.empty_prediction > frame.baby_prediction:
        frame.prediction_str = f'PREDICTION: EMPTY BED ({frame.empty_prediction: .6f}, '\
                                f'{frame.baby_prediction: .6f})'
    else:
        frame.prediction_str = f'PREDICTION: BABY IN BED ({frame.empty_prediction: .6f}, '\
                                f'{frame.baby_prediction: .6f})'
    
    frame.prediction_image = cv2.putText(img = frame.mediumsize,
                                        text = frame.prediction_str,
                                        org = (4, 45),
                                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale = .6,
                                        color = (255, 255, 255),
                                        thickness = 1)
    return frame


def refresh_access_token(client_secret_file, refreshtokens_file, accessstokens_file):
    """Fetches new access token and saves to accesstokens file"""

    with open(client_secret_file) as f:
        data = json.load(f)['web']
        client = data['client_id']
        secret = data['client_secret']

    with open(refreshtokens_file) as f:
        data = json.load(f)
        refresh_token = data['refresh_token']
    print("Getting access token")
    response = requests.post(f'https://www.googleapis.com/oauth2/v4/token'
                            f'?client_id={client}&client_secret={secret}'
                            f'&refresh_token={refresh_token}&grant_type=refresh_token')
    json_response = json.loads(response.content)

    #Save into accesstokens file
    with open(accessstokens_file, 'w') as f:
        json.dump(json_response, f)
    
    return json_response['access_token']


def get_device_ID(projectID, accesstokens_file):
    """Returns first device ID authorized on the device list"""

    #Try getting from local file
    try:
        with open('TokensAndResponses/devicelist.json', 'r') as f:
            device_ID = json.load(f)['devices'][0]['name'].split('/')[3]
    except:
        #Send request
        with open(accesstokens_file, 'r') as f:
            access_token = json.load(f)['access_token']
        
        _headers = {'Content-Type':'application/json',
                    'Authorization': f'Bearer {access_token}'}
        print('Getting Device ID')
        device_response = requests.get('https://smartdevicemanagement.googleapis.com'
                                        f'/v1/enterprises/{projectID}/devices',
                                        headers = _headers)
        json_response = json.loads(device_response.content)

        #Save response to file
        with open('TokensAndResponses/devicelist.json', 'w') as f:
            json.dump(json_response, f)

        #Get device ID from the name in the response
        device_ID = json_response['devices'][0]['name'].split('/')[3]

    return device_ID

def create_tfData(frame_array, data_options):
    """Receives array and turns into tf data object"""
    data = tf.data.Dataset.from_tensor_slices(frame_array)
    data = data.with_options(data_options)
    data = data.batch(1)
    return data

def refresh_stream_token():
    """Returns new stream URL by using stream extension token. Updates streaminfo file"""

    with open('TokensAndResponses/refreshtokens.json', 'r') as f:
        data = json.load(f)
        PROJECT_ID = data['project_id']
        
    device_ID = get_device_ID(PROJECT_ID, 'TokensAndResponses/accesstokens.json')

    #Get previous URL, stream token, and extension token
    with open('TokensAndResponses/streaminfo.json', 'r') as f:
        original_data = json.load(f)
        previous_URL = original_data['results']['streamUrls']['rtspUrl']
        stream_token = original_data['results']['streamToken']
        stream_extension_token = original_data['results']['streamExtensionToken']
        
        #Remove old stream token from url
        base_URL = previous_URL.split(stream_token)[0]
    
    access_token = refresh_access_token('TokensAndResponses/client_secret.json',
                                        'TokensAndResponses/refreshtokens.json',
                                        'TokensAndResponses/accesstokens.json')     

    #Build request for stream extension token
    _headers = {'Content-Type':'application/json', 'Authorization': f'Bearer {access_token}'}
    _data = {"command":"sdm.devices.commands.CameraLiveStream.ExtendRtspStream",
            "params": {"streamExtensionToken": stream_extension_token}}
    response = requests.post('https://smartdevicemanagement.googleapis.com/v1/enterprises/'
                            f'{PROJECT_ID}/devices/'
                            f'{device_ID}:executeCommand',
                            headers = _headers,
                            json = _data)
    json_response = json.loads(response.content)

    #Add new stream token to the end of the old URL
    new_URL = base_URL + json_response['results']['streamToken']

    #Update values originally from streaminfo.json
    original_data['results']['streamUrls']['rtspUrl'] = new_URL
    original_data['results']['streamToken'] = json_response['results']['streamToken']
    original_data['results']['streamExtensionToken'] = json_response['results']['streamExtensionToken']
    original_data['results']['expiresAt'] = json_response['results']['expiresAt']
    
    new_expiry_time_string = json_response['results']['expiresAt']
    
    shareglobals.current_stream_expiration_time = pacific_datetime(new_expiry_time_string)
    print(f'Refreshed token expires at {shareglobals.current_stream_expiration_time}')

    #Save to streaminfo.json file
    with open('TokensAndResponses/streaminfo.json', 'w') as f:
        json.dump(original_data, f)
        
    return new_URL


def pacific_datetime(rfc_string):
    """Converts RFC 3339 string to 'US/Pacific' datetime. Timezone naive."""
    
    gmt_expiration_date = datetime.strptime(rfc_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    old_timezone = pytz.timezone('Etc/GMT+0')
    new_timezone = pytz.timezone('US/Pacific')

    return old_timezone.localize(gmt_expiration_date).astimezone(new_timezone).replace(tzinfo=None)


def start_stream():
    """Returns RTSP URL. Checks if previous stream URL is valid. If not, refreshes access tokens and stream tokens."""
    

    #Get previous stream's URL and expiration date
    with open('TokensAndResponses/streaminfo.json', 'r') as f:
        data = json.load(f)
        try:
            previous_URL = data['results']['streamUrls']['rtspUrl']
            previous_expiration_rfc_string = data['results']['expiresAt'] 
            previous_expiration_localized = pacific_datetime(previous_expiration_rfc_string)
    
            #Previous URL still valid for at least another minute
            if previous_expiration_localized > datetime.now() + timedelta(minutes=1):
                print(f'Previous token still valid, expires at {previous_expiration_localized}')
                shareglobals.current_stream_expiration_time = previous_expiration_localized
                return previous_URL
        except:
            #File may have JSON from earlier bad request
            pass
    #Previous URL expired or about to expire
    #Get new access key, make request for device ID
    new_access_token = refresh_access_token('TokensAndResponses/client_secret.json',
                                            'TokensAndResponses/refreshtokens.json',
                                            'TokensAndResponses/accesstokens.json')
    
    with open('TokensAndResponses/refreshtokens.json', 'r') as f:
        data = json.load(f)
        PROJECT_ID = data['project_id']
    
    device_ID = get_device_ID(PROJECT_ID, 'TokensAndResponses/accesstokens.json')

    #Build request to start RTSP stream
    _headers = {'Content-Type':'application/json', 'Authorization': f'Bearer {new_access_token}'}
    _data = {'command':'sdm.devices.commands.CameraLiveStream.GenerateRtspStream'}
    print('Requesting stream URL')
    response = requests.post('https://smartdevicemanagement.googleapis.com/v1/enterprises/'
                            f'{PROJECT_ID}/devices/'
                            f'{device_ID}:executeCommand',
                            headers = _headers,
                            json = _data)
                    
    json_response = json.loads(response.content)

    #Save stream URL response to local file
    with open('TokensAndResponses/streaminfo.json', 'w') as f:
        json.dump(json_response, f)

    #Get stream URL from the response
    print(json_response)
    url = json_response['results']['streamUrls']['rtspUrl']
    new_expiry_time_string = json_response['results']['expiresAt']
    shareglobals.current_stream_expiration_time = pacific_datetime(new_expiry_time_string)
    
    print(f'Expires at {shareglobals.current_stream_expiration_time}')
    
    return url


class MyFrame:
    """Applies timestamp to videocapture frame, holds image and timestamp metadata"""
    
    def __init__(self, capture_obj):

        self.ret, self.original = capture_obj.read()
        if not self.ret:
            pass
        
        SMALL_SIZE = (240,135)
        MEDIUM_SIZE = (1280, 720)

        #build timestamp string
        self.timestamp_for_video = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.timestamp_for_file = datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = f'{self.timestamp_for_file}.png'
        
        #put timestamp in top left corner of frame
        self.image = cv2.putText(img = self.original,
                                text = self.timestamp_for_video,
                                org = (0, 28),
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 1,
                                color = (255, 255, 255),
                                thickness = 2)

        self.mediumsize = cv2.resize(self.image, MEDIUM_SIZE)

        #Prep image for prediction
        frame_grey = cv2.cvtColor(self.mediumsize, cv2.COLOR_BGR2GRAY)
        self.smallsize = cv2.resize(frame_grey, SMALL_SIZE)

        
def stream_reader(framequeue, endstream_event, response_event, streamgone_event, URL= None):
    """Opens RTSP stream and loads frame into queue"""

    #Initial stream reader will have URL provided. Otherwise, refresh stream and close other threads.
    if URL:
        ('Starting stream with URL provided')
        cap = cv2.VideoCapture(URL)
    else:
        new_URL = refresh_stream_token()
        print('Starting refreshed_stream')
        try:
            cap = cv2.VideoCapture(new_URL)
            print('Nothing wrong with URL')
        except:
            streamgone_event.set()

        #Set event to end other streams saying "i'm the captain now"
        endstream_event.set()
        print('Endstream event set, waiting for response from prior stream.')

        #Wait for other stream to kill themself
        response_event.wait()
        print('Response event set')

        #Clear events
        endstream_event.clear()
        response_event.clear()

    while True:
        if cap.isOpened():
            if not endstream_event.isSet():
                frame = MyFrame(cap)
                if not frame.ret:
                    streamgone_event.set()
                    print(f'Ret gone from {currentThread().getName()}. Streamgone event set')
                    break
                framequeue.put(frame)
            else:
                break
        else:
            streamgone_event.set()
            print(f'Cap not opened in thread {currentThread().getName()}. Streamgone event set')
            break

    response_event.set()
    cap.release()
    print(f'{currentThread().getName()} closed')


if __name__ == '__main__':

    
    stream_url = start_stream()
    frame_q = Queue()
    predicted_q = Queue()
    main_end_event = Event()
    response_event = Event()
    stream_gone_event = Event()
    OUTPUT_DIRECTORY_ORIGINALS = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep Schedule\\static\\Originals'
    OUTPUT_DIRECTORY_RESIZED = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep Schedule\\static\\Resized'
    picture_to_review = False
    rapid_capture = True
    auto_capture = False
    latch = False
    prev_empt_pred = None
    prev_baby_pred = None
    model = load_model(get_recent_model())

    if rapid_capture:
        print('Rapid capture is set')


    initial_reader_thread = Thread(target= stream_reader,
                                    args= [frame_q, main_end_event, response_event, stream_gone_event, stream_url],
                                    daemon= True
                                    )
    initial_reader_thread.start()

    window_name = 'Monitor'
    while True:
        #Exit loop if stream reader threads can't read frame
        if stream_gone_event.isSet():
            break

        #Start refreshed stream if current one only has 30 seconds left
        if shareglobals.current_stream_expiration_time < datetime.now() + timedelta(seconds=30) \
            and active_count() == 2:
            t = Thread(target= refresh_stream_token,
                    args= [],
                    daemon= True)
            t.start()
            print(f'{t.getName()} started')
            print(f'Active threads: {[thread.getName() for thread in enumerate()]}')
        
        #Check if reader thread has placed frame in queue
        if not frame_q.empty():
            frame = frame_q.get()
            frame = predictor(frame, model)

            cv2.imshow(window_name, frame.prediction_image)
           
            #Listen for keypress
            key = cv2.waitKey(43) & 0xFF
            
            #Take snapshot every secs seconds and assign it a certain value
            if rapid_capture:
                secs = 15 
                if datetime.now().second %secs != 0:
                    latch = False
                if not latch and datetime.now().second %secs == 0:
                    latch = True
                    print(f'Waiting for prediction: {frame_q.qsize()}')
                    #key = ord('1') #Save snapshot with this value

            #Automatically save frames where predictions are weak, or where the prediction changes
            if auto_capture:  
                if prev_baby_pred or prev_empt_pred:
                    if frame.empty_prediction < .6 and frame.baby_prediction < .6:
                        key = ord('s')
                        print(f'Weak prediction: ({frame.empty_prediction: .3f},{frame.baby_prediction: .3f})')
                        
                    if (
                        (prev_baby_pred > prev_empt_pred) and (frame.baby_prediction < frame.empty_prediction)
                        or (prev_baby_pred < prev_empt_pred) and (frame.baby_prediction > frame.empty_prediction)
                    ):
                        key = ord('s')
                        print('Prediction is different than previous frame')

            prev_baby_pred = frame.baby_prediction
            prev_empt_pred = frame.empty_prediction  

            if key in [ord('s'), ord('0'), ord('1'), ord('2')] and \
                frame.filename not in os.listdir(OUTPUT_DIRECTORY_ORIGINALS):

                #Save Original
                output_path_originals = f'{OUTPUT_DIRECTORY_ORIGINALS}\\{frame.filename}'
                cv2.imwrite(output_path_originals, frame.original)
                
                #Save resized/greyscale copy
                output_path_resized = f'{OUTPUT_DIRECTORY_RESIZED}\\{frame.filename}'
                cv2.imwrite(output_path_resized, frame.smallsize)

                #Put new record in tobeprocessed CSV if 's' was pressed. Otherwise add to data csv.
                if key == ord('s'):
                    picture_to_review = True
                    print(output_path_originals)
                    with open('tobeprocessed.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([output_path_originals,output_path_resized,''])
                else: #Key was 0, 1, or 2
                    value = format(int(chr(key)), '.1f')
                    print(f'{output_path_originals} value {value}')
                    with open('data.csv', 'a', newline='') as f:
                        csv.writer(f).writerow([output_path_originals,output_path_resized,value])
                
                generateHTML.updateHTML()

            #Close stream if q is pressed or window is closed
            if (key == ord('q')) or \
                cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                main_end_event.set()
                break
    
    main_end_event.set()
    
    print('End of script')

    if picture_to_review:
        import categorizemanually
    
    