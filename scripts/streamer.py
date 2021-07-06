import requests
import json
from cv2 import cv2
from datetime import datetime, timedelta
import os
import csv
import pytz
from threading import Thread, Event, active_count, enumerate, currentThread
from queue import Queue
from tensorflow.keras.models import load_model
import tensorflow as tf
global predictions_list
predictions_list = [0,0,0,0]

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
    frame.prediction_strength = frame.baby_prediction - frame.empty_prediction

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
    
    frame.prediction_image_en = cv2.imencode('.jpg', frame.prediction_image)[1].tobytes()
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
        with open('./TokensAndResponses/devicelist.json', 'r') as f:
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
        with open('./TokensAndResponses/devicelist.json', 'w') as f:
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
    global current_stream_expiration_time

    with open('./TokensAndResponses/refreshtokens.json', 'r') as f:
        data = json.load(f)
        PROJECT_ID = data['project_id']
        
    device_ID = get_device_ID(PROJECT_ID, './TokensAndResponses/accesstokens.json')

    #Get previous URL, stream token, and extension token
    with open('./TokensAndResponses/streaminfo.json', 'r') as f:
        original_data = json.load(f)
        previous_URL = original_data['results']['streamUrls']['rtspUrl']
        stream_token = original_data['results']['streamToken']
        stream_extension_token = original_data['results']['streamExtensionToken']
        
        #Remove old stream token from url
        base_URL = previous_URL.split(stream_token)[0]
    
    access_token = refresh_access_token('./TokensAndResponses/client_secret.json',
                                        './TokensAndResponses/refreshtokens.json',
                                        './TokensAndResponses/accesstokens.json')     

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
    
    current_stream_expiration_time = pacific_datetime(new_expiry_time_string)
    print(f'Refreshed token expires at {current_stream_expiration_time}')

    #Save to streaminfo.json file
    with open('./TokensAndResponses/streaminfo.json', 'w') as f:
        json.dump(original_data, f)
        
    return new_URL


def pacific_datetime(rfc_string):
    """Converts RFC 3339 string to 'US/Pacific' datetime. Timezone naive."""
    
    gmt_expiration_date = datetime.strptime(rfc_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    old_timezone = pytz.timezone('Etc/GMT+0')
    new_timezone = pytz.timezone('US/Pacific')

    return old_timezone.localize(gmt_expiration_date).astimezone(new_timezone).replace(tzinfo=None)


def get_stream_URL():
    """Returns RTSP URL. Checks if previous stream URL is valid. If not, refreshes access tokens and stream tokens."""
    global current_stream_expiration_time

    #Get previous stream's URL and expiration date
    with open('./TokensAndResponses/streaminfo.json', 'r') as f:
        data = json.load(f)
        try:
            previous_URL = data['results']['streamUrls']['rtspUrl']
            previous_expiration_rfc_string = data['results']['expiresAt'] 
            previous_expiration_localized = pacific_datetime(previous_expiration_rfc_string)
    
            #Previous URL still valid for at least another minute
            if previous_expiration_localized > datetime.now() + timedelta(minutes=1):
                print(f'Previous token still valid, expires at {previous_expiration_localized}')
                current_stream_expiration_time = previous_expiration_localized
                return previous_URL
        except:
            #File may have JSON from earlier bad request
            pass
    #Previous URL expired or about to expire
    #Get new access key, make request for device ID
    new_access_token = refresh_access_token('./TokensAndResponses/client_secret.json',
                                            './TokensAndResponses/refreshtokens.json',
                                            './TokensAndResponses/accesstokens.json')
    
    with open('./TokensAndResponses/refreshtokens.json', 'r') as f:
        data = json.load(f)
        PROJECT_ID = data['project_id']
    
    device_ID = get_device_ID(PROJECT_ID, './TokensAndResponses/accesstokens.json')

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
    with open('./TokensAndResponses/streaminfo.json', 'w') as f:
        json.dump(json_response, f)

    #Get stream URL from the response
    url = json_response['results']['streamUrls']['rtspUrl']
    new_expiry_time_string = json_response['results']['expiresAt']
    current_stream_expiration_time = pacific_datetime(new_expiry_time_string)
    
    print(f'Expires at {current_stream_expiration_time}')
    
    return url

def refresh_stream_checker(expires_in_seconds):
    global current_stream_expiration_time

    if current_stream_expiration_time < datetime.now() + timedelta(seconds= expires_in_seconds) \
        and 'Refresh Thread' not in [thread.name for thread in enumerate()]:
        t = Thread( name= 'Refresh Thread',
                target= refresh_stream_token,
                args= [],
                daemon= True
            )
        t.start()
        print(f'{t.getName()} started')
        print(f'Active threads: {[thread.getName() for thread in enumerate()]}')

class MyFrame:
    """Applies timestamp to videocapture frame, holds image and timestamp metadata"""
    
    def __init__(self, capture_obj):

        self.ret, self.original = capture_obj.read()
        if not self.ret:
            pass
        
        SMALL_SIZE = (240,135)
        MEDIUM_SIZE = (1280, 720)
        SMALL_MED_SIZE = (960, 540)

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

        self.mediumsize = cv2.resize(self.image, SMALL_MED_SIZE)

        #Prep image for prediction
        frame_grey = cv2.cvtColor(self.mediumsize, cv2.COLOR_BGR2GRAY)
        self.smallsize = cv2.resize(frame_grey, SMALL_SIZE)

        self.frame_grey_en = cv2.imencode('.jpg', frame_grey)[1].tobytes()

def training_data_saver(frame, value):

    OUTPUT_DIRECTORY_ORIGINALS = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep_Schedule\\scripts\\static\\Originals'
    OUTPUT_DIRECTORY_RESIZED = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep_Schedule\\scripts\\static\\Resized'
    VALUES = {'0.0': 'Empty', '1.0': 'Awake', '2.0': 'Asleep'}

    if frame.filename not in os.listdir(OUTPUT_DIRECTORY_ORIGINALS):
        #Save Original
        output_path_originals = f'{OUTPUT_DIRECTORY_ORIGINALS}\\{frame.filename}'
        cv2.imwrite(output_path_originals, frame.original)

        #Save resized/greyscale copy
        output_path_resized = f'{OUTPUT_DIRECTORY_RESIZED}\\{frame.filename}'
        cv2.imwrite(output_path_resized, frame.smallsize)
        print(f'{output_path_originals} value {value}')

        #Save record in CSV and return success message
        with open('./data/data.csv', 'a', newline='') as f:
            csv.writer(f).writerow([output_path_originals,output_path_resized,value])
        return f'{frame.filename} saved with value "{VALUES.get(value)}"'
    else: 
        pass

class Streamer:
    """Class for serving stream with prediction overlay"""

    def __init__(self, inqueue= None, outqueue= None, event= None):
        global current_stream_expiration_time

        self.stream_url = get_stream_URL()
        self.model = load_model(get_recent_model())
        self.cap = cv2.VideoCapture(self.stream_url)
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.event = event

    def __iter__(self):
        """Serves stream as generator for either HTTP or blind consumption"""

        #List of values to display on live graph
        global predictions_list
        predictions_list = [0,0,0,0]

        while True:
            #Start refreshed stream if current one only has 30 seconds left
            refresh_stream_checker(expires_in_seconds = 45)
            if self.cap.isOpened():
                self.frame = MyFrame(self.cap)
                if not self.frame.ret:
                    break
            else:
                #Reopen capture object, then try again
                self.cap = cv2.VideoCapture(self.stream_url)
                continue
            self.frame = predictor(self.frame, self.model)

            #Add current prediction to beginning of prediction list
            predictions_list = predictions_list[1:]
            predictions_list.append(self.frame.prediction_strength)


            #Listen for keypress (only used here to even out framerate)
            self.key = cv2.waitKey(25) & 0xFF

            if not self.inqueue.empty():
                response = training_data_saver(frame= self.frame, value= self.inqueue.get())
                if response:
                    self.outqueue.put(response)
                    self.event.set()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + self.frame.prediction_image_en + b'\r\n'
            )

    def run_local_display(self):
        """Serves stream for local display in CV2 window"""

        window_name = 'Monitor'
        while True:
            #Start refreshed stream if current one only has 60 seconds left
            refresh_stream_checker(expires_in_seconds= 60)
            if self.cap.isOpened():
                self.frame = MyFrame(self.cap)
                if not self.frame.ret:
                    break
            else:
                #Reopen capture object, then try again
                self.cap = cv2.VideoCapture(self.stream_url)
                continue
            self.frame = predictor(self.frame, self.model)
            cv2.imshow(window_name, self.frame.prediction_image)

            #Listen for keypress
            self.key = cv2.waitKey(43) & 0xFF

            #save data if key is pressed
            if self.key in [ord('0'), ord('1'), ord('2')]:
                response = training_data_saver(frame= self.frame, value= format(int(chr(self.key)), '.1f'))
                if response:
                    print(response)

            #close window
            if (self.key == ord('q')) or \
                cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cap.release()
        return None

    def get_sample(self, sample_size = 10):
        """Opens stream and returns list of MyFrame objects taken from stream"""
        
        self.samples = []

        while len(self.samples) < sample_size:
            #Start refreshed stream if current one only has 30 seconds left
            refresh_stream_checker(expires_in_seconds = 30)
            if self.cap.isOpened():
                self.frame = MyFrame(self.cap)
                if not self.frame.ret:
                    break
            self.frame = predictor(self.frame, self.model)
            self.samples.append(self.frame)

        self.cap.release()
        return self.samples

if __name__ == '__main__':
    #Display stream in local window
    local_stream = Streamer().run_local_display()