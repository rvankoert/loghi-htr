# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
import os
from keras.applications import ResNet50
from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
import keras
from Model import Model, CERMetric, WERMetric, CTCLoss
from tensorflow.keras.utils import get_custom_objects
from DataGeneratorNew import DataGeneratorNew
from utils import Utils
from utils import decode_batch_predictions
from queue import Queue
from AppLocker import AppLocker
import time
from threading import Thread

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
# def app(environ, start_response):
#     data = b"Hello, World!\n"
#     start_response("200 OK", [
#         ("Content-Type", "text/plain"),
#         ("Content-Length", str(len(data)))
#     ])
#     return iter([data])

#app = app()

model = None
modelPath = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062'
charlist_path = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062/charlist.txt'
beam_width = 10
greedy = True
app_locker = AppLocker()
batch_size = 64
COUNT = 0
output_path = '/tmp/output/loghi-htr'

def increment():
    global COUNT
    COUNT = COUNT+1

def load_model():
    global model
    get_custom_objects().update({"CERMetric": CERMetric})
    get_custom_objects().update({"WERMetric": WERMetric})
    get_custom_objects().update({"CTCLoss": CTCLoss})

    model = keras.models.load_model(modelPath)

    with open(charlist_path) as file:
        char_list = list(char for char in file.read())
    AppLocker.utils = Utils(char_list, True)

    print('model loaded')


def prepare_image(identifier, image):
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    # image = DataGeneratorNew.encode_single_sample(image)
    X = []

    print(identifier)
    print(image.shape)
    with app_locker._lock3:
        image = tf.image.resize(image, [64, 99999], preserve_aspect_ratio=True)
    # image = np.expand_dims(image, -1)
    image_height = image.shape[0]
    image_width = image.shape[1]
    # print(image)
    # print(image.shape)
    image = image / 255
    # print(X.shape)
    with app_locker._lock3:
        print(image.shape)
        image = tf.image.resize_with_pad(image, image_height, image_width + 50)
    # print(image.shape)
    image = 0.5 - image
    # print(image.shape)
    with app_locker._lock3:
        image = tf.transpose(image, perm=[1, 0, 2])
    # print(image.shape)
    X.append(image)
    X = tf.convert_to_tensor(X)

    # return the processed image
    return X


def process(line_queue):
    if not app_locker.get_processing() and line_queue.qsize() > 0:
        with app_locker._lock2:
            app_locker.set_processing(True)
            batch = []
            batchIds = []
            for i in range(line_queue.qsize()):
                group_id, identifier, image = line_queue.get()
                batch.append(image[0])
                batchIds.append((group_id, identifier))
                # print('adding image to batch: ' + str(i))
                if i >= batch_size:
                    break
    else:
        return

    max_width = 0
    for image in batch:
        if image.shape[0] > max_width:
            max_width = image.shape[0]
    # print('max_width: ' + str(max_width))
    with app_locker._lock3:
        for i in range(len(batch)):
            batch[i] = tf.image.resize_with_pad(batch[i], max_width, 64)

    with app_locker._lock3:
        batch = tf.convert_to_tensor(batch)
    predictions = model.predict_on_batch(batch)
    with app_locker._lock3:
        predicted_texts = decode_batch_predictions(predictions, AppLocker.utils, greedy, beam_width)
    # print('processing')
    text = ""
    for prediction in predicted_texts:
        for i in range(len(prediction)):
            confidence = prediction[i][0]
            predicted_text = prediction[i][1]
            predicted_text = predicted_text.strip().replace('', '')
            r = {"label": predicted_text, "probability": float(confidence)}
            # data["predictions"].append(r)
            # print(predicted_text)
            # print(batchIds[i] + "\t" + str(confidence) + "\t" + predicted_text)
            group_id = batchIds[i][0]
            identifier = batchIds[i][1]
            confidence = utils.normalize_confidence(confidence, predicted_text)

            text = identifier + "\t" + str(confidence) + "\t" + predicted_text + "\n"
            output_dir = os.path.join(output_path, group_id)
            if not os.path.exists(output_dir):
                print("creating output dir: " + output_dir)
                os.makedirs(output_dir)
                print("created dir: " + output_dir)
            with open(os.path.join(output_path, group_id, identifier+'.txt'), "w") as text_file:
                print(text)
                text_file.write(text)
                text_file.flush()
                increment()

    print('total lines processed: ' + str(COUNT))
    if app_locker.get_processing:
        with app_locker._lock2:
            app_locker.set_processing(False)

    # return data


def continuous_process():
    global counter

    while True:
        if line_queue.qsize() > batch_size:
            # print('processing ' + str(line_queue.qsize()))
            process(line_queue)
        elif line_queue.qsize() > 0:
            time.sleep(3)
            # print('processing ' + str(line_queue.qsize()))
            process(line_queue)
        else:
            # print('sleeping')
            time.sleep(1)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": True}
    # TODO: read charlist from model

    # print('current queue size: ' + str(line_queue.qsize()))
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        request = flask.request
        group_id = request.form["group_id"]
        identifier = request.form["identifier"]

        if identifier and request.files.get("image"):
            # if request.files.get("image"):
            image = request.files["image"].read()
            image = tf.io.decode_jpeg(image)
            
            # preprocess the image and prepare it for classification
            # print(image.shape)
            # while line_queue.qsize() > batch_size:
            #     print('processing ' + str(line_queue.qsize()))
            #     data = process(data, line_queue)

            image = prepare_image(identifier, image)
            result = line_queue.put((group_id, identifier, image))


            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def get_environment_var(name, default):
    if name in os.environ:
        return os.environ[name]
    else:
        return default

def read_environment():
    global modelPath
    modelPath = get_environment_var("LOGHI_MODEL_PATH", modelPath)
    global charlist_path
    charlist_path = get_environment_var("LOGHI_CHARLIST_PATH", charlist_path)
    global beam_width
    beam_width = int(get_environment_var("LOGHI_BEAMWIDTH", beam_width))
    global batch_size
    batch_size = int(get_environment_var("LOGHI_BATCHSIZE", batch_size))
    global output_path
    output_path = get_environment_var("LOGHI_OUTPUT_PATH", output_path)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))
read_environment()
load_model()
global line_queue
line_queue = Queue(256)
daemon = Thread(target=continuous_process, daemon=True, name='Monitor')
daemon.start()

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":

    # continuous_process(line_queue)
    # pool.apply_async(continuous_process, args=[line_queue])
    app.run(host='0.0.0.0')
