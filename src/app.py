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
from keras.utils.generic_utils import get_custom_objects
from DataGeneratorNew import DataGeneratorNew
from utils import Utils
from utils import decode_batch_predictions
from queue import Queue
from AppLocker import AppLocker

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
modelPath = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062'
charlistPath = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062.charlist'
beam_width = 10
greedy = False
line_queue = Queue(256)
app_locker = AppLocker()
batch_size = 32

def load_model():
    global model
    get_custom_objects().update({"CERMetric": CERMetric})
    get_custom_objects().update({"WERMetric": WERMetric})
    get_custom_objects().update({"CTCLoss": CTCLoss})

    model = keras.models.load_model(modelPath)

    with open(charlistPath) as file:
        char_list = list(char for char in file.read())
    AppLocker.utils = Utils(char_list, True)

    print('model loaded')


def prepare_image(image):
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    # image = DataGeneratorNew.encode_single_sample(image)
    X = []
    Y = []

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
        image = tf.image.resize_with_pad(image, image_height, image_width + 50)
    # print(image.shape)
    image = 0.5 - image
    # print(image.shape)
    with app_locker._lock3:
        image = tf.transpose(image, perm=[1, 0, 2])
    # print(image.shape)
    X.append(image)
    # X.append(image)
    Y.append(0)

    # # resize the input image and preprocess it
    # image = image.resize(target)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    # return the processed image
    return X


def process(data, line_queue):
    # classify the input image and then initialize the list
    # of predictions to return to the client
    batch = []
    if line_queue.qsize() >= batch_size:
        print('line_queue.qsize() >= batch_size:')
        print(app_locker.get_processing())
        if not app_locker.get_processing():
            with app_locker._lock2:
                app_locker.set_processing(True)
            for i in range(line_queue.qsize()):
                image = line_queue.get()
                batch.append(image[0])
                print('adding image to batch: ' + str(i))
                if i >= batch_size:
                    break
        else:
            return data
    else:
        return data

    data["predictions"] = []

    max_width = 0
    for image in batch:
        if image.shape[0] > max_width:
            max_width = image.shape[0]
    print('max_width: ' + str(max_width))
    with app_locker._lock3:
        for i in range(len(batch)):
            batch[i] = tf.image.resize_with_pad(batch[i], max_width, 64)

    batch = tf.convert_to_tensor(batch)
    predictions = model.predict(batch)
    predicted_texts = decode_batch_predictions(predictions, AppLocker.utils, greedy, beam_width)
    print('processing')
    for prediction in predicted_texts:
        for i in range(len(prediction)):
            confidence = prediction[i][0]
            predicted_text = prediction[i][1]
            predicted_text = predicted_text.strip().replace('', '')
            r = {"label": predicted_text, "probability": float(confidence)}
            data["predictions"].append(r)
            print(predicted_text)
    if app_locker.get_processing:
        with app_locker._lock2:
            app_locker.set_processing(False)

    return data


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": True}
    # TODO: read charlist from model

    print('current queue size: ' + str(line_queue.qsize()))
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = tf.io.decode_jpeg(image, channels=1)

            # preprocess the image and prepare it for classification
            print(image.shape)
            while line_queue.qsize() > batch_size:
                print('processing ' + str(line_queue.qsize()))
                data = process(data, line_queue)

            print('locking ' + str(line_queue.qsize()))
            image = prepare_image(image)
            line_queue.put(image)


            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0')
