# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

# import the necessary packages
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

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
modelPath = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062'
charlistPath = '/home/rutger/src/loghi-htr-models/republic-2023-01-02-base-generic_new14-2022-12-20-valcer-0.0062.charlist'

def load_model():
    global model
    get_custom_objects().update({"CERMetric": CERMetric})
    get_custom_objects().update({"WERMetric": WERMetric})
    get_custom_objects().update({"CTCLoss": CTCLoss})

    model = keras.models.load_model(modelPath)
    print('model loaded')

def prepare_image(image):
    # if the image mode is not RGB, convert it
    # if image.mode != "RGB":
    #     image = image.convert("RGB")
    # image = DataGeneratorNew.encode_single_sample(image)
    X = []
    Y = []

    image = tf.image.resize(image, [64, 99999], preserve_aspect_ratio=True)
    # image = np.expand_dims(image, -1)
    image_height = image.shape[0]
    image_width = image.shape[1]
    print(image)
    print(image.shape)
    image = image / 255
    # print(X.shape)
    image = tf.image.resize_with_pad(image, image_height, image_width + 50)
    print(image.shape)
    image = 0.5 - image
    print(image.shape)
    image = tf.transpose(image, perm=[1, 0, 2])
    print(image.shape)
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


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    beam_width = 10
    greedy = False
    data = {"success": True}
    # TODO: read charlist from model


    with open(charlistPath) as file:
        char_list = list(char for char in file.read())

    utils = Utils(char_list, True)

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = tf.io.decode_jpeg(image, channels=1)

            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            data["predictions"] = []

            predictions = model.predict(image)
            predicted_texts = decode_batch_predictions(predictions, utils, greedy, beam_width)

            for prediction in predicted_texts:
                for i in range(len(prediction)):
                    confidence = prediction[i][0]
                    predicted_text = prediction[i][1]
                    predicted_text = predicted_text.strip().replace('', '')
                    r = {"label": predicted_text, "probability": float(confidence)}
                    data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(host='0.0.0.0')
