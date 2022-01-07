from config import *
import metrics
from matplotlib import pyplot as plt
from tf_keras_vis.utils import num_of_gpus
import tensorflow.keras as keras
import tensorflow as tf
import argparse
from keras.utils.generic_utils import get_custom_objects

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', metavar='seed', type=int, default=42,
                    help='random seed to be used')
parser.add_argument('--gpu', metavar='gpu', type=int, default=0,
                    help='gpu to be used')
parser.add_argument('--percent_validation', metavar='percent_validation', type=float, default=0.15,
                    help='percent_validation to be used')
parser.add_argument('--learning_rate', metavar='learning_rate', type=float, default=0.00001,
                    help='learning_rate to be used')
parser.add_argument('--epochs', metavar='epochs', type=int, default=40,
                    help='epochs to be used')
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=1,
                    help='batch_size to be used, when using variable sized input this must be 1')

parser.add_argument('--height', metavar='height', type=int, default=51,
                    help='height to be used')
parser.add_argument('--width', metavar='width', type=int, default=751,
                    help='width to be used')
parser.add_argument('--channels', metavar='channels', type=int, default=3,
                    help='channels to be used')
parser.add_argument('--output', metavar='output', type=str, default='output',
                    help='base output to be used')
parser.add_argument('--trainset', metavar='trainset', type=str, default='/data/cvl-database-1-1/train.txt',
                    help='trainset to be used')
parser.add_argument('--testset', metavar='testset', type=str, default='/data/cvl-database-1-1/test.txt',
                    help='testset to be used')
parser.add_argument('--use_testset', metavar='use_testset', type=bool, default=False,
                    help='testset to be used')
parser.add_argument('--spec', metavar='spec ', type=str, default='Cl11,11,32 Mp3,3 Cl7,7,64 Gm',
                    help='spec')
parser.add_argument('--existing_model', metavar='existing_model ', type=str, default='',
                    help='existing_model')
parser.add_argument('--dataset', metavar='dataset ', type=str, default='ecodices',
                    help='dataset. ecodices or iisg')
parser.add_argument('--do_binarize_sauvola', action='store_true',
                    help='do_binarize_sauvola')

args = parser.parse_args()

MODEL_PATH = "checkpoints/difornet17-saved-model-07-0.82.hdf5"
MODEL_PATH = "checkpoints/difornet13-saved-model-68-0.94.hdf5"
MODEL_PATH = "checkpoints/difornet13-saved-model-49-0.94.hdf5" # iisg
MODEL_PATH = "checkpoints/difornet14-saved-model-45-0.97.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet17-saved-model-44-0.92.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-98-0.97.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-19-0.94.hdf5"
MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-128-0.95.hdf5"
MODEL_PATH = "checkpoints/difornet23-best_val_loss"
MODEL_PATH = "checkpoints/difornet24-best_val_loss"
MODEL_PATH = "checkpoints-iisg/difornetC-saved-model-20-0.93.hdf5"
if args.existing_model:
    MODEL_PATH = args.existing_model

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"accuracy": metrics.accuracy})
get_custom_objects().update({"average": metrics.average})

model = keras.models.load_model(MODEL_PATH)
model.summary()
model = model.get_layer(index=2)
model.summary()
# input_layer = InputLayer(input_shape=(51, 751, 3), name="input_1")
# # model.input = input
# model.layers[0]= input_layer
#
# model = Model(inputs=model.input,
#                                  outputs=model.output)
# model.summary()

def model_modifier(cloned_model):
    cloned_model.layers[-2].activation = tf.keras.activations.linear
    return cloned_model

from tf_keras_vis.activation_maximization import ActivationMaximization

activation_maximization = ActivationMaximization(model,
                                                 model_modifier,
                                                 clone=False)

from tf_keras_vis.utils.scores import CategoricalScore


# Instead of CategoricalScore object, you can define the scratch function such as below:
def score_function(output):
    # The `output` variable refer to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    return output[:, 20]

from tf_keras_vis.activation_maximization.callbacks import PrintLogger
seed_input = tf.random.uniform((1, 51, 201, 3),0,255,dtype=tf.dtypes.float32)


for i in range(96):
    score = CategoricalScore(i)

    activations = activation_maximization(score,
                                          steps=1024,
                                          callbacks=[PrintLogger(interval=50)],
                                          seed_input=seed_input)

    # Render
    f, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(activations[0])
    ax.set_title(i, fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('dense/dense-{}.png'.format(i))
