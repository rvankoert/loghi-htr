# Imports

# > Standard Library
import metrics
# Add the above directory to the path
sys.path.append('..')

# > Local dependencies
from vis_arg_parser import get_args

# > Third party libraries
import tensorflow.keras as keras
import tensorflow as tf
from matplotlib import pyplot as plt
from tf_keras_vis.utils import num_of_gpus
from keras.utils.generic_utils import get_custom_objects

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = '1'

_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

args = get_args()

if args.existing_model:
    if not os.path.exists(args.existing_model):
        print('cannot find existing model on disk: ' + args.existing_model)
        exit(1)
    MODEL_PATH = args.existing_model

# MODEL_PATH = "checkpoints/difornet17-saved-model-07-0.82.hdf5"
# MODEL_PATH = "checkpoints/difornet13-saved-model-68-0.94.hdf5"
# MODEL_PATH = "checkpoints/difornet13-saved-model-49-0.94.hdf5" # iisg
# MODEL_PATH = "checkpoints/difornet14-saved-model-45-0.97.hdf5"
# # MODEL_PATH = "checkpoints-iisg/difornet17-saved-model-44-0.92.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-19-0.94.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet14-saved-model-98-0.97.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-19-0.94.hdf5"
# MODEL_PATH = "checkpoints-iisg/difornet19-saved-model-128-0.95.hdf5"
# MODEL_PATH = "checkpoints/difornet23-best_val_loss"
# MODEL_PATH = "checkpoints/difornet24-best_val_loss"
# MODEL_PATH = "checkpoints-iisg/difornetC-saved-model-20-0.93.hdf5"

get_custom_objects().update({"contrastive_loss": metrics.contrastive_loss})
get_custom_objects().update({"accuracy": metrics.accuracy})
get_custom_objects().update({"average": metrics.average})

model = keras.models.load_model(MODEL_PATH)
model.summary()
model = model.get_layer(index=2)
model.summary()

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
