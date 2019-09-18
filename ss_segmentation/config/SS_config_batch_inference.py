# The name of the folder which contains model files
MODEL_FOLDER = "test_network"

ARTIFACT_DETECTION_DIR = "/home/shreyas/ros/subt_ws/src/artifact_detection/"

# Number of channels for input image (default:3)
NUM_CHANNELS = 3

# Number of classes used during training
NUM_CLASSES = 6

# Resolution of input image
IMG_HEIGHT = 360
IMG_WIDTH = 640

# Directory of model files
ARGS_LOAD_DIR = ARTIFACT_DETECTION_DIR + "utilities/network_verify/" + MODEL_FOLDER
ARGS_LOAD_WEIGHTS = "/model_best.pth"
ARGS_LOAD_MODEL = "/SS_network_design.py"

# Directory location to save inference visualizations
ARGS_SAVE_DIR = ARTIFACT_DETECTION_DIR + "/utilities/network_verify/"

# Directory which contains batch of input images
ARGS_INFERENCE_DIR = ARTIFACT_DETECTION_DIR + "utilities/network_verify/network_verification_images/"

# Flag to save colorized masks
ARGS_SAVE_COLOR = 1

ARGS_NUM_WORKERS = 8
ARGS_BATCH_SIZE = 1
ARGS_CPU = False
