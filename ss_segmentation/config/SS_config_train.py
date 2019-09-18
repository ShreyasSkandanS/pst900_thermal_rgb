# Currently input channels are default to 3
NUM_CHANNELS 		= 3

# Input image dimensions
IMG_HEIGHT		= 360
IMG_WIDTH 		= 640

ENC_SCALEDOWN		= 8

OPT_LEARNING_RATE_INIT 	= 5e-4

OPT_BETAS 		= (0.9, 0.999)
OPT_EPS_LOW 		= 1e-08
OPT_WEIGHT_DECAY 	= 1e-4

# For unbiased label rebalancing
NUM_CLASSES             = 5
CLASS_0_WEIGHT		= 1.45471 	# Background
CLASS_1_WEIGHT		= 43.872	# Fire Extinguisher
CLASS_2_WEIGHT		= 34.241		# Backpack
CLASS_3_WEIGHT		= 47.36 	# Hand Drill
#CLASS_4_WEIGHT		= 10.470	# Cellphone
CLASS_4_WEIGHT		= 27.4869	# Survivor

ARGS_NUM_WORKERS	= 40
ARGS_BATCH_SIZE		= 2
ARGS_NUM_EPOCHS 	= 500
ARGS_IOU_TRAIN		= True
ARGS_IOU_VAL		= True
ARGS_STEPS_LOSS		= 20
ARGS_MODEL		= "nr_network_design"
ARGS_STATE		= ""
ARGS_DECODER		= ""
ARGS_PRETRAINED_ENCODER = ""
ARGS_CUDA		= True

# Make sure that the {train|val} folders contain "images" and "labels" subdirectories
# --- To change this data layout :: SS_data_definition.py
# SAVE_FOLDER: Folder which will contain learned model + any metadata files
SAVE_FOLDER             = "test_1"
ARGS_TRAIN_DIR		= "/home/neil/Desktop/test_images_100_wft"
ARGS_VAL_DIR		= "/home/neil/Desktop/test_images_100_wft"
#ARGS_VAL_DIR		= "/data/Docker_Data/RGBDT_100_NC_wft/test"
ARGS_SAVE_DIR           = "/home/neil/Desktop/" + SAVE_FOLDER
ARGS_REPO_DIR		= "/home/neil/Desktop/paper_code/pst900_thermal_rgb/ss_segmentation/"

ARGS_EPOCHS_SAVE	= 5
ARGS_RESUME		= False


