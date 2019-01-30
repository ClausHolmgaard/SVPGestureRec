import os
import datetime


LOG_DIR = os.path.expanduser("~/logs/SVPGestureRec/")
DATA_DIR = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/color")
TRAIN_DIR = os.path.expanduser("~/datasets/RHD/processed/train")
VALIDATION_DIR = os.path.expanduser("~/datasets/RHD/processed/validation")
TRAIN_ANNOTATIONS = os.path.expanduser("~/datasets/RHD/processed/train/annotations")
VALIDATION_ANNOTATIONS = os.path.expanduser("~/datasets/RHD/processed/validation/annotations")
RHD_ANNOTATIONS_FILE = os.path.expanduser("~/datasets/RHD/RHD_published_v2/training/anno_training.pickle")
MODEL_CHECKPOINT_FILE = os.path.expanduser("~/results/SVPGestureRec/model_checkpoint.h5py")
MODEL_SAVE_FILE = os.path.expanduser("~/results/SVPGestureRec/model_save_done.h5py")

# Timestamp for tensorboard log dir
timestamp = '{:%Y-%m-%d_%H_%M}'.format(datetime.datetime.now())
log_folder = os.path.join(LOG_DIR, timestamp)

# Log funktion stability
EPSILON = 1e-16

# Input size
HEIGHT = 320
WIDTH = 320
CHANNELS = 1

# label versus offset weight in loss function
LABEL_WEIGHT = 1.0
OFFSET_LOSS_WEIGHT = 1.0

# Learning rate
INITIAL_LR = 1e-4

# Weight decay in optimizer
OPT_DECAY = 0

# Weight decay in decay method
DECAY_EPOCHES = 100.0
DECAY_DROP = 0.1

# Number of classes
NUM_CLASSES = 42
# Number of hands
NUM_HANDS = 2

# Number of validation samples
VALIDATION_SPLIT = 0.01

# Number of GPU's
NUM_GPU = 1
# Batch size
BATCHSIZE = 64
# Epohcs to run for
NUM_EPOCHS = 500

# Limit the number of samples
LIMIT_SAMPLES = None
# Scale the number of steps per epoch
STEPS_EPOCH_SCALE = 1

# Load all data into RAM
PRELOAD_DATA = False

# Use a reguralizer in layers in model
REGULARIZER = None
# Decay for reguralizer
WEIGHT_DECAY = 0

# Threshold for detection
THRESHOLD = 0.8