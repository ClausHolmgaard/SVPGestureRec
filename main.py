import os
import sys
import numpy as np
from keras import optimizers
from keras import backend as K
from keras.regularizers import l2
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

from Helpers.GeneralHelpers import *
from Helpers.RHDHelpers import *
from DataHandling.PreProcessing import *
from DataHandling.DataGenerator import *
from Model.PoolingAndFire import *

from ModelConfig import *

# Set greyscale
if CHANNELS == 1:
    grey = True
else:
    grey = False

# Handle limiting of samples
if LIMIT_SAMPLES is None:
    num_samples = get_num_samples(DATA_DIR, type_sample='png')
else:
    num_samples = LIMIT_SAMPLES

# Number of training and test samples
num_train_samples = int((1-VALIDATION_SPLIT) * num_samples)
num_validation_samples = int(VALIDATION_SPLIT * num_samples)

# Get all samples and sort them
all_samples = sorted(get_all_samples(DATA_DIR, sample_type='png'))
# And grab the training samples
train_samples = all_samples[:num_train_samples]
# If we need validation samples, get them
if num_validation_samples > 0:
    validation_samples = all_samples[-num_validation_samples:]
else:
    validation_samples = []

# Do the actual split of training and test data
train_validation_split(DATA_DIR, TRAIN_DIR, VALIDATION_DIR, train_samples, validation_samples, sample_type='png')

# Create annotaions for the training data
create_rhd_annotations(RHD_ANNOTATIONS_FILE,
                       TRAIN_ANNOTATIONS,
                       TRAIN_DIR,
                       fingers='ALL',
                       hands_to_annotate='BOTH',
                       annotate_non_visible=True,
                       force_new_files=True)

# Create annotations for validation data
create_rhd_annotations(RHD_ANNOTATIONS_FILE,
                       VALIDATION_ANNOTATIONS,
                       VALIDATION_DIR,
                       fingers='ALL',
                       hands_to_annotate='BOTH',
                       annotate_non_visible=True,
                       force_new_files=True)

# Create the mode
model = create_model(WIDTH, HEIGHT, CHANNELS, NUM_CLASSES, regularizer=REGULARIZER)

# Get output shape from the model
out_shape = model.output_shape
# Define anchor shape, based on model output
anchor_width = out_shape[1]
anchor_height = out_shape[2]
print(f"\nNeeded anchor shape: {anchor_width}x{anchor_height}")

# Set offset scale, based on anchor size
offset_scale = (((WIDTH + HEIGHT) / 2) / ((anchor_height + anchor_width) / 2)) / 2
print(f"Offset scale: {offset_scale}")

# Create the loss function
l = create_loss_function(anchor_width,
                         anchor_height,
                         LABEL_WEIGHT,
                         OFFSET_LOSS_WEIGHT,
                         NUM_CLASSES,
                         EPSILON,
                         BATCHSIZE)

# Set choices for the menu, depending on existance of save files
valid_choices = ['0', '1']
if os.path.exists(MODEL_CHECKPOINT_FILE):
    valid_choices.append('2')
if os.path.exists(MODEL_SAVE_FILE):
    valid_choices.append('3')

# Show the menu and get input
out = None
print("")
if len(valid_choices) > 2:
    while not(out in valid_choices):
        print("0: Exit")
        print("1: Fresh run")
        if '2' in valid_choices:
            print("2: Load checkpoint file")
        if '3' in valid_choices:
            print("3: Load file from completed run.")
        out = input("Selection: ")

# React on the input
# Note, when loading a model file, the earlier model is overwritten
# It must be done this way, due to the loading of the model requiring a reference to the loss function
if out == '0':
    sys.exit()
elif out == '1':
    if NUM_GPU > 1:
        model = multi_gpu_model(model, gpus=NUM_GPU)
elif out == '2':
    print(f"Loading {MODEL_CHECKPOINT_FILE}...")
    model = load_model(MODEL_CHECKPOINT_FILE, custom_objects={'loss_function': l})
elif out == '3':
    print(f"Loading {MODEL_SAVE_FILE}...")
    model = load_model(MODEL_SAVE_FILE, custom_objects={'loss_function': l})

# Print a summary of the active model
model.summary()

# Define an optimizer for the model
opt = optimizers.Adam(lr=INITIAL_LR, decay=OPT_DECAY)
# Compile the model, to prepare for training
model.compile(loss=l, optimizer=opt)

# Calculate the number of steps required for an epoch
print(f"Number of training samples: {num_train_samples}")
steps_epoch = STEPS_EPOCH_SCALE * num_train_samples // BATCHSIZE
if steps_epoch < 1:
    steps_epoch = 1
print(f"Steps per epoch: {steps_epoch}")

# A callback method for printing the actual learning rate.
# This is when using a decay for the optimizer, as this decay is not reflected in the models learning rate.
class PrintInfo(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(f"Learning rate with decay: {K.eval(lr_with_decay)}")
        print("")

print_info = PrintInfo()

# A callback for doing learning rate decay outside the optimizer
def lr_decay(epoch):
	initial_lrate = INITIAL_LR
	drop = DECAY_DROP
	epochs_drop = DECAY_EPOCHES
	lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(lr_decay)

# Keras method for reducing learning rate, of loss stops improving
reduce_lr_plateau = ReduceLROnPlateau(monitor='loss', 
                                      factor=0.5,
                                      patience=3,
                                      verbose=1,
                                      mode='auto',
                                      min_delta=0.01,
                                      cooldown=10,
                                      min_lr=1e-6)

# Callback for tensorboard with learning rate
class LRTensorBoard(TensorBoard):
    def __init__(self,
                 log_dir,
                 histogram_freq=0,
                 batch_size=BATCHSIZE,
                 write_graph=True,
                 update_freq='batch'):

        super().__init__(log_dir=log_dir,
                         histogram_freq=histogram_freq,
                         batch_size=batch_size,
                         write_graph=write_graph,
                         update_freq=update_freq)
    
    def on_batch_end(self, batch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_batch_end(batch, logs)

# Instance of tensorboard without learning rate
tensorboard = TensorBoard(log_dir=log_folder,
                          histogram_freq=0,
                          batch_size=BATCHSIZE,
                          write_graph=True,
                          write_grads=False,
                          write_images=False,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None,
                          embeddings_data=None,
                          update_freq='batch')

# Instance of tensorboard with learning rate
lr_tensorboard = LRTensorBoard(log_dir=log_folder,
                               histogram_freq=0,
                               batch_size=BATCHSIZE,
                               write_graph=True,
                               update_freq='batch')

# Instance of checkpointing of model
checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_FILE,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

# Data generator
train_data_gen = create_data_generator(TRAIN_DIR,
                                       TRAIN_ANNOTATIONS,
                                       BATCHSIZE,
                                       WIDTH, HEIGHT, CHANNELS,
                                       anchor_width,
                                       anchor_height,
                                       offset_scale,  # In which distance additional offets are chosen.
                                       num_classes=NUM_CLASSES,
                                       sample_type='png',
                                       greyscale=grey,
                                       verbose=False,
                                       queue_size=100,
                                       preload_all_data=PRELOAD_DATA)

# Start training of model
model.fit_generator(train_data_gen,
                    steps_per_epoch=steps_epoch,
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    callbacks=[reduce_lr_plateau, lr_tensorboard, checkpoint]
                    )

# Save model after training
print("Saving completed model...")
model.save(MODEL_SAVE_FILE)
