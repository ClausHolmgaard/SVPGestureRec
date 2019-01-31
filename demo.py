import os
import cv2
import time
import numpy as np
from Kinect import Kinect
from keras.models import load_model

from ModelConfig import *
from Helpers.RHDHelpers import *
from Model.PoolingAndFire import create_loss_function
from DataHandling.PreProcessing import get_anchors
from Helpers.GeneralHelpers import get_all_points_from_prediction


class Demo(object):
    """
    Demo showcasing gesture recognistion model.
    """
    def __init__(self, model_file):
        """
        @param model_file: String with path to model file.
        """
        self.model_file = model_file

        print("Loading model...")
        self.load_model()
        print("Model loaded.")

        # Start the kinect
        self.k = Kinect(debug=True, pointcloud=False, color=True)
        self.k.start()
        self.k.wait_for_init()
    
    def load_model(self):
        """
        Method for loading a keras model
        """
        # Due to the way Keras handles custom loss functions, we need to create a loss function here.
        # While some sort of stub could be used, it's easier to just use the one used when training the model
        l = create_loss_function(20, 20,
                         LABEL_WEIGHT,
                         OFFSET_LOSS_WEIGHT,
                         NUM_CLASSES,
                         EPSILON,
                         BATCHSIZE)

        self.model = load_model(self.model_file, custom_objects={'loss_function': l})

    def resize_image(self, im, target_size):
        """
        Resize an image. Cuts size of any dimension too large after resize.
        @param im: Image to resize
        @param target_size: Target size of image
        """
        # Find the resize factor
        resize_x_factor = im.shape[0] / target_size[0]
        resize_y_factor = im.shape[1] / target_size[1]
        resize_factor = np.min([resize_x_factor, resize_y_factor])

        # Find the new size of the image
        new_size = (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor))
        # Resize
        im_resize = cv2.resize(im, new_size)

        # Init array for output image
        im_out = np.zeros(target_size)

        # Cut width, if needed
        if im_out.shape[0] > target_size[0]:
            extra_pixels = im_out.shape[0] - target_size[0]
            remove_from_each_side = int(extra_pixels / 2)
            width_min = remove_from_each_side
            width_max = im_out.shape[0] - remove_from_each_side
        else:
            height_min = 0
            height_max = target_size[0]
        
        # Cut height, if needed
        if im_out.shape[1] > target_size[1]:
            extra_pixels = im_out.shape[1] - target_size[1]
            remove_from_each_side = int(extra_pixels / 2)
            width_min = remove_from_each_side
            width_max = im_out.shape[1] - remove_from_each_side
        else:
            width_min = 0
            width_max = target_size[1]
        
        # Slize image, and load into result
        im_out = im_resize[width_min:width_max,height_min:height_max]
        return im_out

    def preprocess(self, im):
        """
        Preprocess an image, so it's ready for inputting the the model.
        @param im: Image to preprocess
        """
        #print(f"shape:{im.shape}")
        im_pre = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Mean subtraction
        im_pre -= np.mean(im_pre)
        # Normalization
        im_pre /= np.std(im_pre, axis=-1)

        return im_pre.astype(np.float32)

    def get_image(self):
        """
        Get an image from the Kinect
        """
        im = self.k.get_color_image()

        return im

    def stop(self):
        """
        Stop execution, cleanup.
        """
        print("Stopping...")
        self.k.stop()

    def predict_points(self, im):
        """
        Predict points in an image.
        @param im: Image to predict points from
        """
        # Preprocess the image
        im_pre = self.preprocess(im)

        # Make prediction
        res = self.model.predict(im_pre.reshape(1, WIDTH, HEIGHT, CHANNELS))

        # Loop over all points/classes
        for i in range(NUM_CLASSES):

            # Handle which hand it is
            finger_index = i
            if finger_index >= NUM_CLASSES / 2:
                finger_index -= NUM_CLASSES / 2
            
            # Initialize a prediction matrix for single point
            pred = np.zeros((20, 20, 3))
            
            # Load with data
            pred[:, :, 0] = res[0, :, :, i]
            pred[:, :, 1] = res[0, :, :, NUM_CLASSES+i*2]
            pred[:, :, 2] = res[0, :, :, NUM_CLASSES+1+i*2]
            
            # Get anchor matrix
            anchors = get_anchors(WIDTH, HEIGHT, 20, 20)
            # And get all the points from this single prediction
            pred_point = get_all_points_from_prediction(pred,
                                                        anchors,
                                                        threshold=THRESHOLD,
                                                        offset_weight=int(320/20)/2,
                                                        is_label=False)

            # If any points found, get the mean value as result
            if len(pred_point) > 0:
                x_points = np.zeros(len(pred_point))
                y_points = np.zeros(len(pred_point))
                for counter, p in enumerate(pred_point):
                    x_points[counter] = p[0] + p[2]
                    y_points[counter] = p[1] + p[2]

                x = np.mean(x_points)
                y = np.mean(y_points)

                # And draw it in the image
                cv2.circle(im, (int(x), int(y)), 1, (255, 0, 0), thickness=2)

    def run(self):
        """
        Start the demo
        """
        while(True):
            # Get an image
            im = self.get_image()
            # Resize the imaee
            im = self.resize_image(im, (WIDTH, HEIGHT))
            # And make prediction
            self.predict_points(im)

            cv2.imshow("Main", im)

            # Run till q is pressed
            if cv2.waitKey(1) == ord('q'):
                self.stop()
                break

if __name__ == "__main__":
    MODEL_FILE = os.path.expanduser("~/results/SVPGestureRec/all_points_test2.h5py")
    demo = Demo(MODEL_FILE)
    demo.run()
