import os
import cv2
import numpy as np
from keras import backend as K


def get_num_samples(data_dir, type_sample='jpg'):
    """
    Get number of files in data_dir of type type_sample.
    @param data_dir: Directory to get samples from
    @param type_sample: File extension of samples
    """
    num_samples = 0
    for f in os.listdir(data_dir):
        end = f.split('.')[1]
        if end == type_sample:
            num_samples += 1
    
    return num_samples

def get_all_samples(data_dir, sample_type='jpg'):
    """
    Get a list of all samples from data_dir with extension sample_type
    @param data_dir: Directory to get samples from
    @param type_sample: File extension of samples
    """
    samples = []
    for fi in os.listdir(data_dir):
        if fi.endswith(sample_type):
            obj = fi.split('.')
            try:
                ind = int(obj[0])
            except:
                continue
            samples.append(ind)

    return samples

def remove_files_in_folder(folder, filetype=None):
    """
    Clean all files in a folder.
    BE CAREFUL.
    @param folder: Folder to clean files from.
    @param filetype: If different from None, only clean files of this type
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                if filetype is not None:
                    if not file_path.endswith(filetype):
                        continue
                os.unlink(file_path)
        except Exception as e:
            print(e)

def load_image(path, index, grayscale=False):
    """
    Zero pad an integer to 5 zeros, and load the png image from path.
    @param path: path to load file from
    @param index: Index of file to load
    @param greyscale: Convert image to greyscale
    """
    image_name = "%05d.png" % index
    im = cv2.imread(os.path.join(path, image_name))
    if grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def get_all_points_from_prediction(pred, anchors, threshold=1.0, offset_weight=1.0, num_classes=1, is_label=True):
    """
    @param pred: Prediction map in the shape (ANCHOR_HEIGHT, ANCHOR_WIDTH, 3*num_classes)
    @param anchors: Anchor map
    @param threshold: Confidence threshold for point to be detected
    @param offset_weight: Scale for offsets
    @param num_classes: Number of classes
    @param is_label: Is this a label or prediction
    """
    # Get all points with a confidence above threshold
    label_indicies = np.where(pred[:, :, 0] >= threshold)
    num_points = len(label_indicies[0])
    points = np.zeros((num_points, 4))
    
    # Loop through all anchor points
    for c, (x_anchor, y_anchor) in enumerate(zip(label_indicies[0], label_indicies[1])):
        # when anchor location is known, the location of the closest anchor in the actual image can be found
        x_without_offset, y_without_offset = anchors[x_anchor, y_anchor]
        
        # The offset can then be extracted from the labels
        (x_offset, y_offset) = pred[label_indicies[0], label_indicies[1]][c][1:]
        
        if not is_label:
            x_offset = 2 * (x_offset - 0.5)
            y_offset = 2 * (y_offset - 0.5)
        x_offset *= offset_weight
        y_offset *= offset_weight

        points[c] = (x_without_offset, y_without_offset, x_offset, y_offset)
    
    return points

def binary_crossentropy(y, y_hat, epsilon):
    """
    Binary crossentropy using numpy
    @param y: prediction
    @param y_hat: ground truth
    @param epsilon: Small value for log stability
    """
    return y * (-np.log(y_hat + epsilon)) + (1-y) * (-np.log(1-y_hat + epsilon))

def keras_binary_crossentropy(y, y_hat, epsilon):
    """
    Binary crossentropy using keras
    @param y: prediction
    @param y_hat: ground truth
    @param epsilon: Small value for log stability
    """
    return y * (-K.log(y_hat + epsilon)) + (1-y) * (-K.log(1-y_hat + epsilon))