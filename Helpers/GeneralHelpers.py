import os
import cv2
import numpy as np


def get_num_samples(data_dir, type_sample='jpg'):
    num_samples = 0
    for f in os.listdir(data_dir):
        end = f.split('.')[1]
        if end == type_sample:
            num_samples += 1
    
    return num_samples

def get_all_samples(data_dir, sample_type='jpg'):
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
    image_name = "%05d.png" % index
    im = cv2.imread(os.path.join(path, image_name))
    if grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def get_all_points_from_prediction(pred, anchors, threshold=1.0, offset_weight=1.0, num_classes=1, is_label=True):
    """
    pred is a prediction map in the shape (ANCHOR_HEIGHT, ANCHOR_WIDTH, 3*num_classes)
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