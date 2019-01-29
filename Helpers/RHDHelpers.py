import os
import cv2
import pickle
from tqdm import tqdm
from shutil import copyfile

from Helpers.GeneralHelpers import *

# A structure to map fingers to a number, so we're sure the order is preserved.
FINGER_MAP = {"Wrist": 0,
              "Thumb1": 1,
              "Thumb2": 2,
              "Thumb3": 3,
              "Thumb4": 4,
              "Index1": 5,
              "Index2": 6,
              "Index3": 7,
              "Index4": 8,
              "Middle1": 9,
              "Middle2": 10,
              "Middle3": 11,
              "Middle4": 12,
              "Ring1": 13,
              "Ring2": 14,
              "Ring3": 15,
              "Ring4": 16,
              "Pinky1": 17,
              "Pinky2": 18,
              "Pinky3": 19,
              "Pinky4": 20}

# An inverse map for convinience
FINGER_MAP_INV = {v: k for k, v in FINGER_MAP.items()}

def get_hand_points(index, annotations, offset):
    """
    Array with entry for each point. Each entry is (x, y, visible)
    Where visible is 1 for seen points, 0 for hidden.
    This will grab 21 points, starting at offset.
    @param index: index in annotations
    @param annotations: annotations loaded from dataset
    @param offset: 0 or 21, left or right hand.
    """
    # Get the index, and entry in array
    this_index = annotations[index]['uv_vis']
    

    points = [None] * 21

    # Grab all the points
    points[FINGER_MAP["Wrist"]] = this_index[offset + 0]

    points[FINGER_MAP["Thumb1"]] = this_index[offset + 1]
    points[FINGER_MAP["Thumb2"]] = this_index[offset + 2]
    points[FINGER_MAP["Thumb3"]] = this_index[offset + 3]
    points[FINGER_MAP["Thumb4"]] = this_index[offset + 4]

    points[FINGER_MAP["Index1"]] = this_index[offset + 5]
    points[FINGER_MAP["Index2"]] = this_index[offset + 6]
    points[FINGER_MAP["Index3"]] = this_index[offset + 7]
    points[FINGER_MAP["Index4"]] = this_index[offset + 8]

    points[FINGER_MAP["Middle1"]] = this_index[offset + 9]
    points[FINGER_MAP["Middle2"]] = this_index[offset + 10]
    points[FINGER_MAP["Middle3"]] = this_index[offset + 11]
    points[FINGER_MAP["Middle4"]] = this_index[offset + 12]

    points[FINGER_MAP["Ring1"]] = this_index[offset + 13]
    points[FINGER_MAP["Ring2"]] = this_index[offset + 14]
    points[FINGER_MAP["Ring3"]] = this_index[offset + 15]
    points[FINGER_MAP["Ring4"]] = this_index[offset + 16]

    points[FINGER_MAP["Pinky1"]] = this_index[offset + 17]
    points[FINGER_MAP["Pinky2"]] = this_index[offset + 18]
    points[FINGER_MAP["Pinky3"]] = this_index[offset + 19]
    points[FINGER_MAP["Pinky4"]] = this_index[offset + 20]

    return points

def get_left_hand(index, annotations):
    """
    Get all the points for the left hand.
    @param index: index in annotations
    @param annotations: annotations loaded from dataset
    """
    return get_hand_points(index, annotations, 0)

def get_right_hand(index, annotations):
    """
    Get all the points from the right hand
    @param index: index in annotations
    @param annotations: annotations loaded from dataset
    """
    return get_hand_points(index, annotations, 21)

def train_validation_split(data_path, train_path, validation_path, train_samples, validation_samples, sample_type='jpg'):
    """
    Process files in data_path of the format xxxxx.sample_type.
    It will put the samples specified in train_samples into train path, and validation samples into validation path.
    It is done this way to preserve train and validation sample structure between local and remote machine.

    ALL FILES IN train_path AND validation_path WILL BE DELETED
    @param data_path: Path to data
    @param train_path: Where to put training data
    @param validation_path: Where to put validation data
    @param train_samples: Which samples to put in train_path
    @param validation_samples: Which samples to put in validation_path
    @param sample_type: File type of samples
    """
    remove_files_in_folder(train_path)
    remove_files_in_folder(validation_path)
    
    print(f"Doing train/validation split. {len(train_samples)} training samples, {len(validation_samples)} validation samples.")
    for fi in tqdm(os.listdir(data_path)):
        if fi.endswith(sample_type):
            obj = fi.split('.')
            try:
                ind = int(obj[0])
            except:
                continue
            if ind in train_samples:
                copyfile(os.path.join(data_path, fi), os.path.join(train_path, fi))
                
            if ind in validation_samples:
                copyfile(os.path.join(data_path, fi), os.path.join(validation_path, fi))

def create_rhd_annotations(annotations_file,
                           annotations_out_path,
                           color_path,
                           fingers='ALL',
                           hands_to_annotate='BOTH',
                           annotate_non_visible=True,
                           force_new_files=False):
    """
    Create annotations for RHD dataset.
    @param annotations_file: Annotations file that came with the dataset.
    @param annotations_out_path: Where the resulting annotations from this will end up.
    @param color_path: Path to the color images that should be annotated.
    @param fingers: An array with the fingers to annotate, or ALL for all fingers.
    @param hands: right, left or BOTH.
    """
    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    if force_new_files:
        remove_files_in_folder(annotations_out_path)

    print(f"Creating annotations in directory: {color_path}")
    print(f"Using annotation file: {annotations_file}")
    print(f"And outputting to: {annotations_out_path}")
    for fi in tqdm(os.listdir(color_path)):
        if fi.endswith('png'):
            anno_file_name = f"{fi.split('.')[0]}.an"
            anno_file_path = os.path.join(annotations_out_path, anno_file_name)
            ind = int(fi.split('.')[0])
            
            right_hand = get_right_hand(ind, annotations)
            left_hand = get_left_hand(ind, annotations)
            
            with open(anno_file_path, 'w') as write_file:
                if hands_to_annotate.lower() == 'right':
                    hands = [right_hand]
                elif hands_to_annotate.lower() == 'left':
                    hands = [left_hand]
                else:
                    hands = [right_hand, left_hand]

                for h in hands:
                    if fingers == 'ALL':
                        for p in h:
                            visible = p[2] != 0
                            if visible or annotate_non_visible:
                                write_file.write(f"{float(p[0])},{float(p[1])}\n")
                    else:
                        for f in fingers:
                            p = h[FINGER_MAP[f]]
                            visible = p[2] != 0
                            if visible or annotate_non_visible:
                                write_file.write(f"{float(p[0])},{float(p[1])}\n")
