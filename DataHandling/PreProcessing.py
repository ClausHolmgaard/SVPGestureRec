import os
import cv2
import numpy as np
from tqdm import tqdm


def closest_anchor_map(x, y,
                       image_width, image_height,
                       anchor_width, anchor_height,
                       anchor_coords,
                       offset_scale):
    """
    Create a anchor_height x anchor_width x 3 map.
    First entry is 1 if the anchor point is closest to true point. Zero otherwise.
    Second is x offset.
    Third is y offset.
    anchor_coords is is an anchor map created by the get_anchors method.
    offset_scale is a scale to scale offset down by. Ideally to get all offsets between -1 and 1.
    """

    # Set limit for how far away from the actual anchor offsets are set.
    #x_limit = image_width / anchor_width
    #y_limit = image_height / anchor_height
    #dist_limit = np.sqrt(x_limit**2 + y_limit**2)
    dist_limit = offset_scale

    # Initialize result matrix
    res = np.zeros((anchor_width, anchor_height, 3))

    # Make sure values for x and y are valid
    if x is not None and y is not None and x > 0 and y > 0 and x < image_width and y < image_height:
        # Get all image x and y coords for cooresponding anchor coords
        xs = anchor_coords[:, :, 0]
        ys = anchor_coords[:, :, 1]

        # Create a matrix with all distances between the x and y and anchor coords
        dist_matrix = np.sqrt( (xs - x)**2 + (ys - y)**2 )
        # Get the minimum value
        #min_val = np.min(dist_matrix)
        # Get all coords where the values in the distance matrix is less than the limit
        closest_xs, closest_ys = np.where(dist_matrix<=dist_limit)
        
        # Loop through all found values
        for cx, cy in zip(closest_xs, closest_ys):
            # Get the anchor coords in the image
            anchor_x, anchor_y = anchor_coords[cx, cy]
            # Set offsets
            closest_offset_x = (x - anchor_x) / offset_scale
            closest_offset_y = (y - anchor_y) / offset_scale
            res[cx, cy, 1:] = (closest_offset_x, closest_offset_y)

            res[cx, cy, 0] = 1
        
        # Set label in the closest anchor (where the distance matrix is min_val)
        #closest_x, closest_y = np.where(dist_matrix==min_val)
        #closest_x = closest_x[0]  # If multiple values, the first one is used
        #closest_y = closest_y[0]
        #res[closest_x, closest_y, 0] = 1
        
    return res

def get_anchors(image_width, image_height, anchor_width, anchor_height):
    """
    Generate a anchor_height x anchor_width x 2 matrix.
    Each entry is an (x, y) corrdinate mapping to image coordinates.
    """
    # Initialize result matrix
    anchors = np.zeros((anchor_width, anchor_height, 2))
    num_anchor_nodes = anchor_height * anchor_width
    
    # Start values for arrays
    x_start = image_width / (anchor_width + 1)
    x_end = image_width - x_start
    y_start = image_height / (anchor_height + 1)
    y_end = image_height - y_start
    # Create evenly spaced arrasy between min and max values
    xs = np.linspace(x_start, x_end, num=anchor_width, dtype=np.uint32)
    ys = np.linspace(y_start, y_end, num=anchor_height, dtype=np.uint32)
    
    # Use the arrays to fill the anchor matrix
    for ix in range(anchor_height):
        for iy in range(anchor_width):
            anchors[ix, iy] = (xs[ix], ys[iy])
    
    return anchors

def load_data_with_anchors(samples,
                           data_dir,
                           anno_dir,
                           image_width,
                           image_height,
                           anchor_width,
                           anchor_height,
                           offset_scale,
                           sample_type,
                           num_classes=1,
                           only_images=False,
                           greyscale=False,
                           progressbar=False):
    """
    load images
    labels will be:
    anchor_height x anchor_width x (3 * num_classes), 1 confidence score and x,y for offset.
    The first num_classes is confidence scores, then follows the offsets.
    """
    if greyscale:
        channels = 1
    else:
        channels = 3

    anchs = get_anchors(image_width, image_height, anchor_width, anchor_height)
    gt = np.zeros((len(samples), anchor_width, anchor_height, 3*num_classes))
    images = np.zeros((len(samples), image_width, image_height, channels))
    
    if progressbar:
        print(f"Loading {len(samples)} samples")
        ite = enumerate(tqdm(samples))
    else:
        ite = enumerate(samples)

    for c, s in ite:
        
        annotation_file = os.path.join(anno_dir, "%05d.an" % s)
        image_file = os.path.join(data_dir, "%05d.%s" % (s, sample_type))

        if not only_images:
            with open(annotation_file, 'r') as f:
                line_labels = f.readlines()
            """
            x_vals = []
            y_vals = []

            for line_label in line_labels:
                obj = line_label.split(',')
                if obj[0] != '':
                    x = float(obj[0])
                    y = float(obj[1])
                else:
                    x = -1
                    y = -1
                x_vals.append(x)
                y_vals.append(y)

            for cl in range(num_classes):
                x1 = x_vals[cl]
                x2 = x_vals[cl + num_classes]
                y1 = y_vals[cl]
                y2 = y_vals[cl + num_classes]

                cam1 = closest_anchor_map(x1, y1, image_width, image_height, anchor_width, anchor_height, anchs, offset_scale)
                cam2 = closest_anchor_map(x2, y2, image_width, image_height, anchor_width, anchor_height, anchs, offset_scale)
                cam = np.clip(cam1 + cam2, 0, 1.0)
                gt[c, :, :, cl] = cam[:, :, 0]
                gt[c, :, :, num_classes+cl*2] = cam[:, :, 1]
                gt[c, :, :, num_classes+1+cl*2] = cam[:, :, 2]
            """

            point = 0
            for line_label in line_labels:
                obj = line_label.split(',')
                if obj[0] != '':
                    x = float(obj[0])
                    y = float(obj[1])
                else:
                    x = None
                    y = None
                cam = closest_anchor_map(x, y, image_width, image_height, anchor_width, anchor_height, anchs, offset_scale)
                gt[c, :, :, point] = cam[:, :, 0]
                gt[c, :, :, num_classes+point*2] = cam[:, :, 1]
                gt[c, :, :, num_classes+1+point*2] = cam[:, :, 2]

                if x is not None and y is not None:
                    point += 1

        im = cv2.imread(image_file)
        if greyscale:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Mean subtraction
        im -= np.mean(im, axis=-1)
        # Normalization
        im /= np.std(im, axis=-1)

        # Reshape to fit input of model
        images[c] = im.reshape(image_width, image_height, channels)

    return gt, images
