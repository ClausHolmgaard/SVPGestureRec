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

def get_hand_points(index, annotations, offset):
    """
    Array with entry for each point. Each entry is (x, y, visible)
    Where visible is 1 for seen points, 0 for hidden.
    This will grab 21 points, starting at offset.
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
    """
    return get_hand_points(index, annotations, 0)

def get_right_hand(index, annotations):
    """
    Get all the points from the right hand
    """
    return get_hand_points(index, annotations, 21)
