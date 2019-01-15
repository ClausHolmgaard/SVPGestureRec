import os
import cv2


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
