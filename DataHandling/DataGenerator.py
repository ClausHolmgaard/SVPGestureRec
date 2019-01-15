import queue
import threading
import numpy as np

from Helpers.GeneralHelpers import *
from DataHandling.PreProcessing import *


def create_data_generator(directory,
                          annotations_dir,
                          batch_size,
                          image_width,
                          image_height,
                          channels,
                          anchor_width,
                          anchor_height,
                          offset_scale,
                          num_classes=1,
                          sample_type='jpg',
                          greyscale=False,
                          verbose=False,
                          queue_size=1,
                          preload_all_data=False):

    print(f"Starting data generator in: {directory}, with annotations in {annotations_dir}")
    if verbose:
        print(f"Samples: {samples}")

    samples = get_all_samples(directory, sample_type=sample_type)
    
    if preload_all_data:
        all_labels, all_images = load_data_with_anchors(samples,
                                                        directory,
                                                        annotations_dir,
                                                        image_width,
                                                        image_height,
                                                        anchor_width,
                                                        anchor_height,
                                                        offset_scale,
                                                        sample_type,
                                                        num_classes=num_classes,
                                                        greyscale=greyscale,
                                                        progressbar=True)
        
        while True:
            ind = np.random.randint(0, len(samples), size=batch_size)

            batch_labels = all_labels[ind]
            batch_images = all_images[ind]

            yield batch_images, batch_labels
    else:
        gen = BackgroundGenerator(directory,
                                  batch_size,
                                  annotations_dir,
                                  image_width,
                                  image_height,
                                  anchor_width,
                                  anchor_height,
                                  offset_scale,
                                  sample_type,
                                  num_classes,
                                  greyscale,
                                  samples,
                                  queue_size)

        while True:
            batch_labels, batch_images = gen.next()

            yield batch_images, batch_labels

class BackgroundGenerator(threading.Thread):
    def __init__(self,
                 directory,
                 batch_size,
                 annotations_dir,
                 image_width,
                 image_height,
                 anchor_width,
                 anchor_height,
                 offset_scale,
                 sample_type,
                 num_classes,
                 greyscale,
                 available_sampels,
                 queue_size=1):
        threading.Thread.__init__(self)

        self.directory = directory
        self.batch_size = batch_size
        self.annotations_dir = annotations_dir
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_width = anchor_width
        self.anchor_height = anchor_height
        self.offset_scale = offset_scale
        self.sample_type = sample_type
        self.num_classes = num_classes
        self.greyscale = greyscale
        self.available_sampels = available_sampels

        self.queue = queue.Queue(maxsize=queue_size)
        self.daemon = True
        self.is_running = True
        self.start()

    def run(self):
        while self.is_running:
            batch_samples = np.random.choice(self.available_sampels, size=self.batch_size)

            self.queue.put(load_data_with_anchors(batch_samples,
                                                  self.directory,
                                                  self.annotations_dir,
                                                  self.image_width,
                                                  self.image_height,
                                                  self.anchor_width,
                                                  self.anchor_height,
                                                  self.offset_scale,
                                                  self.sample_type,
                                                  num_classes=self.num_classes,
                                                  greyscale=self.greyscale))

    def stop(self):
        self.is_running = False

    def next(self):
        return self.queue.get()
