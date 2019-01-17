from keras.layers import *
from keras.models import Model
from keras.initializers import TruncatedNormal


def create_model(width, height, channels, num_classes, weight_decay=0, keep_prob=0.5, regularizer=None):
    """
    Same as above, except now we want to detect multiple classes and multiple offsets.
    """
    # Are reference to the fire layer, so it's easier to switch between different versions.
    fl = fire_layer_batchnorm

    # Input to the model
    input_layer = Input(shape=(width, height, channels), name="input")

    conv1 = Conv2D(name='conv1',
                   filters=128, kernel_size=(3, 3), strides=(2, 2),
                   activation='relu',
                   padding="SAME",
                   use_bias=True,
                   kernel_initializer=TruncatedNormal(stddev=0.01),
                   kernel_regularizer=regularizer
                   )(input_layer)

    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool1")(conv1)

    fire1_1 = fl(name="fire1_1", input=pool1,   s1x1=32, e1x1=128, e3x3=128, regularizer=regularizer)
    fire1_2 = fl(name="fire1_2", input=fire1_1, s1x1=32, e1x1=128, e3x3=128, regularizer=regularizer)
    fire1_3 = fl(name="fire1_3", input=fire1_2, s1x1=32, e1x1=128, e3x3=128, regularizer=regularizer)
    fire1_4 = fl(name="fire1_4", input=fire1_3, s1x1=32, e1x1=128, e3x3=128, regularizer=regularizer)

    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool2")(fire1_4)

    fire2_1 = fl(name="fire2_1", input=pool2,   s1x1=48, e1x1=192, e3x3=192, regularizer=regularizer)
    fire2_2 = fl(name="fire2_2", input=fire2_1, s1x1=48, e1x1=192, e3x3=192, regularizer=regularizer)
    fire2_3 = fl(name="fire2_3", input=fire2_2, s1x1=48, e1x1=192, e3x3=192, regularizer=regularizer)
    fire2_4 = fl(name="fire2_4", input=fire2_3, s1x1=48, e1x1=192, e3x3=192, regularizer=regularizer)

    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool3")(fire2_4)

    fire3_1 = fl(name="fire3_1", input=pool3,   s1x1=64, e1x1=256, e3x3=256, regularizer=regularizer)
    fire3_2 = fl(name="fire3_2", input=fire3_1, s1x1=64, e1x1=256, e3x3=256, regularizer=regularizer)
    fire3_3 = fl(name="fire3_3", input=fire3_2, s1x1=96, e1x1=384, e3x3=384, regularizer=regularizer)
    fire3_4 = fl(name="fire3_4", input=fire3_3, s1x1=96, e1x1=384, e3x3=384, regularizer=regularizer)

    preds = Conv2D(name='preds',
                   filters=3*num_classes, kernel_size=(1, 1), strides=(1, 1),
                   activation='sigmoid',
                   padding="SAME",
                   kernel_initializer=TruncatedNormal(stddev=0.01)
                   )(fire3_4)

    return Model(inputs=input_layer, outputs=preds)

def create_loss_function(anchor_width,
                         anchor_height,
                         label_weight,
                         offset_weight,
                         offset_loss_weight,
                         num_classes,
                         epsilon,
                         batchsize):
    """
    Create a loss function for the model.
    label_weight makes label errors weigh mode.
    offset_weight is not currently used. TODO: Remove offset_weight
    offset_loss_weight makes offset errors weight more. TODO: Change this to offset_weight
    num_classes is the number of classes
    epsilon is a small number for calculation stability in log functions
    batchsize is batchsize
    """

    def loss_function(y_true, y_pred):
        """
        Number of outputfilters is num_classes + 2*num_classes.
        So the predicion output is batchsize x anchorwidth x anchorheight x (3 * num_classes)
        y_true is ground truth labels
        y_pred is predicted values
        """
        # number of labels
        num_labels = num_classes  # TODO: If more labels are needed, this needs changing
        num_non_labels = anchor_width * anchor_height - num_labels

        # the first num_classes are confidence scores
        c_labels = y_true[:, :, :, :num_classes]
        c_predictions = y_pred[:, :, :, :num_classes]
        
        # And then we have the offsets
        offset_labels = y_true[:, :, :, num_classes:]
        offset_predictions = y_pred[:, :, :, num_classes:]

        # First the confidence loss

        # Loss matrix for all confidence entries
        confidence_m_all = keras_binary_crossentropy(c_labels, c_predictions, epsilon)

        # Loss matrix for the correct label
        confidence_m_label = keras_binary_crossentropy(c_labels, c_predictions, epsilon) * c_labels

        # Loss matrix for non labels
        confidence_m_nonlabel = confidence_m_all - confidence_m_label
        
        # Summing and adding weight to label loss
        c_loss_label = K.sum(
            confidence_m_label
        ) / num_labels
        
        # summing and adding weight to non label loss
        c_loss_nonlabel = K.sum(
            confidence_m_nonlabel
        ) / num_non_labels
        
        c_loss = c_loss_label * label_weight + c_loss_nonlabel * (1 / label_weight)

        # And then the offset loss

        # Ground truth offsets
        true_offset_x = offset_labels[:, :, :, 0::2]
        true_offset_y = offset_labels[:, :, :, 1::2]

        # Predicted labels, sigma scaled to -1, 1 codomain.
        pred_offset_x = 2 * (offset_predictions[:, :, :, 0::2] - 0.5)
        pred_offset_y = 2 * (offset_predictions[:, :, :, 1::2] - 0.5)
        
        # Create a mask of entries different from 0
        g_x = K.less(true_offset_x, 0)
        l_x = K.greater(true_offset_x, 0)
        g_y = K.greater(true_offset_y, 0)
        l_y = K.less(true_offset_y, 0)
        
        g_x_i = K.cast(g_x, dtype='float32')
        l_x_i = K.cast(l_x, dtype='float32')
        g_y_i = K.cast(g_y, dtype='float32')
        l_y_i = K.cast(l_y, dtype='float32')

        mask_offset_x = K.clip(g_x_i + l_x_i, 0, 1.0)
        mask_offset_y = K.clip(g_y_i + l_y_i, 0, 1.0)
        
        o_loss_x = K.sum(
            #K.clip(
                K.square(
                    (true_offset_x - pred_offset_x) * mask_offset_x
                    )
            #, 0, 1.0)
        ) / K.sum(mask_offset_x)
        
        o_loss_y = K.sum(
            #K.clip(
                K.square(
                    (true_offset_y - pred_offset_y) * mask_offset_y
                    )
            #, 0, 1.0)
        ) / K.sum(mask_offset_y)
        
        o_loss = (o_loss_x + o_loss_y) * offset_loss_weight
        
        total_loss = K.abs(c_loss) + K.abs(o_loss)  # abs due to rounding errors. TODO: Find a better way to handle rounding errors.
        total_loss /= batchsize

        return total_loss
    return loss_function
        
def fire_layer(name, input, s1x1, e1x1, e3x3, stdd=0.01, regularizer=None):
    """
    wrapper for fire layer constructions
    :param name: name for layer
    :param input: previous layer
    :param s1x1: number of filters for squeezing
    :param e1x1: number of filter for expand 1x1
    :param e3x3: number of filter for expand 3x3
    :param stdd: standard deviation used for intialization
    :return: a keras fire layer
    """

    sq1x1 = Conv2D(
        name = name + '/squeeze1x1',
        filters=s1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(input)

    ex1x1 = Conv2D(
        name = name + '/expand1x1',
        filters=e1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(sq1x1)

    ex3x3 = Conv2D(
        name = name + '/expand3x3',
        filters=e3x3, kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=True,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation='relu',
        kernel_regularizer=regularizer
        )(sq1x1)

    return concatenate([ex1x1, ex3x3], axis=3)

def fire_layer_batchnorm(name, input, s1x1, e1x1, e3x3, stdd=0.01, regularizer=None):
    """
    wrapper for fire layer constructions, with batchnorm layers
    :param name: name for layer
    :param input: previous layer
    :param s1x1: number of filters for squeezing
    :param e1x1: number of filter for expand 1x1
    :param e3x3: number of filter for expand 3x3
    :param stdd: standard deviation used for intialization
    :return: a keras fire layer
    """

    sq1x1 = Conv2D(
        name = name + '/squeeze1x1',
        filters=s1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=regularizer
        )(input)

    bn1 = BatchNormalization(name=name+'/bn1')(sq1x1)
    act1 = Activation('relu', name=name+'/act1')(bn1)

    ex1x1 = Conv2D(
        name = name + '/expand1x1',
        filters=e1x1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=regularizer
        )(act1)
    
    bn2 = BatchNormalization(name=name+'/bn2')(ex1x1)
    act2 = Activation('relu', name=name+'/act2')(bn2)

    ex3x3 = Conv2D(
        name = name + '/expand3x3',
        filters=e3x3, kernel_size=(3, 3),
        strides=(1, 1),
        use_bias=False,
        padding='SAME',
        kernel_initializer=TruncatedNormal(stddev=stdd),
        activation=None,
        kernel_regularizer=regularizer
        )(act1)
    
    bn3 = BatchNormalization(name=name+'/bn3')(ex3x3)
    act3 = Activation('relu', name=name+'/act3')(bn3)

    return concatenate([act2, act3], axis=-1)


def binary_crossentropy(y, y_hat, epsilon):
    return y * (-np.log(y_hat + epsilon)) + (1-y) * (-np.log(1-y_hat + epsilon))

def keras_binary_crossentropy(y, y_hat, epsilon):
    return y * (-K.log(y_hat + epsilon)) + (1-y) * (-K.log(1-y_hat + epsilon))