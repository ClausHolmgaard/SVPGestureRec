sq1x1 = Conv2D(
    name = name + '/squeeze1x1', filters=s1x1,
    kernel_size=(1, 1), strides=(1, 1),
    use_bias=True, activation='None',
    padding='SAME', kernel_regularizer=regularizer
    )(input)

ex1x1 = Conv2D(
    name = name + '/expand1x1', filters=e1x1,
    kernel_size=(1, 1), strides=(1, 1),
    use_bias=False, activation=None,
    padding='SAME', kernel_regularizer=regularizer
    )(sq1x1)

bn2 = BatchNormalization(name=name+'/bn2')(ex1x1)
act2 = Activation('relu', name=name+'/act2')(bn2)

ex3x3 = Conv2D(
    name = name + '/expand3x3', filters=e3x3,
    kernel_size=(3, 3), strides=(1, 1),
    use_bias=False, activation=None,
    padding='SAME', kernel_regularizer=regularizer
    )(sq1x1)

bn3 = BatchNormalization(name=name+'/bn3')(ex3x3)
act3 = Activation('relu', name=name+'/act3')(bn3)

return concatenate([act2, act3], axis=-1)



