# Model imports
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv3D, UpSampling3D, Activation, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add, SpatialDropout3D, Conv3DTranspose #old: Deconvolution3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow_addons.layers import InstanceNormalization
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# Define model functions
def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),
                             activation=LeakyReLU, padding='same', strides=(1, 1, 1), instance_normalization=True):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2

def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def get_model(input_shape=(128, 128, 128, 4), n_base_filters=8, depth=5, dropout_rate=0.5,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      activation_name="softmax", loss_function=None):
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='mse', metrics=['accuracy'])
    return model
