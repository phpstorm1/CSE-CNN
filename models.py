from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import scipy.signal as sig
import tensorflow as tf

# TODO      replace all model_settings['spectrogram_size'] and model_settings['spectrogram_length'] by using shape of
#           the input spectrogram
#   complete

def prepare_model_settings(sample_rate, clip_duration_ms,
                           win_len, win_shift, nDFT,
                           frame_neighbor, snr, noise_type,
                           data_dir, save_path, model_architecture):
    """Calculates common settings needed for all models.

    Args:
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      win_len: Length of analysis window, in samples
      win_shift: Non-overlap length of the analysis window, in samples
      nDFT: Number of point for frequency analysis.
      frame_neighbor: Number of frames to join the current frame as the input.
          The total number of frames will be (2*frame_neighbor+1)
    Returns:
      Dictionary containing common settings.
    """
    model_settings = {
        'sample_rate': sample_rate,
        'clip_duration_ms': clip_duration_ms,
        'win_len': win_len,
        'win_shift': win_shift,
        'nDFT': nDFT,
        'frame_neighbor': frame_neighbor,
        'snr': snr,
        'noise_type': noise_type,
        'win_fun': np.hanning,
        'data_dir': data_dir,
        'save_path': save_path,
        'model_architecture': model_architecture,
    }
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    model_settings['desired_samples'] = desired_samples
    length_minus_window = (desired_samples - win_len)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / win_shift)
        # spectrogram_length = 2 + int(length_minus_window / win_shift)
    model_settings['spectrogram_length'] = spectrogram_length
    model_settings['spectrogram_size'] = (spectrogram_length * (int(model_settings['nDFT']/2)+1))

    return model_settings


def create_model(spectrogram_input, model_settings, is_training,
                 additonal_spectrogram_input=None, runtime_settings=None):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['spectrogram_size'].

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    Args:
      spectrogram_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
      additional_spectrogram_input: required when two spectrograms are used.
      runtime_settings: Dictionary of information about the runtime.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
      Exception: If the architecture type isn't recognized.
    """
    model_architecture = model_settings['model_architecture']
    if model_architecture == 'cs-cnn':
        return create_conv_net(spectrogram_input, additonal_spectrogram_input, model_settings, is_training)

    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "fc", "conv"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def pad_adjacent(spectrogram_input, model_settings):

    input_spectrogram_nDFT = int(model_settings['nDFT'] / 2) + 1
    frame_neighbor = model_settings['frame_neighbor']

    # pad every spectrogram with zeros frames in both sides
    spectrogram_input = tf.to_float(spectrogram_input)
    num_files = tf.shape(spectrogram_input)[0]
    spectrogram_4d = tf.reshape(spectrogram_input,
                                [tf.to_int32(num_files), -1, input_spectrogram_nDFT, 1])
    pad_zero = tf.zeros(shape=[tf.shape(spectrogram_4d)[0], frame_neighbor,
                               tf.shape(spectrogram_4d)[2], tf.shape(spectrogram_4d)[3]])
    pad_spectrogram = tf.concat(values=[pad_zero, spectrogram_4d, pad_zero], axis=1)

    pad_neighbor = tf.TensorArray(dtype=tf.float32, size=tf.to_int32(tf.shape(spectrogram_4d)[1]
                                                                     * tf.shape(pad_spectrogram)[0]), name="pad_neighbor")

    def t_fn(i, j):
        return i, j+1

    def f_fn(i, j):
        return i+1, 0

    def cond(i, j, pad_neighbor):
        return tf.shape(pad_spectrogram)[0] > i

    def body(i, j, pad_neighbor):
        pad_neighbor = pad_neighbor.write(i * (tf.shape(pad_spectrogram)[1] - 2 * frame_neighbor) + j,
                                          tf.slice(pad_spectrogram, [i, j, 0, 0], [1, (2*frame_neighbor+1), input_spectrogram_nDFT, 1]))
        i, j = tf.cond(j < (tf.shape(pad_spectrogram)[1] - 2 * frame_neighbor - 1),
                       lambda: t_fn(i, j), lambda: f_fn(i, j))
        return i, j, pad_neighbor

    i, j, pad_neighbor = tf.while_loop(cond=cond, body=body, loop_vars=[0, 0, pad_neighbor])

    # get neighbor-padded spectrogram
    pad_neighbor = pad_neighbor.stack()
    # seems no need of tf.squeeze for convets
    pad_neighbor = tf.squeeze(pad_neighbor, axis=1)
    return pad_neighbor


def create_convnet_setting(model_settings):

    net_settings = model_settings
    layers = dict()

    layers['conv2d_layers'] = (
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
    )

    layers['fc_layers'] = (
        'fcl_1',
        'fcl_2',
        'fcl_3_real',
        'fcl_3_imag'
    )

    net_settings.update({
        'layers': layers,
        'filter_height': 1,
        'filter_width': 25,
        'filter_channel': 50,
        'fcl_node': 512,
        'input_height': model_settings['frame_neighbor'] * 2 + 1,
        'input_width': int(model_settings['nDFT'] / 2) + 1,

    })

    return net_settings


def create_variable(name, shape):
    """Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition."""
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_conv2d_variables(net_settings):
    layers = net_settings['layers']['conv2d_layers']
    filter_height = net_settings['filter_height']
    filter_width = net_settings['filter_width']
    filter_channel = net_settings['filter_channel']
    input_height = net_settings['input_height']
    input_width = net_settings['input_width']
    frame_neighbor = net_settings['frame_neighbor']

    var = dict()
    how_many_conv2d_layer = layers[-1][4]

    var['conv2d'] = list()
    with tf.variable_scope('conv2d'):
        for i, name in enumerate(layers):
            layer_index = name[4]
            if layer_index > how_many_conv2d_layer:
                break
            with tf.variable_scope('layer{}'.format(layer_index)):
                cur = dict()

                if int(layer_index) == 1:
                    cur['filter'] = create_variable('filter',
                                                    [filter_height, filter_width, 1, filter_channel])
                    cur['bias'] = create_variable('bias', [filter_channel])
                elif int(layer_index) == int(how_many_conv2d_layer):
                    cur['filter'] = create_variable('filter',
                                                    [1, 1, filter_channel, 1])
                    cur['bias'] = create_variable('bias', [1])
                else:
                    cur['filter'] = create_variable('filter',
                                                    [filter_height, filter_width, filter_channel, filter_channel])
                    cur['bias'] = create_variable('bias', [filter_channel])

                var['conv2d'].append(cur)

    return var


def create_fcl_variables(net_settings):
    layers = net_settings['layers']['fc_layers']
    fcl_node = net_settings['fcl_node']
    fcl_output_size = net_settings['input_width']
    fcl_input_size = fcl_output_size


    var = dict()
    var['fcl'] = list()
    how_mant_fcl_layers = int(layers[-1][4])
    with tf.variable_scope("fcl"):
        for i, name in enumerate(layers):
            layer_index = name[4]
            with tf.name_scope('layer{}'.format(layer_index)):
                cur = dict()

                if int(layer_index) == 1:
                    cur['weights'] = create_variable('weights',
                                                     [fcl_input_size, fcl_node])
                    cur['bias'] = create_variable('bias',
                                                  [fcl_node])

                elif int(layer_index) == how_mant_fcl_layers and name[6:] == 'real':
                    cur['weights_real'] = create_variable('weights_real',
                                                     [fcl_node, fcl_output_size])
                    cur['bias_real'] = create_variable('bias_real',
                                                  [fcl_output_size])

                elif int(layer_index) == how_mant_fcl_layers and name[6:] == 'imag':
                    cur['weights_imag'] = create_variable('weights_imag',
                                                     [fcl_node, fcl_output_size])
                    cur['bias_imag'] = create_variable('bias_imag',
                                                  [fcl_output_size])

                else:
                    cur['weights'] = create_variable('weights',
                                                     [fcl_node, fcl_node])
                    cur['bias'] = create_variable('bias',
                                                  [fcl_node])

                var['fcl'].append(cur)

    return var


def convnet_conv2d(input_spectrogram, weights, net_settings):
    layers = net_settings['layers']['conv2d_layers']
    weights = weights['conv2d']
    current = input_spectrogram

    how_many_conv2d_layer = layers[-1][4]
    for i, name in enumerate(layers):
        layer_index = name[4]
        cur_weight = weights[i]['filter']
        cur_bias = weights[i]['bias']
        current = tf.nn.conv2d(current,
                               cur_weight,
                               strides=[1, 1, 1, 1],
                               padding="SAME")
        current = tf.nn.bias_add(current, cur_bias)
        if int(how_many_conv2d_layer) > int(layer_index):
            current = tf.nn.relu(current, name='conv2d_relu{}'.format(i))

    return current


def convnet_fcl(conv2d_out, is_input_real, weights, net_settings):
    layers = net_settings['layers']['fc_layers']
    weights = weights['fcl']
    fcl_input_size = net_settings['input_width']
    # conv2d_out_sum = tf.reduce_sum(conv2d_out[:, net_settings['frame_neighbor'], :, :],
    #                                2,
    #                                keepdims=True)
    conv2d_out_sum = conv2d_out[:, net_settings['frame_neighbor'], :, :]
    current = tf.reshape(conv2d_out_sum, [-1, fcl_input_size])
    how_mant_fcl_layers = int(layers[-1][4])
    
    for i, name in enumerate(layers):
        layer_index = name[4]

        if int(layer_index) < how_mant_fcl_layers:
            cur_weight = weights[i]['weights']
            cur_bias = weights[i]['bias']
            current = tf.matmul(current, cur_weight) + cur_bias
            current = tf.nn.relu(current, name='fcl_relu{}'.format(layer_index))

        # process real part
        elif int(layer_index) == how_mant_fcl_layers and name[6:] == 'real' and is_input_real:
            cur_weight = weights[i]['weights_real']
            cur_bias = weights[i]['bias_real']
            current = tf.matmul(current, cur_weight) + cur_bias

        # process imaginary part
        elif int(layer_index) == how_mant_fcl_layers and name[6:] == 'imag' and not is_input_real:
            cur_weight = weights[i]['weights_imag']
            cur_bias = weights[i]['bias_imag']
            current = tf.matmul(current, cur_weight) + cur_bias

        else:
            pass

    return current


def create_conv_net(spectrogram_input, additional_spectrogram_input, model_settings, is_training):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    convnet_settings = create_convnet_setting(model_settings)
    conv2d_var = create_conv2d_variables(convnet_settings)
    fcl_var = create_fcl_variables(convnet_settings)

    '''
    num_files = tf.shape(spectrogram_input)[0]

    pad_neighbor_real = pad_adjacent(spectrogram_input, model_settings)
    pad_neighbor_imag = pad_adjacent(additional_spectrogram_input, model_settings)
    pad_neighbor_concat = tf.concat([pad_neighbor_real, pad_neighbor_imag], axis=3)

    conv2d_out = convnet_conv2d(pad_neighbor_concat, conv2d_var, convnet_settings)
    [fcl_out_real, fcl_out_imag] = convnet_fcl(conv2d_out, fcl_var, convnet_settings)
    spectrogram_output_real = tf.reshape(fcl_out_real, [tf.to_int32(num_files), -1])
    spectrogram_output_imag = tf.reshape(fcl_out_imag, [tf.to_int32(num_files), -1])
    '''

    num_files = tf.shape(spectrogram_input)[0]

    pad_neighbor_real = pad_adjacent(spectrogram_input, model_settings)

    conv2d_out_real = convnet_conv2d(pad_neighbor_real, conv2d_var, convnet_settings)
    fcl_out_real = convnet_fcl(conv2d_out_real, True, fcl_var, convnet_settings)
    spectrogram_output_real = tf.reshape(fcl_out_real, [tf.to_int32(num_files), -1])

    pad_neighbor_imag = pad_adjacent(additional_spectrogram_input, model_settings)
    conv2d_out_imag = convnet_conv2d(pad_neighbor_imag, conv2d_var, convnet_settings)
    fcl_out_imag = convnet_fcl(conv2d_out_imag, False, fcl_var, convnet_settings)
    spectrogram_output_imag = tf.reshape(fcl_out_imag,
                                         [tf.to_int32(num_files), -1])

    return spectrogram_output_real, spectrogram_output_imag, dropout_prob

