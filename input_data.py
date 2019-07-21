from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import python_speech_features

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
NOISE_DIR_NAME = 'noise'
SPEECH_DIR_NAME = 'clean_speech'
RANDOM_SEED = 59185


def get_spectrum(wav, win_len=320, win_shift=160, nDFT=320, win_fun=np.hanning):
    """Get the spectrogram for given signal. lib 'python_speech_features' is required.
    :param wav: 1-D signal
    :param win_len: length of the signal, in samples
    :param win_shift: non-overlap portion, in samples
    :param win_fun: window function for framing, default is np.hanning
    :param mode: spectrogram for 'magnitude' or 'phase'
    :return: a matrix of size NFRAME*(NFFT/2+1), each row is the spectrum of the corresponding frame
    """

    wav_np = np.array(wav).flatten()
    wav_np = np.reshape(wav_np, [len(wav_np)])
    # flatten the vector
    #wav = tf.reshape(wav, [-1])

    # convert int32 to int
    win_len = int(win_len)
    win_shift = int(win_shift)
    nDFT = int(nDFT)

    wav_frame = python_speech_features.sigproc.framesig(sig=wav_np,
                                                        frame_len=win_len,
                                                        frame_step=win_shift,
                                                        winfunc=win_fun
                                                        )
    wav_fft = np.empty([wav_frame.shape[0], int(win_len / 2 + 1)], dtype=complex)
    for frame in range(wav_frame.shape[0]):
        wav_fft[frame] = np.fft.rfft(a=wav_frame[frame], n=nDFT)
    mag_spectrum = np.abs(wav_fft)
    phase_spectrum = np.arctan2(wav_fft.imag, wav_fft.real)

    return mag_spectrum, phase_spectrum, wav_fft.real, wav_fft.imag


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def load_wav_file(filename):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
      filename: Path to the .wav file to load.

    Returns:
      Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
        return sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio sample data to a .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: 2D array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    with tf.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.placeholder(tf.string, [])
        sample_rate_placeholder = tf.placeholder(tf.int32, [])
        wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
        wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                               sample_rate_placeholder)
        wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
        sess.run(
            wav_saver,
            feed_dict={
                wav_filename_placeholder: filename,
                sample_rate_placeholder: sample_rate,
                wav_data_placeholder: np.reshape(wav_data, (-1, 1))
            })


class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_dir, validation_percentage,
                 testing_percentage, model_settings):
        self.data_dir = data_dir
        self.noise_type = model_settings['noise_type']
        self.prepare_data_index(validation_percentage, testing_percentage)
        self.prepare_noise_data()
        self.prepare_processing_graph(model_settings)

    def prepare_data_index(self, validation_percentage, testing_percentage):
        """Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with its file name as labels.
        This function analyzes the wave files below the `data_dir\\SPEECH_DIR_NAME`,
        and uses a stable hash to assign it to a data set partition(validation,
        testing, training).

        Args:
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.

        Raises:
          Exception: If the folder containing speech files doesn't exist.
          Exception: If no .wav files are found in the folder speficified.
          Exception: If any set for training, testing or validation is empty.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        speech_dir = os.path.join(self.data_dir, SPEECH_DIR_NAME)

        if not os.path.exists(speech_dir):
            raise Exception("The folder containing speech doesn't exists")
        search_path = os.path.join(self.data_dir, SPEECH_DIR_NAME, '*.wav')

        for wav_path in gfile.Glob(search_path):
            # _, word = os.path.split(os.path.dirname(wav_path))
            # word = word.lower()
            basename = os.path.basename(wav_path)
            filename = os.path.splitext(basename)[0]
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            self.data_index[set_index].append({'file': wav_path, 'filename': filename})

        # Make sure all sets aren't empty and the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            if not self.data_index[set_index]:
                raise Exception('Set ' + set_index + " is empty")
            random.shuffle(self.data_index[set_index])

    def prepare_noise_data(self):
        """Searches a folder for noise audio, and loads it into memory.

        It's expected that the background audio samples will be in a subdirectory
        named 'noise' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.

        If the 'noise' folder doesn't exist, or the noise couldn't be found in the
        folder, the function will throw an error.

        NOTE: the sampling rate of noise audio should be the same as speech audio.
              noise and speech audio should be noralmized to [-1, 1]

        Returns:
          List of raw PCM-encoded audio samples of background noise.

        Raises:
          Exception: If the folder doesn't exist.
          Exception: If noise files aren't found in the folder.
        """
        self.noise_data = []
        noise_dir = os.path.join(self.data_dir, NOISE_DIR_NAME)
        if not os.path.exists(noise_dir):
            raise Exception("The folder containing noise files doesn't exist")
        noise_names = self.noise_type
        if not noise_names:
            raise Exception("Must specificify at least one type of noise")
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)

            # decoding result will be padded with zero if the original length is shorter than the desired length
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            for noise_name in noise_names:
                search_path = os.path.join(self.data_dir, NOISE_DIR_NAME, (noise_name
                                                                           + ".wav"))
                if not os.path.isfile(search_path):
                    raise Exception('No wav file found for ' + noise_name)
                wav_data = sess.run(
                    wav_decoder,
                    feed_dict = {wav_filename_placeholder:search_path}).audio.flatten()
                # form a with np.shape (1,n) instead of (n,) so it could be appended to the list
                #wav_data = np.reshape(wav_data, [1, len(wav_data)])
                self.noise_data.append([wav_data])

    def prepare_processing_graph(self, model_settings):
        """Builds a TensorFlow graph to apply the input distortions.

        Creates a graph that loads a .wav file, decodes it, add the noise by given SNR,
        calculates spectrograms as input feature and output labels that feed to neural
        nets. If noise is shorter than speech, the noise will be repeated.

        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:

          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - noise_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.

        Args:
          model_settings: Information about the current model being trained.
        """
        desired_samples = model_settings['desired_samples']
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [], 'wav_filename_placeholder_')
        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)

        audio = tf.reshape(wav_decoder.audio, [-1])
        #audio = tf.zeros(shape=[90000])
        speech_square = tf.square(audio)
        speech_energy = tf.reduce_sum(speech_square)

        # Mix in background noise.
        self.noise_data_placeholder_ = tf.placeholder(dtype=tf.float32, name='noise_data_ph_')

        # for test only
        #self.noise_data_placeholder_ = tf.zeros(shape=[40000])
        self.snr = tf.placeholder(dtype=tf.float32, name='snr')

        # extend the noise if it is shorter than speech
        def cond(i, extend_times, extend_noise):
            return tf.greater(extend_times, i)

        def body(i, extend_times, extend_noise):
            return i+1, extend_times, extend_noise.write(i, self.noise_data_placeholder_)

        extend_times = tf.to_int32(tf.ceil(tf.size(audio)
                                           /tf.size(self.noise_data_placeholder_)))
        extend_noise = tf.TensorArray(tf.float32, tf.to_int32(extend_times))

        #i = tf.Variable(initial_value=0, dtype=tf.int32)
        #tmp_test = tf.identity(self.noise_data_placeholder_)
        #temp_test = self.noise_data_placeholder_

        i, extend_times, extend_noise = tf.while_loop(cond=cond, body=body,
                                                      loop_vars=[0, extend_times, extend_noise])

        extend_noise = tf.reshape(extend_noise.stack(), [-1])

        # randomly choose a part of the noise
        start_noise_point = tf.random_uniform(shape=[1,1], maxval=(tf.size(extend_noise)
                                                                   - tf.size(audio)), dtype=tf.int32)

        #  start_noise_point + tf.size(audio)]
        start_noise_point = tf.reshape(start_noise_point, [-1])
        noise_rand = tf.slice(input_=extend_noise, begin=start_noise_point,
                              size=[tf.size(audio)])

        noise_square = tf.square(noise_rand)
        noise_energy = tf.reduce_sum(noise_square)

        energy_ratio = tf.sqrt(speech_energy/(noise_energy*(10**(self.snr/10))))

        mix_ = tf.multiply(noise_rand, energy_ratio)
        mix_ = tf.add(mix_, audio)

        # normalize to [-1, 1]
        mix_max = tf.reduce_max(mix_)
        mix_ = tf.divide(mix_, mix_max)

        # get the spectrogram for clean speech and noisy speech
        '''
        self.spectrogram_mix = get_spectrum(tf_wav=mix_, win_len=model_settings['win_len'], 
                    win_shift=model_settings['win_shift'], nDFT=model_settings['nDFT'])
        self.spectrogram_speech = get_spectrum(tf_wav=audio, win_len=model_settings['win_len'],
                    win_shift=model_settings['win_shift'], nDFT = model_settings['nDFT'])
        '''
        tensor_win_len = tf.Variable(initial_value=model_settings['win_len'])
        tensor_win_shift = tf.Variable(initial_value=model_settings['win_shift'])
        tensor_nDFT = tf.Variable(initial_value=model_settings['nDFT'])
        #tensor_mode = tf.Variable(initial_value='phase', dtype=tf.string)

        # magnitude spectrograms
        self.spectrogram_mix, self.phase_spectrogram_mix, self.mix_fft_real, self.mix_fft_imag = tf.py_func(
                func=get_spectrum, inp=[mix_, tensor_win_len, tensor_win_shift, tensor_nDFT],
                Tout=[tf.float64, tf.float64, tf.float64, tf.float64])

        self.spectrogram_speech, self.phase_spectrogram_speech, self.speech_fft_real, self.speech_fft_imag = tf.py_func(
                func=get_spectrum, inp=[audio, tensor_win_len, tensor_win_shift, tensor_nDFT],
                Tout=[tf.float64, tf.float64, tf.float64, tf.float64])


        # phase spectrograms for reconstruction
        '''
        self.phase_spectrogram_mix = tf.py_func(func=get_spectrum, inp=[mix_, tensor_win_len,
                tensor_win_shift, tensor_nDFT, 'phase'], Tout=tf.float64)
        self.phase_spectrogram_speech = tf.py_func(func=get_spectrum, inp=[audio, tensor_win_len,
                tensor_win_shift, tensor_nDFT, 'phase'], Tout=tf.float64)
        '''
        #self.spectrogram_mix = tf.py_func(func=get_spectrum, inp=[mix_], Tout=tf.float64)
        #self.spectrogram_speech = tf.py_func(func=get_spectrum, inp=[audio], Tout=tf.float64)
    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, snr, noise_data, model_settings, mode, sess):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of label indexes
        """
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        num_row = sample_count * len(snr) * len(noise_data)

        data_fft_real = np.zeros((num_row, model_settings['spectrogram_size']))
        data_fft_imag = np.zeros((num_row, model_settings['spectrogram_size']))
        label_fft_real = np.zeros((num_row, model_settings['spectrogram_size']))
        label_fft_imag = np.zeros((num_row, model_settings['spectrogram_size']))

        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
            }

            # generate samples for every single snr and noise type
            for idx_snr, snr in enumerate(snr):
                # in case that integer snr is input
                snr = np.float32(snr)
                for idx_noise, noise in enumerate(noise_data):
                    noise_reshape = noise[0].reshape([len(noise[0]), 1])
                    #print(noise_reshape.shape)
                    input_dict[self.noise_data_placeholder_] = noise_reshape
                    input_dict[self.snr] = snr
                    # run the graph to get the features and labels
                    spectrogram_speech, phase_speech, speech_fft_real, speech_fft_imag = sess.run(
                        [self.spectrogram_speech, self.phase_spectrogram_speech, self.speech_fft_real, self.speech_fft_imag],
                        feed_dict=input_dict)
                    spectrogram_mix, phase_mix, mix_fft_real, mix_fft_imag = sess.run(
                        [self.spectrogram_mix, self.phase_spectrogram_mix, self.mix_fft_real, self.mix_fft_imag],
                        feed_dict=input_dict)
                    idx = i - offset + sample_count * (idx_snr * len(self.noise_data) + idx_noise)

                    data_fft_real[idx, :] = mix_fft_real.flatten()
                    data_fft_imag[idx, :] = mix_fft_imag.flatten()
                    label_fft_real[idx, :] = speech_fft_real.flatten()
                    label_fft_imag[idx, :] = speech_fft_imag.flatten()

        if mode != 'testing':
            return data_fft_real, label_fft_real, data_fft_imag, label_fft_imag
        else:
            return data_fft_real, label_fft_real, data_fft_imag, label_fft_imag, sample['filename']
