from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import tensorflow as tf
import input_data
import models
import output_data
from scipy.io import wavfile


FLAGS = None


def main(_):

    sess = tf.InteractiveSession()

    root_data_path = FLAGS.data_dir

    model_settings = models.prepare_model_settings(
        sample_rate=FLAGS.sample_rate,
        clip_duration_ms=-1,
        win_len=FLAGS.win_len,
        win_shift=FLAGS.win_shift,
        nDFT=FLAGS.nDFT,
        frame_neighbor=FLAGS.frame_neighbor,
        snr=None,
        noise_type=None,
        data_dir=None,
        save_path=None,
        model_architecture=FLAGS.model_architecture)

    spectrogram_input = tf.placeholder(
        tf.float32, name='spectrogram_input')

    additional_spectrogram_input = tf.placeholder(
        tf.float32, name='additional_spectrogram_input')

    model_out_real, model_out_imag, dropout_prob = models.create_model(
                                                        spectrogram_input=spectrogram_input,
                                                        model_settings=model_settings,
                                                        is_training=True,
                                                        additonal_spectrogram_input=additional_spectrogram_input)

    models.load_variables_from_checkpoint(sess, FLAGS.load_from_checkpoint)

    snr = [sub_dir for sub_dir in os.listdir(root_data_path)
           if os.path.isdir(os.path.join(root_data_path, sub_dir))]

    if not snr:
        raise Exception("Cannot find sub-dirs in data_dir")

    noise = [sub_dir for sub_dir in os.listdir(os.path.join(root_data_path, snr[0]))
             if os.path.isdir(os.path.join(root_data_path, snr[0], sub_dir))]

    if not noise:
        raise Exception("Cannot find sub-dirs in snr")

    for each_snr in snr:
        print("current snr: " + each_snr)
        for each_noise in noise:
            print("current noise: " + each_noise)
            read_path = os.path.join(root_data_path, each_snr, each_noise)
            for file in os.listdir(read_path):
                if not file.endswith(".wav"):
                    continue
                print("processing file: " + file)
                fs, mix_wav = wavfile.read(os.path.join(read_path, file))
                [_, _, mix_spectrogram_real, mix_spectrogram_imag] = input_data.get_spectrum(wav=mix_wav,
                                                                                             win_len=FLAGS.win_len,
                                                                                             win_shift=FLAGS.win_shift,
                                                                                             nDFT=FLAGS.nDFT)
                num_frames = mix_spectrogram_real.shape[0]
                mix_spectrogram_real = np.reshape(mix_spectrogram_real, [1, -1])
                mix_spectrogram_imag = np.reshape(mix_spectrogram_imag, [1, -1])

                test_out_real, test_out_imag = sess.run(
                    [model_out_real, model_out_imag],
                    feed_dict={
                        spectrogram_input: mix_spectrogram_real,
                        additional_spectrogram_input: mix_spectrogram_imag,
                        dropout_prob: 0.0
                    })

                test_out_real = np.reshape(test_out_real, [num_frames, -1])
                test_out_imag = np.reshape(test_out_imag, [num_frames, -1])
                rec_wav = output_data.rec_wav(mag_spectrum=test_out_real,
                                              additional_mag_spectrum=test_out_imag,
                                              win_len=FLAGS.win_len,
                                              win_shift=FLAGS.win_shift,
                                              nDFT=FLAGS.nDFT)

                save_path = os.path.join(FLAGS.save_path, each_snr, each_noise)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                comp_save_path = os.path.join(save_path, file)
                output_data.save_wav_file(comp_save_path, rec_wav, fs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default='/coding/PycharmProjects/data/mix',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--save_path',
        type=str,
        default='/coding/PycharmProjects/data/est',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--win_len',
        type=int,
        default=500,
        help='Length of the window',)
    parser.add_argument(
        '--win_shift',
        type=int,
        default=250,
        help='Length of windor shift',)
    parser.add_argument(
        '--nDFT',
        type=int,
        default=500,
        help='Number of points for frequency analysis',)
    parser.add_argument(
        '--frame_neighbor',
        type=int,
        default=3,
        help='Number of neighbor frames for stacking',)
    parser.add_argument(
        '--load_from_checkpoint',
        type=str,
        default='/tmp/CSE-CNN/train/cs-cnn.ckpt-20000',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='cs-cnn',
        help='What model architecture to use')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
