# TODO use a json parser to import training and network settings
# TODO use two label sets - real magnitude and the difference between real and imaginary magnitude

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
import output_data
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):

    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    model_settings = models.prepare_model_settings(
        sample_rate=FLAGS.sample_rate, clip_duration_ms=FLAGS.clip_duration_ms,
        win_len=FLAGS.win_len, win_shift=FLAGS.win_shift, nDFT=FLAGS.nDFT,
        frame_neighbor=FLAGS.frame_neighbor, snr=FLAGS.snr, noise_type=FLAGS.noise_type,
        data_dir=FLAGS.data_dir, save_path=FLAGS.save_path,
        model_architecture=FLAGS.model_architecture)
    audio_processor = input_data.AudioProcessor(
        data_dir=FLAGS.data_dir,
        validation_percentage=FLAGS.validation_percentage,
        testing_percentage=FLAGS.testing_percentage,
        model_settings=model_settings)
    spectrogram_size = model_settings['spectrogram_size']
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))

    spectrogram_input = tf.placeholder(
        tf.float32, [None, spectrogram_size], name='spectrogram_input')

    additional_spectrogram_input = tf.placeholder(
        tf.float32, [None, spectrogram_size], name='additional_spectrogram_input')

    model_out_real, model_out_imag, dropout_prob = models.create_model(
                                                        spectrogram_input=spectrogram_input,
                                                        model_settings=model_settings,
                                                        is_training=True,
                                                        additonal_spectrogram_input=additional_spectrogram_input)
    model_out = tf.concat(values=[model_out_real, model_out_imag], axis=1)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, name='groundtruth_input')

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    # control_dependencies = []
    # if FLAGS.check_nans:
    #     checks = tf.add_check_numerics_ops()
    #     control_dependencies = [checks]

    # Create the back propagation and training evaluation machinery in the graph.
    Adam_lr = tf.placeholder(tf.float32)
    with tf.name_scope('mean_squared_error'):
        mse = tf.losses.mean_squared_error(
                labels=ground_truth_input, predictions=model_out)
        tf.summary.scalar('mean_squared_error', mse)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # control_dependencies = extra_update_ops
    with tf.name_scope('train'), tf.control_dependencies(extra_update_ops):
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(Adam_lr, step, 1, 0.999885)
        adam = tf.train.AdamOptimizer(rate)
        train_step = adam.minimize(mse, global_step=step)
        tf_print = tf.Print(rate, [rate, step, mse])


    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    tf_print_global_step = tf.Print(global_step, [global_step])

    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in range(start_step, int(training_steps_max + 1)):

        '''
        Pull the audio samples we'll use for training. phase is useless for training
        To support multi-snr and multi-noise training with cnn, consider training with one type of noise and one snr
        only to minimize the GPU memory it will use
        For DNN, GPU memory shouldn't be a serious issue, so one could use the old version of this code

        train_fft_real, label_fft_real, train_fft_imag, label_fft_imag = audio_processor.get_data(
                                                    FLAGS.batch_size, 0, model_settings, 'training', sess)
        '''

        # training_mse = 0

        snr_idx = training_step % len(model_settings['snr'])
        noise_idx = training_step % len(model_settings['noise_type'])

        train_fft_real, label_fft_real, train_fft_imag, label_fft_imag = audio_processor.get_data(
                                                    how_many=FLAGS.batch_size,
                                                    offset=0,
                                                    snr=[model_settings['snr'][snr_idx]],
                                                    noise_data=[audio_processor.noise_data[noise_idx]],
                                                    model_settings=model_settings,
                                                    mode='training',
                                                    sess=sess)

        # print("size of one input: ", train_fft_imag.shape)
        # Run the graph with this batch of training data.

        train_ground_truth = np.concatenate((label_fft_real, label_fft_imag), axis=1)
        train_summary, training_mse, _ = sess.run(
                    [
                        merged_summaries, mse, train_step
                    ],
                    feed_dict={
                        spectrogram_input: train_fft_real,
                        additional_spectrogram_input: train_fft_imag,
                        ground_truth_input: train_ground_truth,
                        Adam_lr: FLAGS.Adam_learn_rate,
                        dropout_prob: 0.0
                    })

                # training_mse = training_mse + partial_training_mse / (len(model_settings['snr']) * len(model_settings['noise_type']))

        # sess.run([increment_global_step], feed_dict={})

        tf.logging.info('Step #%d: Training mse %.6f' %
                                (training_step, training_mse))

        train_writer.add_summary(train_summary, training_step)

        is_last_step = (training_step == training_steps_max)
        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            set_size = audio_processor.set_size('validation')

            total_validation_mse = 0
            for i in range(0, set_size, FLAGS.batch_size):
                # validation_fft_real, validation_fft_real_truth, validation_fft_imag, validation_fft_imag_truth = (
                #                 audio_processor.get_data(FLAGS.batch_size, i, model_settings, 'validation', sess))
                batch_validation_mse = 0
                for snr_idx in range(len(model_settings['snr'])):
                    for noise_idx in range(len(model_settings['noise_type'])):

                        validation_fft_real, validation_fft_real_truth, validation_fft_imag, validation_fft_imag_truth = (
                                        audio_processor.get_data(how_many=FLAGS.batch_size,
                                                                 offset=i,
                                                                 snr=[model_settings['snr'][snr_idx]],
                                                                 noise_data=[audio_processor.noise_data[noise_idx]],
                                                                 model_settings=model_settings,
                                                                 mode='validation',
                                                                 sess=sess))

                        # Run a validation step and capture training summaries for TensorBoard
                        # with the `merged` op.
                        validation_ground_truth = np.concatenate((validation_fft_real_truth, validation_fft_imag_truth), axis=1)
                        validation_summary, validation_mse = sess.run(
                                    [merged_summaries, mse],
                                    feed_dict={
                                        spectrogram_input: validation_fft_real,
                                        additional_spectrogram_input: validation_fft_imag,
                                        ground_truth_input: validation_ground_truth,
                                        Adam_lr: FLAGS.Adam_learn_rate,
                                        dropout_prob: 0.0
                                    })
                        batch_validation_mse += validation_mse / (len(model_settings['snr'])*len(model_settings['noise_type']))

                batch_size = min(FLAGS.batch_size, set_size - i)
                total_validation_mse += (batch_validation_mse * batch_size) / set_size

            tf.logging.info('Step %d: Validation mse = %.6f (N=%d)' %
                                (training_step, total_validation_mse, set_size))

            validation_writer.add_summary(validation_summary, training_step)

        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            checkpoint_path = os.path.join(FLAGS.train_dir,
                                           FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_mse = 0
    additional_total_mse = 0
    for i in xrange(0, set_size, FLAGS.batch_size):


        batch_testing_mse = 0
        for snr_idx in range(len(model_settings['snr'])):
            for noise_idx in range(len(model_settings['noise_type'])):

                test_fft_real, test_fft_real_truth, test_fft_imag, test_fft_imag_truth, filename = (
                        audio_processor.get_data(how_many=FLAGS.batch_size,
                                                 offset=i,
                                                 snr=[model_settings['snr'][snr_idx]],
                                                 noise_data=[audio_processor.noise_data[noise_idx]],
                                                 model_settings=model_settings,
                                                 mode='testing',
                                                 sess=sess))

                test_ground_truth = np.concatenate((test_fft_real_truth, test_fft_imag_truth), axis=1)
                test_out_real, test_out_imag, testing_mse = sess.run(
                            [model_out_real, model_out_imag, mse],
                            feed_dict={
                                spectrogram_input: test_fft_real,
                                additional_spectrogram_input: test_fft_imag,
                                ground_truth_input: test_ground_truth,
                                dropout_prob: 0.0
                            })
                batch_testing_mse += testing_mse / (len(model_settings['snr'])*len(model_settings['noise_type']))

                # input_fft_real = output_data.reshape_out(test_fft_real, model_settings)
                # input_fft_imag = output_data.reshape_out(test_fft_imag, model_settings)
                # label_fft_real = output_data.reshape_out(test_fft_real_truth, model_settings)
                # label_fft_imag = output_data.reshape_out(test_fft_imag_truth, model_settings)
                # output_fft_real = output_data.reshape_out(test_out_real, model_settings)
                # output_fft_imag = output_data.reshape_out(test_out_imag, model_settings)
                #
                # save_idx = i * len(FLAGS.snr) * len(FLAGS.noise_type) + snr_idx *len(model_settings['noise_type']) + noise_idx

                # save mix .wav files
                # output_data.save_batch_to_path(batch_mag_spectrum=input_fft_real,
                #                                 additional_mag_spectrum=input_fft_imag,
                #                                 saved_batch_size=save_idx,
                #                                 wav_type='mix',
                #                                 model_settings=model_settings)
                #
                # # save clean .wav files
                # output_data.save_batch_to_path(batch_mag_spectrum=label_fft_real,
                #                                 additional_mag_spectrum=label_fft_imag,
                #                                 saved_batch_size=save_idx,
                #                                 wav_type='clean',
                #                                 model_settings=model_settings)
                #
                # # save ideal .wav files
                # output_data.save_batch_to_path(batch_mag_spectrum=label_fft_real,
                #                                 additional_mag_spectrum=output_fft_imag,
                #                                 saved_batch_size=save_idx,
                #                                 wav_type='ideal',
                #                                 model_settings=model_settings)
                #
                # # save estimated .wav files
                # output_data.save_batch_to_path(batch_mag_spectrum=output_fft_real,
                #                                 additional_mag_spectrum=output_fft_imag,
                #                                 saved_batch_size=save_idx,
                #                                 wav_type='estimated',
                #                                 model_settings=model_settings)

                output_data.save_testing_file(test_out_real=test_fft_real,
                                              test_out_imag=test_fft_imag,
                                              nDFT=FLAGS.nDFT,
                                              win_len=FLAGS.win_len,
                                              win_shift=FLAGS.win_shift,
                                              fs=FLAGS.sample_rate,
                                              save_path=FLAGS.save_path,
                                              wav_type="mix",
                                              cur_snr=model_settings['snr'][snr_idx],
                                              cur_noise=model_settings['noise_type'][noise_idx],
                                              filename=filename)

                output_data.save_testing_file(test_out_real=test_fft_real_truth,
                                              test_out_imag=test_fft_imag_truth,
                                              nDFT=FLAGS.nDFT,
                                              win_len=FLAGS.win_len,
                                              win_shift=FLAGS.win_shift,
                                              fs=FLAGS.sample_rate,
                                              save_path=FLAGS.save_path,
                                              wav_type="clean",
                                              cur_snr=model_settings['snr'][snr_idx],
                                              cur_noise=model_settings['noise_type'][noise_idx],
                                              filename=filename)

                output_data.save_testing_file(test_out_real=test_out_real,
                                              test_out_imag=test_out_imag,
                                              nDFT=FLAGS.nDFT,
                                              win_len=FLAGS.win_len,
                                              win_shift=FLAGS.win_shift,
                                              fs=FLAGS.sample_rate,
                                              save_path=FLAGS.save_path,
                                              wav_type="estimated",
                                              cur_snr=model_settings['snr'][snr_idx],
                                              cur_noise=model_settings['noise_type'][noise_idx],
                                              filename=filename)

        batch_size = min(FLAGS.batch_size, set_size - i)

        total_mse += (batch_testing_mse * batch_size) / set_size

    tf.logging.info('Final test mse = %.6f (N=%d)' % (total_mse, set_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default='.\\data\\16kHz',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--save_path',
        type=str,
        default='.\\data\\test',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--noise_type',
        type=str,
        default=['babble', 'factory', 'street'],
        help="""\
      The type of noise for training.
      """)
    parser.add_argument(
        '--snr',
        type=float,
        default=[-5, 0, 5, 10],
        help="""\
      SNR to mix the noises.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=2,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=2000,
        help='Expected duration in milliseconds of the wavs',)
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
        '--how_many_training_steps',
        type=str,
        default='25000',
        help='How many training loops to run',)
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=500,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='How many items to train with at once',)
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./tmp/CSE-CNN/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./tmp/CSE-CNN/train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=1000,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='cs-cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--Adam_learn_rate',
        type=float,
        default=1e-3,
        help='Learning rate for Adam'
    )
    parser.add_argument(
        '--normLabel',
        type=bool,
        default=False,
        help='Learning rate for Adam'
    )
    parser.add_argument(
        '--logLabel',
        type=bool,
        default=False,
        help='Learning rate for Adam'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
