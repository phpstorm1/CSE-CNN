"""
Generate output .wav data with batch. Write wav files into folders "clean", "mix", "ideal" or "estimated".
"""
import math
import numpy as np
# import matlab.engine
import warnings
import numpy as np
import python_speech_features as pysp
import scipy.io.wavfile
import os

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from scipy import signal
from tensorflow.python.ops import io_ops


def reshape_out(spectrum, model_settings):
    spectrum = np.array(spectrum)
    nDFT = int(model_settings['nDFT'] / 2) + 1
    num_frames = int(model_settings['spectrogram_length'])
    num_files = int(spectrum.size / (nDFT*num_frames))
    reshape_spectrum = np.reshape(spectrum.flatten(), [num_files, num_frames, nDFT])
    return reshape_spectrum


def rec_wav(mag_spectrum, phase_spectrum=None, additional_mag_spectrum=None, win_len=320, win_shift=160, nDFT=320, win_fun=np.hanning):
    mag = np.array(mag_spectrum)
    if phase_spectrum is not None:
        phase = np.array(phase_spectrum)
        if mag.shape != phase.shape:
            raise Exception("The shape of mag_spectrum and phase_spectrum doesn't match")
        rec_fft = np.multiply(mag, np.exp(1j*phase))
    elif additional_mag_spectrum is not None:
        fft_imag = np.array(additional_mag_spectrum)
        if mag.shape != fft_imag.shape:
            raise Exception("The shape of mag_spectrum and additional_mag_spectrum doesn't match")
        rec_fft = mag_spectrum + 1.0j * fft_imag
    else:
        raise Exception("Invalid input: both phase_spectrum and additional_mag_spectrym are missing")
    #wav_ifft = np.empty([mag.shape[0], win_len])
    wav_ifft = np.fft.irfft(a=rec_fft, n=win_len, axis=1)
    wav_deframe = pysp.sigproc.deframesig(frames=wav_ifft,
                                          siglen=0,
                                          frame_len=win_len,
                                          frame_step=win_shift,
                                          winfunc=win_fun
                                          )

    # set first frame and last frame to zeros to avoid the impulse caused by inconsistent STFT
    wav_deframe[0:win_len] = 0
    wav_deframe[-win_len:] = 0

    # clip the wav
    #wav_deframe = wav_deframe - (abs(wav_deframe) > 1) * wav_deframe

    check_nan = np.isnan(wav_deframe)
    for elem in check_nan:
        if elem:
            raise Exception("Error: NaN in wav_deframe")
    if np.max(abs(wav_deframe)) == 0:
        raise Exception("Error: zeros array for wav_deframe")
    wav_deframe = wav_deframe / np.max(abs(wav_deframe))
    return wav_deframe


def save_wav_file(filename, wav_data, sample_rate):
    """Saves audio data to .wav audio file.

    Args:
      filename: Path to save the file to.
      wav_data: Array of float PCM-encoded audio data.
      sample_rate: Samples per second to encode in the file.
    """
    scipy.io.wavfile.write(filename, sample_rate, wav_data)


def save_wav_to_path(save_path, filename, wav_data, sample_rate, wav_type):
    save_path = os.path.join(save_path, wav_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    prefix_filename = filename.split(".wav")[0]
    if wav_type == "mix":
        fullname = prefix_filename + "_mix.wav"
    elif wav_type == "clean":
        fullname = prefix_filename + "_clean.wav"
    elif wav_type == "ideal":
        fullname = prefix_filename + "_ideal.wav"
    elif wav_type == "estimated":
        fullname = prefix_filename + "_estimated.wav"
    else:
        raise Exception("Unrecognized wav_type " + wav_type)

    fullname = os.path.join(save_path, fullname)
    save_wav_file(fullname, wav_data, sample_rate)


def save_batch_to_path(batch_mag_spectrum, saved_batch_size, wav_type, model_settings, batch_phase_spectrum=None, additional_mag_spectrum=None):
    """Recontruction a batch of wav files from a batch of magnitude and phase spectrum.
     Save the batch to given directory.

     Args:
        batch_mag_spectrum: 2D array of magntiude spectrum with size [num_wav, (num_frames*len_frame)], note that
        					num_wav may be (num_files*num_snr*num_noise) for multi-snr & noise training
        batch_phase_spectrum: 2D array of phase spectrum with same size as batch_mag_spectrum
        saved_batch_size: size of the past saved batch. Note that it is NOT the number of wav files saved when using
        			multi-snr & noise training. Instead it is (num_files_saved*num_snr*num_noise).
        wav_type: Indicating the type of wav wish to generate. Should be one of "mix", "clean", "ideal" or "estimated"
        model_settings: model_settings used in input_data.py
    """
    nDFT = int(model_settings['nDFT'] / 2) + 1
    num_frames = model_settings['spectrogram_length']
    num_files = batch_mag_spectrum.size / (nDFT*num_frames)

    if batch_phase_spectrum is not None:
        batch_mag_spectrum = np.array(batch_mag_spectrum)
        batch_phase_spectrum = np.array(batch_phase_spectrum)
        if batch_mag_spectrum.shape != batch_phase_spectrum.shape:
            raise Exception("The shapes of batch_mag_spectrum and batch_phase_spectrum don't match \n"+
                            "batch_mag_spectrum.shape: " + str(batch_mag_spectrum.shape) +
                            "\nbatch_phase_spectrum.shape" + str(batch_phase_spectrum.shape))
        batch_mag_spectrum = reshape_out(batch_mag_spectrum, model_settings)
        batch_phase_spectrum = reshape_out(batch_phase_spectrum, model_settings)
    elif additional_mag_spectrum is not None:
        batch_mag_spectrum = np.array(batch_mag_spectrum)
        batch_imag_spectrum = np.array(additional_mag_spectrum)
        if batch_mag_spectrum.shape != batch_imag_spectrum.shape:
            raise Exception("The shapes of batch_mag_spectrum and additional_mag_spectrum don't match \n"+
                            "batch_mag_spectrum.shape: " + str(batch_mag_spectrum.shape) +
                            "\nadditional_mag_spectrum.shape" + str(batch_imag_spectrum.shape))
        batch_mag_spectrum = reshape_out(batch_mag_spectrum, model_settings)
        batch_imag_spectrum = reshape_out(batch_imag_spectrum, model_settings)
    else:
        raise Exception("Invalid input: both phase_spectrum and additional_mag_spectrym are missing")

    for i in range(int(num_files)):
        single_mag_spectrum = batch_mag_spectrum[i]
        if batch_phase_spectrum is not None:
            single_phase_spectrum = batch_phase_spectrum[i]
            single_wav = rec_wav(mag_spectrum=single_mag_spectrum, phase_spectrum=single_phase_spectrum,
                                 win_len=model_settings['win_len'], win_shift=model_settings['win_shift'],
                                 nDFT=model_settings['nDFT'], win_fun=model_settings['win_fun'])
        else:
            single_imag_spectrum = batch_imag_spectrum[i]
            single_wav = rec_wav(mag_spectrum=single_mag_spectrum, additional_mag_spectrum=single_imag_spectrum,
                                 win_len=model_settings['win_len'], win_shift=model_settings['win_shift'],
                                 nDFT=model_settings['nDFT'], win_fun=model_settings['win_fun'])
        idx = saved_batch_size + i
        full_file_name = "test_" + str(idx) + ".wav"
        save_wav_to_path(save_path=model_settings['save_path'], filename=full_file_name, wav_data=single_wav,
                         sample_rate=model_settings['sample_rate'], wav_type=wav_type)

def save_testing_file(test_out_real, test_out_imag, nDFT, win_len, win_shift, fs, save_path, wav_type, cur_snr, cur_noise, filename):
    test_out_real = np.reshape(test_out_real, [-1, int(nDFT/2)+1])
    test_out_imag = np.reshape(test_out_imag, [-1, int(nDFT/2)+1])
    wav = rec_wav(mag_spectrum=test_out_real,
                  additional_mag_spectrum=test_out_imag,
                  win_len=win_len,
                  win_shift=win_shift,
                  nDFT=nDFT)
    cur_save_path = os.path.join(save_path, wav_type, str(cur_snr), cur_noise)
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)
    cur_save_path = os.path.join(cur_save_path, filename + '.wav')
    save_wav_file(cur_save_path, wav, fs)
