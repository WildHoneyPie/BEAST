from spleeter.audio.adapter import AudioAdapter
# from spleeter.audio import Codec, STFTBackend
# from spleeter.separator import Separator
# from spleeter.audio import STFTBackend

from librosa.feature import melspectrogram

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import librosa
import numpy as np 
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='-1'

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, Processor, SequentialProcessor

global_sr = 44100

### calculating filtered spectrograms and first order derivative using Madmom API
def madmom_feature(wav):
    """ returns the madmom features mentioned in the paper"""
    sig = SignalProcessor(num_channels=1, sample_rate=global_sr )
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    feature = pre_processor.process( wav)
    return feature

def get_wav(audio_file):
    r"""Returns a numpy array of the audio section at the sampling rate
    determined by the `constants` module."""

    wav = librosa.load(audio_file, sr= global_sr)[0]
    return wav



class prepare_spectrgram_data(object):
    def __init__(self):
        super(prepare_spectrgram_data, self).__init__()
        self.data_dir = {
            'ballroom': '/media/ccbs1012/A4A079C8A079A20A/data_beattracking/Ballroom/BallroomData/',
            'carnetic': '/media/ccbs1012/A4A079C8A079A20A/data_beattracking/CMR_full_dataset/audio/',
            'gtzan': '/home/ccbs1012/dataset/data_beattracking/gtzan/audio', 
            'hainsworth': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/Hainsworth/wavs/',
            'smc': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/SMC_MIREX/SMC_MIREX_Audio/',
            'harmonix': '/data1/zhaojw/dataset/Harmonix/audio/',
            'beatles': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/beatles/'
        }
        self.beat_annotation_dir = {
            'ballroom': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/ISMIR2019/ballroom/annotations/beats/',
            'carnetic': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/CMR_full_dataset_1.0/annotations/beats/',
            'gtzan': '/home/ccbs1012/dataset/dataset_2/ISMIR2019/gtzan/annotations/beats/',
            'hainsworth': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/ISMIR2019/hainsworth/annotations/beats/',
            'smc': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/ISMIR2019/smc/annotations/beats/',
            'harmonix': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/harmonixset/dataset/beats_and_downbeats/',
            'beatles': '/media/ccbs1012/A4A079C8A079A20A/dataset_2/ISMIR2019/beatles/annotations/beats/'
        }

        self.SR = 44100
        self.n_fft = 4096
        self.hop_length = 1024
        self.n_mels = 128


        self.audio_loader = AudioAdapter.default()


        # print('initailize carnetic ...')
        # carnetic_data, carnetic_annotation, not_found = self.initialize_carnetic()
        # print(carnetic_data.shape, carnetic_annotation.shape)
        # print('annotation not found:', not_found, '\n')
        # np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_carnetic.npy', carnetic_data, allow_pickle=True)

        print('initailize gtzan ...')
        gtzan_data, gtzan_annotation, not_found = self.initialize_gtzan()
        print(gtzan_data.shape, gtzan_annotation.shape)
        print('annotation not found:', not_found, '\n')
        np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_gtzan.npy', gtzan_data, allow_pickle=True)

        # print('initailize hainsworth ...')
        # hainsworth_data, hainsworth_annotation, not_found = self.initialize_hainsworth()
        # print(hainsworth_data.shape, hainsworth_annotation.shape)
        # print('annotation not found:', not_found, '\n')
        # np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_hainsworth.npy', hainsworth_data, allow_pickle=True)

        # print('initailize smc ...')
        # smc_data, smc_annotation, not_found = self.initialize_smc()
        # print(smc_data.shape, smc_annotation.shape)
        # print('annotation not found:', not_found, '\n')
        # np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_smc.npy', smc_data, allow_pickle=True)

        # print('initailize beatles ...')
        # beatles_data, beatles_annotation, not_found = self.initialize_beatles()
        # print(beatles_data.shape, beatles_annotation.shape)
        # print('annotation not found:', not_found, '\n')
        # np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_beatles.npy', beatles_data, allow_pickle=True)


        # print('initailize ballroom ...')
        # ballroom_data, ballroom_annotation, not_found = self.initialize_ballroom()
        # print(ballroom_data.shape, ballroom_annotation.shape)
        # print('annotation not found:', not_found, '\n')
        # np.save('/media/ccbs1012/A4A079C8A079A20A/audio_128/linear_spectrogram_ballroom.npy', ballroom_data, allow_pickle=True)


        np.savez_compressed('/media/ccbs1012/A4A079C8A079A20A/audio_128/gtzan_data.npz', gtzan=gtzan_data)


        np.savez_compressed('/media/ccbs1012/A4A079C8A079A20A/audio_128/gtzan_beat_annotation.npz', gtzan = gtzan_annotation)


    def initialize_ballroom(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['ballroom']
        annotation_dir = self.beat_annotation_dir['ballroom']
        # audio_list = []
        for gnere in tqdm(os.listdir(data_dir)):
            if gnere[0].isupper():
                gnere_dir = os.path.join(data_dir, gnere)
                for audio_name in os.listdir(gnere_dir):
                    #load audio
                    audio_dir = os.path.join(gnere_dir, audio_name)
                    # audio_list.append(audio_dir+'\n')
                    # print(audio_list)
                    waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
                    #load beat annotations
                    beat_dir = os.path.join(annotation_dir, 'ballroom_'+audio_name.split('.')[0]+'.beats')
                    try:
                        values = np.loadtxt(beat_dir, ndmin=1)
                    except OSError:
                        not_found_error.append(audio_name)
                    specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T
                    #print('specs', specs.shape)
                    spectrogram.append(specs)
                    annotation.append(values)
        # with open(f'../data/audio_lists2/ballroom.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram, dtype=object), np.array(annotation), not_found_error
    
    def initialize_beatles(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['beatles']
        annotation_dir = self.beat_annotation_dir['beatles']
        # audio_list = []
        for album in tqdm(os.listdir(data_dir)):
            album_dir = os.path.join(data_dir, album)
            for audio_name in os.listdir(album_dir):
                #load audio
                audio_dir = os.path.join(album_dir, audio_name)
                # audio_list.append(audio_dir+'\n')
                waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
                #load beat annotations
                beat_dir = os.path.join(annotation_dir, 'beatles_'+album+'_'+audio_name.replace('-_','').split('.')[0]+'.beats')
                # print(beat_dir)
                try:
                    values = np.loadtxt(beat_dir, ndmin=1)
                except OSError:
                    not_found_error.append(audio_name)
                specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T

                #print('specs', specs.shape)
                spectrogram.append(specs)
                annotation.append(values)
        # with open(f'../data/audio_lists2/beatles.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram, dtype=object), np.array(annotation), not_found_error



    def initialize_carnetic(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['carnetic']
        annotation_dir = self.beat_annotation_dir['carnetic']
        # audio_list = []
        for audio_name in (os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            # audio_list.append(audio_dir+'\n')
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            #print(audio.shape, np.mean(audio))
            beat_dir = os.path.join(annotation_dir, audio_name.split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, delimiter=',', ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T

            spectrogram.append(specs)
            annotation.append(values)
        # with open(f'../data/audio_lists2/carnetic.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram), np.array(annotation), not_found_error
    
    def initialize_gtzan(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['gtzan']
        annotation_dir = self.beat_annotation_dir['gtzan']
        # audio_list = []
        for audio_name in os.listdir(data_dir):
            audio_dir = os.path.join(data_dir, audio_name)
            # audio_list.append(audio_dir+'\n')
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, 'gtzan_'+audio_name.split('.')[0]+'_'+audio_name.split('.')[1]+'.beats')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T
            spectrogram.append(specs)
            annotation.append(values)
        # with open(f'../data/audio_lists2/gtzan.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram), np.array(annotation), not_found_error

    def initialize_hainsworth(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['hainsworth']
        annotation_dir = self.beat_annotation_dir['hainsworth']
        # audio_list = []
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            # audio_list.append(audio_dir+'\n')
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, 'hainsworth_'+audio_name.split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            # specs = madmom_feature(waveform)
            specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T

            spectrogram.append(specs)
            annotation.append(values)
        # with open(f'../data/audio_lists2/hainsworth.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram), np.array(annotation), not_found_error

    def initialize_smc(self):
        spectrogram = []
        annotation = []
        not_found_error = []
        data_dir = self.data_dir['smc']
        annotation_dir = self.beat_annotation_dir['smc']
        # audio_list = []
        for audio_name in tqdm(os.listdir(data_dir)):
            audio_dir = os.path.join(data_dir, audio_name)
            # audio_list.append(audio_dir+'\n')
            waveform, _ = self.audio_loader.load(audio_dir, sample_rate=self.SR)
            beat_dir = os.path.join(annotation_dir, audio_name.lower().split('.')[0]+'.beats')
            try:
                values = np.loadtxt(beat_dir, ndmin=1)
            except OSError:
                not_found_error.append(audio_name)
            specs = melspectrogram(y=waveform[:,0], sr=self.SR, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, fmin=30, fmax=11000).T

            spectrogram.append(specs)
            annotation.append(values)
        # with open(f'../data/audio_lists2/smc.txt', 'w') as f:
        #     f.writelines(audio_list)
        return np.array(spectrogram), np.array(annotation), not_found_error


if __name__ == '__main__':
    data = prepare_spectrgram_data()