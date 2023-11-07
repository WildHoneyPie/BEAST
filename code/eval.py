# The code is borrowed from Beat Transformer (https://github.com/zhaojw1998/Beat-Transformer/)
import os
import pickle
import torch
import madmom
import numpy as np
from utils import AverageMeter
from torch.utils.data import DataLoader
from StreamingTransformer import TransformerModel

from spectrogram_dataset import audioDataset

from tqdm import tqdm
import shutil

import warnings
warnings.filterwarnings('ignore')

FPS = 44100 / 1024
NUM_FOLDS = 1
#model
NTOKEN=2
DMODEL=256
NHEAD=8
DHID=1024
NLAYER=9
DROPOUT=.1
LEFT=256
CENTER=16
RIGHT=16
DEVICE='cuda:0'

#directories
DATASET_PATH = '../data/gtzan_data.npz'
ANNOTATION_PATH = '../data/gtzan_beat_annotation.npz'

DEMO_SAVE_ROOT = '../save/inference'
if not os.path.exists(DEMO_SAVE_ROOT):
    os.makedirs(DEMO_SAVE_ROOT)



PARAM_PATH = {0:"../data/BEAST_param.pt"}


def infer_gtzan_activation():
    """
    predict (down-)beat activations for the test-only GTZAN dataset
    """
    dataset = audioDataset(data_to_load=['gtzan'],
                            test_only_data = ['gtzan'],
                            data_path = DATASET_PATH, 
                            annotation_path = ANNOTATION_PATH,
                            fps = FPS,
                            sample_size = None,
                            num_folds = NUM_FOLDS)

    inference_pred = {}
    beat_gt = {}
    downbeat_gt = {}

    FOLD = 0
    train_set, val_set, test_set = dataset.get_fold(fold=FOLD)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = TransformerModel(ntoken=NTOKEN, 
                        dmodel=DMODEL,
                        nhead=NHEAD,
                        d_hid=DHID,
                        nlayers=NLAYER,
                        dropout=DROPOUT,
                        left_size=LEFT,
                        center_size=CENTER,
                        right_size=RIGHT
                        )
    
    model.load_state_dict(torch.load(PARAM_PATH[FOLD], map_location=torch.device('cuda'))['state_dict'])
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for idx, (dataset_key, data, beat, downbeat, tempo, root) in tqdm(enumerate(loader), total=len(loader)):
            #data
            data = data.float().to(DEVICE)
            pred, _ = model(data)
            beat_pred = torch.sigmoid(pred[0, :, 0]).detach().cpu().numpy()
            downbeat_pred = torch.sigmoid(pred[0, :, 1]).detach().cpu().numpy()

            beat = torch.nonzero(beat[0]>.5)[:, 0].detach().numpy() / (FPS)
            downbeat = torch.nonzero(downbeat[0]>.5)[:, 0].detach().numpy() / (FPS)

            dataset_key = dataset_key[0]
            if not dataset_key in inference_pred:

                inference_pred[dataset_key] = []
                beat_gt[dataset_key] = []
                downbeat_gt[dataset_key] = []
            inference_pred[dataset_key].append(np.stack((beat_pred, downbeat_pred), axis=0))
            beat_gt[dataset_key].append(beat)
            downbeat_gt[dataset_key].append(downbeat)


    print('saving prediction ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'pred.pkl'), 'wb') as f:
       pickle.dump( inference_pred, f)
    print('saving gt ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gt.pkl'), 'wb') as f:
       pickle.dump(beat_gt, f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'downbeat_gt.pkl'), 'wb') as f:
       pickle.dump(downbeat_gt, f)



    
def inference_gtzan_dbn():
    """
    locate (down-)beat timesteps from activations for the test-only GTZAN dataset
    """
    print('loading activations ...')
    with open(os.path.join(DEMO_SAVE_ROOT, 'pred.pkl'), 'rb') as f:
        activations = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'beat_gt.pkl'), 'rb') as f:
        beat_gt = pickle.load(f)
    with open(os.path.join(DEMO_SAVE_ROOT, 'downbeat_gt.pkl'), 'rb') as f:
        downbeat_gt = pickle.load(f)
    
    dataset_key ='gtzan'
    print(f'inferencing on {dataset_key} ...')

    beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=2, observation_lambda=8,online=True)
    downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=80, observation_lambda=15,online=True)

    beat_DBN_meter = AverageMeter()
    Downbeat_DBN_meter = AverageMeter()

    beat_error = 0
    downbeat_error = 0
    for i in tqdm(range(len(activations[dataset_key]))):
        pred = activations[dataset_key][i]
        #print(pred.shape)
        beat = beat_gt[dataset_key][i]
        downbeat = downbeat_gt[dataset_key][i]

        try:
            dbn_beat_pred = beat_tracker(pred[0])
            beat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_beat_pred, beat)
            beat_DBN_meter.update(f'{dataset_key}-fmeasure', beat_score_DBN.fmeasure)
            # beat_DBN_meter.update(f'{dataset_key}-cmlt', beat_score_DBN.cmlt)
            # beat_DBN_meter.update(f'{dataset_key}-amlt', beat_score_DBN.amlt)

            combined_act = np.concatenate((np.maximum(pred[0] - pred[1], np.zeros(pred[0].shape))[:, np.newaxis], pred[1][:, np.newaxis]), axis=-1)   #(T, 2)
            #print(combined_act.shape)
            dbn_downbeat_pred = downbeat_tracker(combined_act)
            dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

            downbeat_score_DBN = madmom.evaluation.beats.BeatEvaluation(dbn_downbeat_pred, downbeat)
            Downbeat_DBN_meter.update(f'{dataset_key}-fmeasure', downbeat_score_DBN.fmeasure)
            # Downbeat_DBN_meter.update(f'{dataset_key}-cmlt', downbeat_score_DBN.cmlt)
            # Downbeat_DBN_meter.update(f'{dataset_key}-amlt', downbeat_score_DBN.amlt)


        except Exception as e:
            #print(f'beat inference encounter exception {e}')
            beat_error += 1
            downbeat_error += 1

    print(f'beat error: {beat_error}; downbeat error: {downbeat_error}')

    print('DBN beat detection')
    for key in beat_DBN_meter.avg.keys():
        print('\t', key, beat_DBN_meter.avg[key])
    print('DBN downbeat detection')
    for key in Downbeat_DBN_meter.avg.keys():
        print('\t', key, Downbeat_DBN_meter.avg[key])



if __name__ == '__main__':
    infer_gtzan_activation()
    inference_gtzan_dbn()
