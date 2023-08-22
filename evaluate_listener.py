import argparse
import os
import torch
import numpy as np
import pickle as pkl
import subprocess
import json
import cv2
import random
random.seed(224)
from tqdm import tqdm
from scipy import linalg
from pathlib import Path
import pandas as pd


import models.vqvae as vqvae

import sys
sys.path.append(os.environ['DECA_PATH'])
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets

from gdl.utils.other import get_path_to_assets
from gdl_apps.EmotionRecognition.utils.io import load_model, test
import scipy.stats as stats

def calc_pearson(in_features, out_features):
    T,F = in_features.shape
    res_corr = np.zeros(F)
    for f in range(F):
        r,p = stats.pearsonr(in_features[:,f], out_features[:,f])
        res_corr[f] = r
    return abs(np.mean(np.mean(res_corr, axis=-1)))

def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def face_valence(gt_exp, gt_pose, gt_shape, affect_model):
    gt_dict = {"expcode": torch.reshape(gt_exp, (-1, 50)), 
               "posecode": torch.reshape(gt_pose, (-1, 6)), 
               "shapecode": torch.reshape(gt_shape, (-1, 100))}
    with torch.no_grad():
        gt_affect = affect_model(gt_dict)
    return gt_affect['valence']

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()

def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def main(args):
    total_l2 = []
    total_fid = []
    total_fid2 = []
    total_diversity = []
    total_diversity_gt = []
    total_var = []
    total_var_gt = []
    total_windowed_l2v = []
    total_peak_windowed_l2v = []
    total_l2v = []
    processed = []

    # NOTE: added affect model here
    model_name = 'EMOCA-emorec'
    path_to_models = get_path_to_assets() /"EmotionRecognition"
    path_to_models = path_to_models / "face_reconstruction_based" # for 3dmm model
    affect_model = load_model(Path(path_to_models) / model_name)
    affect_model.eval() # .cuda()

    segments = torch.load(args.segments_path, map_location='cpu')
    segments_dict = {datum['fname']+'_'+str(datum['split_start_frame']): datum for datum in segments}

    frame_map = {}
    for index, seg in enumerate(segments):
        for i in range(seg['split_start_frame'], seg['split_end_frame']):
            frame_map[seg['fname']+'_'+str(i)] = np.concatenate((seg['p0_exp'][i-seg['split_start_frame'],:], seg['p0_pose'][i-seg['split_start_frame'],:], seg['p0_shape'][i-seg['split_start_frame'],:]), axis=0)
 
    speaker_map = {}
    for index, seg in enumerate(segments):
        for i in range(seg['split_start_frame'], seg['split_end_frame']):
            speaker_map[seg['fname']+'_'+str(i)] = np.concatenate((seg['p1_exp'][i-seg['split_start_frame'],:], seg['p1_pose'][i-seg['split_start_frame'],:], seg['p1_shape'][i-seg['split_start_frame'],:]), axis=0)

    fps = args.fps
    fname_pairs = []
    for root, _, files in os.walk(args.output_dir):
        for fname in files:
            if '_pred.npy' in fname:
                # print(fname)
                fname_pairs.append((root, fname))
    fname_pairs = sorted(fname_pairs, key=lambda x: '/'.join(os.path.join(x[0], x[1]).split('/')[2:]))
    
    fids = []
    fid2s = []
    l2s = []
    gt_diversities = []
    pred_diversities = []
    gt_vars = []
    pred_vars = []
    # trevor_videos/done_trevor_videos1/025YouTubetrevor_videos/done_trevor_videos1/025YouTube/025YouTube.mp4_10916

    for root, fname in fname_pairs:
        final_name = "_".join(root.split('/')[-3:])

        pred = np.load(os.path.join(root, fname)).reshape(-1, 56)
        # gt = np.load(os.path.join(root, fname.replace('_pred.npy', '_gt.npy')))[:,:56]
        root_parts = root.split('/')
        # 10946

        if not fname.split('_')[-2].isnumeric():
            continue
        start_frame = int(fname.split('_')[-2])
        fn = '/'.join(root_parts[-3:])+'/'+root_parts[-1]+'.mp4'
        valid_keys = [x for x in frame_map.keys() if fn in x]
        # print(frame_map.keys())
        res = []

        if pred.shape[0] < args.min_num_frames:
            continue

        if any([fn+'_'+str(f) not in frame_map for f in range(start_frame, start_frame+pred.shape[0])]):
            print(fn+' NOT FOUND')
            continue

        gt = np.stack([frame_map[fn+'_'+str(f)] for f in range(start_frame, start_frame+pred.shape[0])])
        speaker = np.stack([speaker_map[fn+'_'+str(f)] for f in range(start_frame, start_frame+pred.shape[0])])

        gt_v = face_valence(torch.from_numpy(gt[:,:50]), torch.from_numpy(gt[:,50:56]), torch.from_numpy(gt[:,56:]), affect_model).cpu().detach().numpy()
        pred_v = face_valence(torch.from_numpy(pred[:,:50]), torch.from_numpy(pred[:,50:56]), torch.from_numpy(gt[:,56:]), affect_model).cpu().detach().numpy()
        
        # 1. fid
        gt_mu, gt_cov  = calculate_activation_statistics(gt[:,:56])
        mu, cov = calculate_activation_statistics(pred[:,:56])
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        total_fid.append(fid)
        # 2. paired fid
        gt_mu2, gt_cov2  = calculate_activation_statistics(np.concatenate([speaker[:,:56], gt[:,:56]], axis=-1))
        mu2, cov2 = calculate_activation_statistics(np.concatenate([speaker[:,:56], pred[:,:56]], axis=-1))
        fid2 = calculate_frechet_distance(gt_mu2, gt_cov2, mu2, cov2)
        total_fid2.append(fid2)
        # 3. l2
        mse = ((gt[:,:56] - pred[:,:56])**2).mean()
        total_l2.append(mse)
        # 4. diversity
        gt_diversity = calculate_diversity(gt[:,:56], 30 if len(gt[:,:56]) > 30 else 10)
        pred_diversity = calculate_diversity(pred[:,:56], 30 if len(pred[:,:56]) > 30 else 10)
        total_diversity.append(pred_diversity)
        total_diversity_gt.append(gt_diversity)
        # 5. variance
        gt_var = np.mean(np.var(gt[:,:56], axis=0))
        pred_var = np.mean(np.var(pred[:,:56], axis=0))
        total_var.append(pred_var)
        total_var_gt.append(gt_var)
        # # 7. diff in valence
        mse_v = ((gt_v - pred_v)**2).mean()
        total_l2v.append(mse_v)

        # Windowed valence
        windowed_gt_v = torch.from_numpy(gt_v).view(-1).unfold(dimension=0, size=min(args.valence_window_size, gt_v.shape[0]), step=args.valence_window_size)
        index_per_window_gt = windowed_gt_v.abs().argmax(dim=-1)
        assert windowed_gt_v.shape[-1] == min(args.valence_window_size, gt_v.shape[0])
        # windowed_gt_v = windowed_gt_v.mean(dim=-1)
        windowed_pred_v = torch.from_numpy(pred_v).view(-1).unfold(dimension=0, size=min(args.valence_window_size, pred_v.shape[0]), step=args.valence_window_size)
        assert windowed_pred_v.shape[-1] == min(args.valence_window_size, pred_v.shape[0])
        index_per_window_pred = windowed_pred_v.abs().argmax(dim=-1)
        value_per_window_gt = windowed_gt_v.gather(dim=1, index=index_per_window_gt.view(-1, 1))
        value_per_window_pred = windowed_pred_v.gather(dim=1, index=index_per_window_pred.view(-1, 1))
        windowed_mse_v = ((value_per_window_pred-value_per_window_gt)**2).mean()
        # windowed_pred_v = windowed_pred_v.mean(dim=-1).numpy()
        # windowed_mse_v = ((windowed_gt_v-windowed_pred_v)**2).mean()
        total_peak_windowed_l2v.append(windowed_mse_v)

        # Windowed valence
        windowed_gt_v = torch.from_numpy(gt_v).view(-1, 1).unfold(dimension=0, size=min(args.valence_window_size, gt_v.shape[0]), step=args.valence_window_size)
        windowed_gt_v = windowed_gt_v.mean(dim=-1)
        windowed_pred_v = torch.from_numpy(pred_v).view(-1, 1).unfold(dimension=0, size=min(args.valence_window_size, pred_v.shape[0]), step=args.valence_window_size)
        windowed_pred_v = windowed_pred_v.mean(dim=-1).numpy()
        windowed_mse_v = ((windowed_gt_v-windowed_pred_v)**2).mean()
        total_windowed_l2v.append(windowed_mse_v)

        processed.append((root, fname))

    print("l2", np.mean(np.array(total_l2)))
    print("windowed avg.l2v", np.mean(np.array(total_windowed_l2v)))
    print("fid", np.mean(np.array(total_fid)))
    print("fid2", np.mean(np.array(total_fid2)))
    print("diversity", np.mean(np.array(total_diversity)))
    print("diversity GT", np.mean(np.array(total_diversity_gt)))
    print("var", np.mean(np.array(total_var)))
    print("var GT", np.mean(np.array(total_var_gt)))

    result = {
        "name": args.output_dir,
        "l2": str(np.mean(np.array(total_l2))),
        "windowed avg.l2v": str(np.mean(np.array(total_windowed_l2v))),
        "fid": str(np.mean(np.array(total_fid))),
        "fid2": str(np.mean(np.array(total_fid2))),
        "diversity": str(np.mean(np.array(total_diversity))),
        "var": str(np.mean(np.array(total_var))),
    }

    tag = "talkshow"
    with open(f"{args.output_dir}/{tag}_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(f"{args.output_dir}/{tag}_scores.json", "w") as fout:
        json.dump({
            'paths': processed,
            'l2': [float(val) for val in total_l2],
            'fid': [float(val) for val in total_fid],
            'fid2': [float(val) for val in total_fid2],
            'windowed_avg_l2v': [float(val) for val in total_windowed_l2v],
            'diversity': [float(val) for val in total_diversity],
            'diversity_gt': [float(val) for val in total_diversity_gt],
            'var': [float(val) for val in total_var],
            'var_gt': [float(val) for val in total_var_gt],
        }, fout)
    print(f"dumped to: {args.output_dir}/{tag}_eval.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    # parser.add_argument("--vq_dir")
    parser.add_argument("--segments_path")
    parser.add_argument("--default_code_path")
    parser.add_argument("--mean_std_path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--valence_window_size", type=int, default=30)
    parser.add_argument("--min-num-frames", type=int, default=0)
    args = parser.parse_args()
    main(args)
