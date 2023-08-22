import argparse
import cv2
import torch
import models.vqvae as vqvae
import numpy as np
import subprocess
import json
import os
import cv2
from tqdm import tqdm
import pickle as pkl

import sys
sys.path.append(os.environ['DECA_PATH'])
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets

def gen_image(deca, codedict, include_im, fix_cam=True):
    #codedict['cam'] = [5.,-0.02,0.02]
    if fix_cam:
        codedict['cam'][0,0] = 5.
        codedict['cam'][0,1] = 0.
        codedict['cam'][0,2] = 0.05
    #print(codedict['cam'])
    opdict, visdict = deca.decode(codedict) # , include_im=include_im) #tensor
    landmarks = {'landmarks2d': visdict['landmarks2d']}
    if include_im:
        #remainder = {'inputs': visdict['inputs'], 'shape_detail_images': visdict['shape_detail_images']}
        remainder = {'shape_detail_images': visdict['shape_detail_images'], 'inputs': visdict['inputs']}
    else:
        remainder = {'shape_detail_images': visdict['shape_detail_images']}

    #if include_im:
    #    remainder['inputs'] = visdict['inputs']
    return deca.visualize(remainder, size=640), deca.visualize(landmarks, size=640)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", help="comma-separated list of things to visualize (choices: \"video\", \"gt\", \"vq\", or any output directory)")
    parser.add_argument("--output_dir")
    parser.add_argument("--segments_path")
    parser.add_argument("--default_code_path")
    parser.add_argument("--params_path")
    parser.add_argument("--mean_std_path")
    parser.add_argument("--audio_root")
    parser.add_argument("--video_root")
    parser.add_argument("--tmp-dir", default="vis_tmp")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--history", type=int, default=0)
    args = parser.parse_args()

    deca = DECA(config = deca_cfg, device='cuda')
    with open(args.default_code_path, 'rb') as f:
        default_code = pkl.load(f)
    basename = os.path.basename(os.path.abspath(args.output_dir))

    params = None
    if args.params_path is not None:
        with open(args.params_path) as f:
            params = json.load(f)
        for key in params:
            if not hasattr(args, key):
                setattr(args, key, params[key])
        net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                               args.nb_code,
                               args.code_dim,
                               args.output_emb_width,
                               args.down_t,
                               args.stride_t,
                               args.width,
                               args.depth,
                               args.dilation_growth_rate)
        unit_length = 2**args.down_t
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
        net.eval()
        net.cuda()
    mean = torch.from_numpy(np.load(os.path.join(args.mean_std_path, 'mean.npy'))).cuda().view(1, -1)
    std = torch.from_numpy(np.load(os.path.join(args.mean_std_path, 'std.npy'))).cuda().view(1, -1)
    segments = torch.load(args.segments_path, map_location='cpu')
    frame_map = {}
    for index, seg in enumerate(segments):
        for i in range(seg['split_start_frame'], seg['split_end_frame']):
            frame_map[seg['fname'].split('/')[-1]+'_'+str(i)] = torch.from_numpy(np.concatenate((seg['p0_exp'][i-seg['split_start_frame'],:], seg['p0_pose'][i-seg['split_start_frame'],:]), axis=0))

    items = [item.strip() for item in args.items.split(',')]
    fname_pairs = []
    fname_maps = {}
    for item in items:
        if item not in ["gt", "vq", "video"]:
            fname_pairs = []
            fname_maps[item] = {}
            for root, _, files in os.walk(item):
                for fname in files:
                    if '_pred.npy' in fname:
                        fname_pairs.append(('_'.join(fname.split('_')[:-2]), int(fname.split('_')[-2])))
                        fname_maps[item]['_'.join(fname.split('_')[:-1])] = root+'/'+'_'.join(fname.split('_')[:-1])
    if len(fname_pairs) == 0:
        for i, datum in enumerate(segments):
            fname_pairs.append((datum['fname'], datum['split_start_frame']))
    audio_fname_map = {}
    for root, dirs, files in os.walk(args.audio_root+'/'):
        for fname in files:
            if fname[-4:] in {'.mp3', '.wav'}:
                audio_fname_map[fname.split('.')[0]] = root.split(args.audio_root+'/')[1]
    video_fname_map = {}
    for root, dirs, files in os.walk(args.video_root+'/'):
        for fname in files:
            if fname[-4:] in {'.mp4'}:
                video_fname_map[fname.split('.')[0]] = root.split(args.video_root+'/')[1]
    os.system('mkdir '+args.tmp_dir)
    for path, start_frame in tqdm(fname_pairs):
        fname = path.split('/')[-1]
        frames = []
        num_frames = None
        f = start_frame
        pred_dict = {}
        for item in items:
            if item not in {'video', 'gt', 'vq'}:
                pred = np.load(os.path.join(fname_maps[item][path+'_'+str(start_frame)]+'_pred.npy'))
                pred_dict[item] = pred.reshape(-1, pred.shape[-1])
        cap = None
        video_fname = fname
        if video_fname[-4:] != '.mp4':
            video_fname += '.mp4'
        if "video" in items:
            assert args.video_root is not None, "video_root must be non-None if you want to include the video in the visualization"
            print(os.path.join(args.video_root, video_fname_map[fname.split('.')[0]], fname))
            cap = cv2.VideoCapture(os.path.join(args.video_root, video_fname_map[fname.split('.')[0]], video_fname))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)*start_frame/args.fps)-1)
        while num_frames is None or len(frames) < num_frames:
            frame = []
            for item in items:
                if item == "gt":
                    gt_code = {
                        'exp': frame_map[fname+'_'+str(f)][:50].cuda().view(1, -1),
                        'pose': frame_map[fname+'_'+str(f)][50:56].cuda().view(1, -1)
                    }
                    for key in default_code:
                        if key not in {'exp', 'pose'}:
                            gt_code[key] = default_code[key].float().cuda()
                        gt_code[key] = gt_code[key].cuda()
                    gt_image, _ = gen_image(deca, gt_code, include_im=False)
                    frame.append(gt_image)
                    if num_frames is None:
                        t = f
                        while fname+'_'+str(t) in frame_map:
                            t += 1
                        num_frames = t-f
                        # print('GT', num_frames)
                elif item == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)*(start_frame+len(frames))/args.fps)-1)
                    res, video_frame = cap.read()
                    assert res
                    video_frame = cv2.resize(video_frame, (640, 640), interpolation = cv2.INTER_AREA)
                    frame.append(video_frame)
                elif item not in {"vq", "video"}:
                    pred_code = {
                        'exp': torch.from_numpy(pred_dict[item][f-start_frame,:50]).cuda().view(1, -1),
                        'pose': torch.from_numpy(pred_dict[item][f-start_frame,50:56]).cuda().view(1, -1)
                    }
                    for key in default_code:
                        if key not in {'exp', 'pose'}:
                            pred_code[key] = default_code[key].float().cuda()
                        pred_code[key] = pred_code[key].cuda()
                    pred_image, _ = gen_image(deca, pred_code, include_im=False)
                    frame.append(pred_image)
                    if num_frames is None:
                        num_frames = pred_dict[item].shape[0]
                    num_frames = min(num_frames, pred_dict[item].shape[0])
                    # print('PRED', num_frames, pred_dict[item].shape)
            frames.append(tuple(frame))
            f += 1
        if "vq" in items:
            gt = torch.stack([
                frame_map[fname+'_'+str(f)]
                for f in range(start_frame, start_frame+num_frames)
            ]).cuda()
            normalized = ((gt-mean.cuda()) / std.cuda()).unsqueeze(0).cuda()
            with torch.no_grad():
                encoded = net.encode(normalized)
                decoded = net.forward_decoder(encoded).view(-1, 56)
            denorm = (std*decoded+mean)
            while denorm.shape[0] < num_frames:
                denorm = torch.cat((denorm, denorm[-1:,:]), dim=0)
            for f in range(num_frames):
                vq_code = {
                    'exp': denorm[f,:50].view(1, -1),
                    'pose': denorm[f,50:].view(1, -1),
                }
                for key in default_code:
                    if key not in {'exp', 'pose'}:
                        vq_code[key] = default_code[key].float().cuda()
                    vq_code[key] = vq_code[key].cuda()
                vq_image, _ = gen_image(deca, vq_code, include_im=False)
                frames[f] = frames[f][:items.index('vq')]+(vq_image,)+frames[f][items.index('vq'):]
        print('NUM_FRAMES', len(frames))
        vis_start_frame = max(0, start_frame - args.fps * args.history)
        start_time = vis_start_frame / args.fps
        num_frames += start_frame-vis_start_frame
        interval = (num_frames) / args.fps
        prefix_frames = []
        for f in range(vis_start_frame, start_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)*(f)/args.fps)-1)
            res, video_frame = cap.read()
            assert res
            video_frame = cv2.resize(video_frame, (640, 640), interpolation = cv2.INTER_AREA)
            frame = []
            for item in items:
                if item == "video":
                    frame.append(video_frame)
                else:
                    frame.append(np.zeros_like(video_frame))
            prefix_frames.append(tuple(frame))
        frames = prefix_frames + frames
        for f, frame in enumerate(frames):
            cv2.imwrite(args.tmp_dir+'/{:08d}.jpg'.format(f), np.concatenate(frame, axis=1))
        audio_path = os.path.join(args.audio_root, audio_fname_map[fname.split('.')[0]], video_fname.replace('.mp4', '.wav'))
        if not os.path.exists(audio_path):
            audio_path = audio_path.replace('.wav', '.mp3')
        subprocess.call('ffmpeg -y -ss '+str(start_time)+' -t '+str(interval)+' -i '+audio_path+' '+args.tmp_dir+'/audio.wav', shell=True)
        cmd = "ffmpeg -y -r "+str(args.fps)+f" -start_number 0 -i "+args.tmp_dir+"/%8d.jpg -i "+args.tmp_dir+f"/audio.wav -pix_fmt yuv420p -vframes {num_frames} "+os.path.join(args.output_dir, fname+'_'+str(start_frame))+'.mp4'
        subprocess.call(cmd, shell=True)
        os.system('rm -rf '+args.tmp_dir+'/*')
