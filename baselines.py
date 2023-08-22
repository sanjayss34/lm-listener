import random
import json
import numpy as np
from collections import defaultdict
import argparse
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import models.vqvae as vqvae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-path")
    parser.add_argument("--mean-std-path")
    parser.add_argument("--train-segments-path")
    parser.add_argument("--val-segments-path")
    parser.add_argument("--max-motion-length", type=int)
    parser.add_argument("--history-size", type=int)
    parser.add_argument("--nearest-neighbor", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--embedding-model-name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--static-face", action="store_true")
    parser.add_argument("--random-train-select", action="store_true")
    args = parser.parse_args()
    os.system('mkdir '+args.output_dir)
    random.seed(args.seed)
    np.random.seed(seed=args.seed)
    if args.mean_std_path is not None:
        mean = np.load(os.path.join(args.mean_std_path, 'mean.npy'))
        std = np.load(os.path.join(args.mean_std_path, 'std.npy'))
    with open(args.params_path) as f:
        params = json.load(f)
    for key in params:
        if not hasattr(args, key):
            setattr(args, key, params[key])
    if args.nearest_neighbor or args.random_train_select:
        segments = torch.load(args.train_segments_path, map_location='cpu')
        text_to_motion = {}
        fps = args.fps
        text_to_file_id = {}
        for seg in segments:
            if seg['split_end_frame']-seg['split_start_frame'] < args.max_motion_length:
                continue
            for start in range(seg['split_start_frame'], seg['split_end_frame']-args.max_motion_length+1):
                words = [word for word in seg['before_words']+seg['during_words'] if word['end']*fps >= start-args.history_size*fps and word['end']*fps < start+args.max_motion_length]
                text = ' '.join([word['text'] for word in words])
                if text not in text_to_motion:
                    text_to_motion[text] = torch.cat((seg['p0_exp'][start-seg['split_start_frame']:start-seg['split_start_frame']+args.max_motion_length,:], seg['p0_pose'][start-seg['split_start_frame']:start-seg['split_start_frame']+args.max_motion_length,:]), dim=1).numpy()
                    text_to_file_id[text] = seg['fname']+'_'+str(start)
                    # text_to_motion[text] = torch.cat((seg['p0_exp'], seg['p0_pose']), dim=1)
        all_texts = []
        text_id_to_motion = {}
        file_ids = []
        for i, text in enumerate(text_to_motion):
            all_texts.append(text)
            text_id_to_motion[i] = text_to_motion[text]
            file_ids.append(text_to_file_id[text])
    if args.nearest_neighbor:
        text_embeddings = []
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = SentenceTransformer(args.embedding_model_name).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        # bos_token = ''
        for i in tqdm(range(0, len(all_texts), args.batch_size)):
            sentence_embeddings = torch.from_numpy(model.encode(all_texts[i:i+args.batch_size]))
            for j in range(sentence_embeddings.shape[0]):
                text_embeddings.append(sentence_embeddings[j:j+1,:].cpu())
        text_embeddings = torch.cat(text_embeddings, dim=0)
        assert text_embeddings.shape[0] == len(all_texts), str(text_embeddings.shape)+', '+str(len(all_texts))
        print(text_embeddings.shape)
        print(text_embeddings[:2,:])
        if torch.cuda.is_available():
            text_embeddings = text_embeddings.cuda()
        if args.normalize:
            # text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
            text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
        val_segments = torch.load(args.val_segments_path, map_location="cpu")
        val_segments_dict = {}
        for seg in val_segments:
            for i in range(seg['split_start_frame'], seg['split_end_frame']):
                val_segments_dict[seg['fname'].split('/')[-1]+'_'+str(i)] = seg
        # print(val_segments_dict.keys())
    else:
        net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                               args.nb_code,
                               args.code_dim,
                               args.output_emb_width,
                               args.down_t,
                               args.stride_t,
                               args.width,
                               args.depth,
                               args.dilation_growth_rate)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
        net.eval()
        if torch.cuda.is_available():
            net.cuda()
    chosen = defaultdict(int)
    count = 0
    for root, _, files in tqdm(os.walk(args.vq_dir)):
        for fname in files:
            if fname[-4:] == ".npy":
                count += 1
                gt_vq = np.load(os.path.join(root, fname))
                num_frames = gt_vq.reshape(-1).shape[0]*(2**args.down_t)
                if args.nearest_neighbor:
                    seg = val_segments_dict[fname.split('.npy')[0]]
                    start_frame = int(fname.split('.npy')[0].split('_')[-1])
                    words = [word['text'] for word in seg['before_words']+seg['during_words'] if word['end']*fps >= start_frame-args.history_size*fps and word['end']*fps < start_frame+num_frames]
                    text = ' '.join(words)
                    embedding = torch.from_numpy(model.encode([text])).view(1, -1)
                    if torch.cuda.is_available():
                        embedding = embedding.to('cuda:0')
                    if args.normalize:
                        # text_embeddings = text_embeddings / torch.linalg.norm(text_embeddings, dim=-1, keepdim=True)
                        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                    best_index = (text_embeddings @ embedding.t()).view(-1).argmax().item()
                    # if count > 180:
                    print(fname, text, '|||', all_texts[best_index])
                    chosen[best_index] += 1
                    motion = text_id_to_motion[best_index][:num_frames,:]
                elif args.random_train_select:
                    index = random.choice(list(range(len(text_id_to_motion))))
                    motion = text_id_to_motion[index][:num_frames,:]
                elif args.static_face:
                    motion = np.expand_dims(mean, axis=0)
                    if len(gt_vq.shape) == 3:
                        motion = np.repeat(np.expand_dims(motion, axis=0), num_frames, axis=1)
                    else:
                        motion = np.repeat(motion, num_frames, axis=0)
                else:
                    random_pred = np.random.randint(low=0, high=args.nb_code, size=gt_vq.shape)
                    inp = torch.from_numpy(random_pred).view(1, -1)
                    if torch.cuda.is_available():
                        inp = inp.cuda()
                    with torch.no_grad():
                        decoded = net.forward_decoder(inp)
                    motion = decoded.cpu().view(-1, 56).numpy()
                    motion = (motion*std.reshape(1, -1))+mean.reshape(1, -1)
                path_parts = os.path.join(root, fname).replace('.npy', '_pred.npy').split('/')
                new_path = os.path.join(args.output_dir, *path_parts[-4:])
                path_parts = new_path.split('/')
                # print(path_parts)
                for j in range(len(path_parts)-1):
                    if not os.path.exists('/'.join(path_parts[:j+1])):
                        os.system('mkdir '+'/'.join(path_parts[:j+1]))
                np.save(new_path, motion)
