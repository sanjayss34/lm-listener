from collections import defaultdict
import string
import torch
from torch.utils import data
import math
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
import librosa
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
from transformers import Wav2Vec2Processor, AutoModel, AutoTokenizer
import pickle

"""
python train_t2m_trans.py --exp-name m3_a3_t3_hist3 --batch-size 16 --nb-code 256 --drop-out-rate 0.1 --resume-pth output/VQVAE_l1smooth_c256_p8_datav2/net_iter300000.pth --vq-name VQVAE_l1smooth_c256_p8_datav2_300k_delay96 --out-dir output --total-iter 100000 --lr-scheduler 150000 --lr 0.00005 --dataname face_trevor --down-t 3 --depth 3 --quantizer ema_reset --eval-iter 2000 --pkeep 0.5 --dilation-growth-rate 3 --vq-act relu --max-motion-length 240 --gpt2 gpt2-medium --print_val_pred --max-time-before 3 --include-speaker --fix-before-text --delay-start-frames 96 --normalize-speaker --speaker-pkeep 0.5 --include-audio --audio-pkeep 0.5 --normalize-audio
"""

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def create_vq_token(index, remove_space_before_vq_tokens):
    if remove_space_before_vq_tokens:
        return " <<<vq"+str(index)+">>>"
    else:
        return "<<<vq"+str(index)+">>>"

'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, split="train", max_motion_length=None, 
                 text_token_level=False, evaluation=False, gpt2_config=None, no_text=False, 
                 max_tokens=None, no_vq=False, no_before_text=False, max_time_before=None, 
                 fps=30, fixed_text_token=False, fixed_text_token_not_space=False, fixed_text_token_not_punctuation=False, unaligned_text=False, remove_space_before_vq_tokens=False, random_text_token_order=False):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            self.max_motion_length *= unit_length
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            self.max_motion_length *= unit_length
            kinematic_chain = paramUtil.kit_kinematic_chain
        elif dataset_name.split('_')[0] == 'face':
            self.data_root = './dataset/'+dataset_name.split('_')[1]

        if max_motion_length is not None:
            self.max_motion_length = max_motion_length
        self.split = split
        self.evaluation = evaluation
        self.max_time_before = max_time_before
        self.text_token_level = text_token_level
        self.no_vq = no_vq

        if random_text_token_order:
            random.seed(222)
        
        new_name_list = []
        data_dict = {}
        self.before_text_dict = None
        self.during_text_dict = None
        self.fps = fps
        if dataset_name.split('_')[0] == 'face':
            segments = torch.load(pjoin(self.data_root, "segments_"+split+".pth"), map_location="cpu")
            segments_dict = {datum['fname']+'_'+str(datum['split_start_frame']): datum for datum in segments}
            frame_map = defaultdict(dict)
            for key in segments_dict:
                for f in range(segments_dict[key]['split_start_frame'], segments_dict[key]['split_end_frame']):
                    frame_map[segments_dict[key]['fname']][f] = segments_dict[key]['split_start_frame']
            self.before_text_dict = {}
            self.during_text_dict = {}
            self.mean = np.load(pjoin(self.data_root, "mean.npy"))
            self.std = np.load(pjoin(self.data_root, "std.npy"))
            if gpt2_config is not None:
                self.tokenizer_config = gpt2_config
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_config._name_or_path)
                self.max_tokens_length = 2*self.max_motion_length # self.tokenizer_config.n_positions
                if max_tokens is not None:
                    self.max_tokens_length = max_tokens
                assert self.max_tokens_length >= self.max_motion_length // unit_length
                vocab = self.tokenizer.get_vocab()
                punctuation_tokens = []
                for key in vocab:
                    if all([c in string.punctuation or c == 'Ġ' for c in key]):
                        punctuation_tokens.append(vocab[key])
                self.tokenizer_vq_start = len(vocab)
                self.vq_str_map = {i: create_vq_token(i, remove_space_before_vq_tokens) for i in range(codebook_size+1)}
                curr_codebook_size = codebook_size+1
                self.tokenizer.add_tokens([self.vq_str_map[i] for i in range(len(self.vq_str_map))])
            print(pjoin(self.data_root, tokenizer_name))
            for root, dirs, files in tqdm(os.walk(pjoin(self.data_root, tokenizer_name))):
                for fname in files:
                    if fname[-4:] == '.npy':
                        path = pjoin(root, fname)
                        target = np.load(path)
                        target = target.reshape(-1)
                        parts = fname.split('.npy')[0].split('_')
                        if 'realtalk' in dataset_name:
                            prefix = '_'.join(parts[:-1])
                            name = prefix+'_'+parts[-1]
                        else:
                            prefix = '/'.join(root.split('/')[-3:])+'/'+'_'.join(parts[:-1])
                            name = prefix+'_'+parts[-1]
                        new_name_list.append(name)
                        start_frame = int(parts[-1])
                        end_frame = start_frame+target.shape[0]*unit_length
                        # print(pjoin(root, fname))
                        if start_frame not in frame_map[prefix]:
                            print(prefix)
                            print(pjoin(self.data_root, tokenizer_name))
                            print(frame_map[prefix].keys())
                        overall_start = frame_map[prefix][start_frame]
                        dict_name = prefix+'_'+str(overall_start)
                        # print('HELLO', segments_dict[dict_name])
                        before_text = [word['text'] for word in segments_dict[dict_name]['before_words']]
                        before_token_frames = [int(math.ceil(word['end']*fps)-start_frame) for word in segments_dict[dict_name]['before_words']]
                        if max_time_before is not None:
                            before_text = [word['text'] for word in segments_dict[dict_name]['before_words'] if start_frame/fps-word['end'] < max_time_before]
                            before_token_frames = [int(math.ceil(word['end']*fps)-start_frame) for word in segments_dict[dict_name]['before_words'] if start_frame/fps-word['end'] < max_time_before]
                        during_text = []
                        token_frames = []
                        for word in segments_dict[dict_name]['during_words']:
                            if word['end'] < start_frame/fps:
                                if max_time_before is None or start_frame/fps-word['end'] < max_time_before:
                                    before_text.append(word['text'])
                                    before_token_frames.append(int(math.ceil(word['end']*fps)-start_frame))
                            elif word['end'] < (end_frame-1)/fps:
                                during_text.append(word['text'])
                                token_frames.append(int(math.ceil(word['end']*fps)-start_frame))
                        if remove_space_before_vq_tokens:
                            before_text = [" "+word for word in before_text]
                            during_text = [" "+word for word in during_text]
                        if random_text_token_order:
                            random.shuffle(before_text)
                            random.shuffle(during_text)
                        if no_before_text:
                            before_text = []
                        char_indices = [[] for _ in range(end_frame-start_frame)]
                        full_text = ""
                        for f, word in zip(token_frames, during_text):
                            if f >= end_frame-start_frame:
                                assert f == end_frame-start_frame
                                continue
                            if len(full_text) > 0:
                                full_text += " "
                            char_indices[f].append(len(full_text))
                            full_text += word
                        data_dict[name] = {
                            "m_tokens": target,
                            "before_text": " ".join(before_text),
                            "during_text": " ".join(during_text),
                            "during_text_tokens": during_text,
                            "during_text_frames": token_frames,
                            "during_text_char_indices": char_indices,
                        }
                        assert target.shape[0] <= self.max_motion_length // unit_length
                        if gpt2_config is not None:
                            tokens = []
                            next_token_index = 0
                            for j in range(len(target)):
                                while next_token_index < len(token_frames) and token_frames[next_token_index] < j*unit_length:
                                    if not no_text:
                                        tokens.append(during_text[next_token_index])
                                    next_token_index += 1
                                tokens.append(self.vq_str_map[int(target[j])])
                            assert next_token_index == len(token_frames) or token_frames[next_token_index] >= unit_length*(len(target)-1), ', '.join([str(next_token_index), str(len(token_frames)), str(token_frames[next_token_index]), str(target[-1]), str(start_frame), str(end_frame)])
                            if remove_space_before_vq_tokens:
                                full_str_rep = "".join(tokens)
                                # print(full_str_rep)
                            else:
                                full_str_rep = " ".join(tokens)
                            tokens = self.tokenizer([full_str_rep], padding=False, truncation=False, return_tensors='pt').input_ids[0,:].tolist()
                            if isinstance(before_text, list):
                                if remove_space_before_vq_tokens:
                                    before_text = ''.join(before_text)
                                else:
                                    before_text = ' '.join(before_text)
                            before_tokens = self.tokenizer([before_text], padding=False, truncation=False, return_tensors='pt').input_ids[0,:].tolist()
                            before_tokens = [tok if tok != self.tokenizer_vq_start else -1 for tok in before_tokens]
                            
                            if len(before_tokens) > 0 and before_tokens[0] == self.tokenizer.bos_token_id:
                                before_tokens = before_tokens[1:]
                            if tokens[0] == self.tokenizer.bos_token_id:
                                tokens = tokens[1:]
                            if len(before_tokens) < self.max_tokens_length-len(tokens)-1 and not no_text:
                                tokens = before_tokens+tokens
                            elif self.max_tokens_length-len(tokens)-1 > 0 and not no_text:
                                tokens = before_tokens[-(self.max_tokens_length-len(tokens)-1):]+tokens
                            if no_text:
                                tokens = [j for j in before_tokens if j == -1]+tokens
                            if unaligned_text:
                                text_tokens = []
                                other_tokens = []
                                for token in tokens:
                                    if token >= 0 and token < self.tokenizer_vq_start:
                                        text_tokens.append(token)
                                    else:
                                        other_tokens.append(token)
                                tokens = text_tokens+other_tokens
                            if fixed_text_token:
                                tokens = [tok if (tok >= self.tokenizer_vq_start or (fixed_text_token_not_space and tok == 220) or (fixed_text_token_not_punctuation and tok in punctuation_tokens)) else self.tokenizer_vq_start+codebook_size for tok in tokens]
                            tokens = [self.tokenizer.bos_token_id]+tokens
                            print(tokens)
                            print(self.tokenizer.decode(tokens))
                            data_dict[name]["tokens"] = tokens
                            data_dict[name]["str_rep"] = full_str_rep
                        for key in ["p0_pose", "p0_exp", "p0_shape", "p0_detail", "p1_pose", "p1_exp"]:
                            start_index = start_frame-overall_start
                            max_length = self.max_motion_length
                            assert start_index >= 0
                            data_dict[name][key] = segments_dict[dict_name][key][start_index:end_frame-overall_start,:].numpy()
                            # if key == 'p0_pose':
                            #     print(key, data_dict[name][key].shape)
                            if data_dict[name][key].shape[0] < max_length:
                                data_dict[name][key] = np.concatenate((
                                    data_dict[name][key],
                                    np.zeros_like(data_dict[name][key][:1,:]).repeat(max_length-data_dict[name][key].shape[0], axis=0),
                                ), axis=0)
        else:
            split_file = pjoin(self.data_root, 'train.txt')


            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            for name in tqdm(id_list):
                try:
                    m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))

                    # Read text
                    with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()

                        for line in lines:
                            try:
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                t_tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = t_tokens
                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                    if len(m_token_list_new) == 0:
                                        continue
                                    new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                    data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                           'text':[text_dict]}
                                    new_name_list.append(new_name)
                            except:
                                pass

                    if flag:
                        data_dict[name] = {'m_token_list': m_token_list,
                                           'text':text_data}
                        new_name_list.append(name)
                except:
                    pass
        self.data_dict = data_dict
        with open('dataset/full_data_2-19_c128.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        self.name_list = new_name_list
        print('Data length:', len(self.data_dict))
        self.unit_length = unit_length

    def __len__(self):
        return len(self.data_dict)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        caption = (data["before_text"], data["during_text"])
        speaker_inputs = []
        m_tokens = torch.LongTensor(data["tokens"])
        m_tokens_len = torch.ones_like(m_tokens)
        if m_tokens.shape[0] < self.max_tokens_length:
            m_tokens = torch.cat((m_tokens, torch.zeros((self.max_tokens_length-m_tokens.shape[0])).long()), dim=0)
            m_tokens_len = torch.cat((m_tokens_len, torch.zeros((self.max_tokens_length-m_tokens_len.shape[0])).long()), dim=0)
        assert m_tokens.shape[0] == self.max_tokens_length, str(m_tokens.shape[0])+', '+str(self.max_tokens_length)
        assert m_tokens_len.shape[0] == self.max_tokens_length, str(m_tokens_len.shape[0])+', '+str(self.max_tokens_length)

        if self.evaluation:
            if isinstance(caption, tuple):
                return *caption, m_tokens.reshape(-1), m_tokens_len, *speaker_inputs, data["p0_exp"], data["p0_pose"], data["p0_shape"], data["p0_detail"], self.name_list[item]
            else:
                return caption, m_tokens.reshape(-1), m_tokens_len, *speaker_inputs, data["p0_exp"], data["p0_pose"], data["p0_shape"], data["p0_detail"], self.name_list[item]
        if isinstance(caption, tuple):
            return *caption, m_tokens.reshape(-1), m_tokens_len, *speaker_inputs
        else:
            return caption, m_tokens.reshape(-1), m_tokens_len, *speaker_inputs


def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 1,
                split="train",
                max_motion_length=None,
                include_speaker_motion=False,
                include_audio=False,
                evaluation=False,
                gpt2_config=None,
                no_text=False,
                max_tokens=None,
                no_before_text=False,
                max_time_before=None,
                fps=30,
                fixed_text_token=False,
                fixed_text_token_not_space=False,
                fixed_text_token_not_punctuation=False,
                unaligned_text=False,
                remove_space_before_vq_tokens=False,
                random_text_token_order=False):

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, split=split, max_motion_length=max_motion_length, 
                                                                  evaluation=evaluation, gpt2_config=gpt2_config, no_text=no_text, max_tokens=max_tokens, no_before_text=no_before_text, max_time_before=max_time_before,
                                                                  fps=fps,
                                                                  fixed_text_token=fixed_text_token, fixed_text_token_not_space=fixed_text_token_not_space, fixed_text_token_not_punctuation=fixed_text_token_not_punctuation, unaligned_text=unaligned_text, remove_space_before_vq_tokens=remove_space_before_vq_tokens, random_text_token_order=random_text_token_order),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = False)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


