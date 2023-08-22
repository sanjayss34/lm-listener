import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8, split = "train", max_motion_length = None, delay_start_frames=0, fps=30, min_length=0):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        min_motion_len = 40 if dataset_name =='t2m' else 24
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.kit_kinematic_chain
        elif dataset_name.split('_')[0] == 'face':
            self.data_root = './dataset/'+dataset_name.split('_')[-1]
            self.data = torch.load(pjoin(self.data_root, "segments_"+split+".pth"), map_location="cpu")
            self.lengths = [datum["split_end_frame"]-datum["split_start_frame"] for datum in self.data]
            self.names = [datum["fname"]+"_"+str(datum["split_start_frame"]) for datum in self.data]
            self.max_motion_length = None

        min_motion_len = max(min_motion_len, min_length)
        if max_motion_length is not None:
            self.max_motion_length = max_motion_length
        self.split = split
        mean = np.load(pjoin(self.data_root, 'mean.npy'))
        std = np.load(pjoin(self.data_root, 'std.npy'))
        person_id = 0
        
        if dataset_name.split('_')[0] == 'face':
            data_dict = {}
            length_list = []
            new_name_list = []
            step_size = self.max_motion_length
            # print(person_id, unit_length, history_size)
            # print(unit_length, history_size)
            for i in range(len(self.data)):
                motion_len = self.lengths[i]
                for start in range(delay_start_frames, motion_len, step_size):
                    segment_len = min(motion_len-start, self.max_motion_length)
                    if segment_len >= min_motion_len: #  and motion_len <= self.max_motion_length:
                        s = start
                        e = start+segment_len
                        length_list.append(e-s)
                        parts = self.names[i].split('_')
                        new_name_list.append('_'.join(parts[:-1])+'_'+str(int(parts[-1])+start))
                        # print('NAME', split, new_name_list[-1], s, e, fix_vq_tokenizer_start_frame)
                        if dataset_name.split('_')[0] == 'face':
                            curr_motion = torch.cat((self.data[i]['p'+str(person_id)+'_exp'][s:e,:], self.data[i]['p'+str(person_id)+'_pose'][s:e,:]), dim=1).numpy()
                        data_dict[new_name_list[-1]] = {
                            'motion': curr_motion,
                            'length': segment_len,
                            'name': new_name_list[-1],
                        }
        else:
            split_file = pjoin(self.data_root, split+'.txt')
            joints_num = self.joints_num
            
            data_dict = {}
            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            new_name_list = []
            length_list = []
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                        continue

                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'name': name}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                except:
                    # Some motion may not exist in KIT dataset
                    pass


        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4, split = "train", max_motion_length = None, delay_start_frames=0, fps=30, min_length=0) : 
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length, split=split, max_motion_length=max_motion_length, delay_start_frames=delay_start_frames, fps=fps, min_length=min_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
