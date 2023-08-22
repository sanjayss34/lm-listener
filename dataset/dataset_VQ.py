import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import json
import codecs as cs
from tqdm import tqdm


"""
python train_vq.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 256 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname audio_trevor --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name audio_p8 --window-size 32
"""

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, split = "train", add_velocity=False, person_id = 0):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name.split('_')[0] == 'face':
            self.data_root = './dataset/'+dataset_name.split('_')[-1]
            self.segments = torch.load(pjoin(self.data_root, 'segments_'+split+'.pth'), map_location='cpu')
            self.data = [torch.cat((seg['p'+str(person_id)+'_exp'], seg['p'+str(person_id)+'_pose']), dim=1).numpy() for seg in self.segments]
            # print(self.data[0].shape)
            # assert False
            self.fnames = [seg['fname'] for seg, faces in zip(self.segments, self.data) if faces.shape[0] >= window_size]
            self.starts = [seg['split_start_frame'] for seg, faces in zip(self.segments, self.data) if faces.shape[0] >= window_size]
            self.data = [datum for datum in self.data if datum.shape[0] >= window_size]
            self.lengths = [datum.shape[0] for datum in self.data]
            assert len(self.fnames) == len(self.data) == len(self.lengths) == len(self.starts)

        self.split = split
        mean = np.load(pjoin(self.data_root, 'p'+str(person_id)+'_mean.npy'))
        std = np.load(pjoin(self.data_root, 'p'+str(person_id)+'_std.npy'))

        self.add_velocity = add_velocity

        print(dataset_name.split('_')[0])
        if dataset_name.split('_')[0] != 'face' and dataset_name.split('_')[0] != 'audio':
            self.data = []
            self.lengths = []
            id_list = []
            self.names = []
            split_file = pjoin(self.data_root, split+'.txt')
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if motion.shape[0] < self.window_size:
                        continue
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.data.append(motion)
                    self.names.append(name)
                except:
                    # Some motion may not exist in KIT dataset
                    pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
       
        # print(len(motion), self.data[item].shape)
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        """if self.add_velocity:
            vel = np.concatenate((
                np.zeros((1, motion.shape[-1]), dtype=motion.dtype),
                motion[1:] - motion[:-1]
            ), axis=0)
            motion = np.concatenate((motion, vel), axis=1)"""

        if "train" in self.split:
            return motion
        return motion, np.array([self.window_size]).squeeze(), self.fnames[item]+'_'+str(self.starts[item]+idx)

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               split = "train",
               person_id = 0):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, split=split, person_id=person_id)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = (split == "train"))
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
