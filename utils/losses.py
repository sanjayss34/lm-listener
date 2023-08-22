import torch
import torch.nn as nn

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, pose_alpha):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        self.nb_joints = None
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        if nb_joints is not None:
            self.nb_joints = nb_joints
            self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        else:
            self.jaw_alpha = 1.0
            self.pose_alpha = pose_alpha
        
    def forward(self, motion_pred, motion_gt) :
        if self.nb_joints is not None:
            loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        else:
            exp_loss = self.Loss(motion_pred[:,:,:50], motion_gt[:,:,:50])
            rot_loss = self.Loss(motion_pred[:,:,50:53], motion_gt[:,:,50:53])*self.pose_alpha
            jaw_loss = self.jaw_alpha * self.Loss(motion_pred[:,:,53:56], motion_gt[:,:,53:56])
            loss = exp_loss+rot_loss+jaw_loss
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) :
        if self.nb_joints is None:
            vel_pred = torch.cat((
                torch.zeros_like(motion_pred[:,:1,:]),
                motion_pred[:,1:,:]-motion_pred[:,:-1,:]
            ), dim=1)
            vel_gt = torch.cat((
                torch.zeros_like(motion_gt[:,:1,:]),
                motion_gt[:,1:,:]-motion_gt[:,:-1,:]
            ), dim=1)
            loss = self.Loss(vel_pred, vel_gt)
        else:
            loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
    
    
