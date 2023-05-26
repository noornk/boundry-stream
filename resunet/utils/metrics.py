from torch import nn
import numpy
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient


# +
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target, pred):
        preds = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(preds, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (preds * truth).double().sum() + 1) / (
            preds.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, pred, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice
    
        smooth = 1.
        print("targets.shape, inputs.shape", targets.shape, inputs.shape)
        s_targets = SpatialGradient()(targets)[:, 0, :, :, :]
#         s_targets1 = SpatialGradient()(targets)[:, 0, :, :, :]
#         s_targets2 = SpatialGradient()(targets)[:, 1, :, :, :]
#         print(s_targets.shape)
#         print(s_targets.shape, targets.shape)

        # have to use contiguous since they may from a torch.view op
        iflat = inputs.contiguous().view(-1)
#         iflat = iflat > 0.7
        tflat = targets.contiguous().view(-1)
        t = tflat
        t[t<0.7] = 0
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        beta = torch.div(torch.count_nonzero(tflat), tflat.size(dim=0))
        loss_1 = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
        
        iflat = pred.contiguous().view(-1)
        tflat = s_targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        loss_2 = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
#         loss3 = nn.BCELoss()
#         loss3 = nn.BCEWithLogitsLoss()
#         loss_3 = loss3(pred, s_targets)
#         beta = nn.Parameter(1/100)
        
        loss_3 = beta*(-(pred.log()*s_targets) + (1-beta)*(s_targets)*(1-pred).log()).mean()
        
        return loss_1 + loss_2 + loss_3
#         return loss_1
#         return loss_3

    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, pred, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice
    
        smooth = 1.
        print("targets.shape, inputs.shape", targets.shape, inputs.shape)
        s_targets = SpatialGradient()(targets)[:, 0, :, :, :]
#         s_targets1 = SpatialGradient()(targets)[:, 0, :, :, :]
#         s_targets2 = SpatialGradient()(targets)[:, 1, :, :, :]
#         print(s_targets.shape)
#         print(s_targets.shape, targets.shape)

        # have to use contiguous since they may from a torch.view op
        iflat = inputs.contiguous().view(-1)
        print(iflat.shape)
#         iflat = iflat > 0.7
        tflat = targets.contiguous().view(-1)
#         t = tflat
#         t[t<0.7] = 0
#         intersection = (iflat * tflat).sum()

#         A_sum = torch.sum(tflat * iflat)
#         B_sum = torch.sum(tflat * tflat)
#         beta = torch.div(torch.count_nonzero(tflat), tflat.size(dim=0))
#         loss_1 = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
        alpha=0.5
        beta=0.25
        pts = np.reshape(flattened_pts, (int(len(flattened_pts)/2), 2))
    
        # external energy (favors low values of distance image)
        dist_vals = ndimage.interpolation.map_coordinates(edge_dist, [pts[:,0], pts[:,1]], order=1)
        edge_energy = np.sum(dist_vals)
        external_energy = edge_energy

        # spacing energy (favors equi-distant points)
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        displacements = pts - prev_pts
        point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)
        mean_dist = np.mean(point_distances)
        spacing_energy = np.sum((point_distances - mean_dist)**2)

        # curvature energy (favors smooth curves)
        curvature_1d = prev_pts - 2*pts + next_pts
        curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2)
        curvature_energy = np.sum(curvature)

        loss_1 = external_energy + alpha*spacing_energy + beta*curvature_energy


        
        return loss_1


# -

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()


# +
import torch


SMOOTH = 1e-6

def iou_pytorch(input, target):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    num_in_target = input.size(0)

    smooth = 1e-6

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)
    intersection = (input * target).long().sum()
    union = (
        input.long().sum()
        + target.long().sum()
        - intersection
    )
#     print(union)
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    if union == 0:
        iou = float("nan")
    else:
        iou = float(intersection) / float(max(union, 1))
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#     print(iou)
    
#     thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou # Or thresholded.mean() if you are interested in average across the batch
