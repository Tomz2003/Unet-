import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
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


def plot_segmentation_results(image_tensor, mask_tensor, pred_tensor, save_path=None):
    """
    Hàm vẽ kết quả phân đoạn 1x4: Original, Predicted, Mask, Overlayed.
    """

    img = image_tensor.cpu().numpy()
    if len(img.shape) == 3 and img.shape[0] in [1, 3]: 
        img = np.transpose(img, (1, 2, 0))
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    

    if len(img.shape) == 3 and img.shape[-1] == 3:
        img = img[:, :, ::-1] 
    

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    
    mask = mask_tensor.squeeze().cpu().numpy()

    
    pred = torch.sigmoid(pred_tensor).squeeze().cpu().detach().numpy()
    pred_binary = (pred > 0.5).astype(np.uint8)

  
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary[0]
        mask = mask[0] if len(mask.shape) > 2 else mask

    if len(img.shape) == 2: 
        overlayed = np.stack((img,)*3, axis=-1)
    else:
        overlayed = img.copy()
        
    lime_green = np.array([0.4, 1.0, 0.1])
    alpha = 0.4
        
    overlayed[pred_binary == 1] = overlayed[pred_binary == 1] * (1 - alpha) + lime_green * alpha
    overlayed = np.clip(overlayed, 0, 1)


    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img, cmap='gray' if len(img.shape)==2 else None)
    axes[0].set_title('original', fontsize=14)
    axes[1].imshow(pred_binary, cmap='gray')
    axes[1].set_title('predicted', fontsize=14)
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('mask', fontsize=14)
    axes[3].imshow(overlayed)
    axes[3].set_title('overlayed', fontsize=14)
    
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=10)
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()