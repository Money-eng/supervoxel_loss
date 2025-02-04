"""
Created on Sun November 05 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Supervoxel-based topological loss function for training neural networks to
perform instance segmentation.


TO DO: UPDATE WITH REDEFINITIONS OF ALPHA AND BETA
"""

import cupy
import numpy as np
from supervoxel_loss.critical_detection_3d import detect_critical
from torch import nn
from torch.utils.dlpack import from_dlpack


class SuperVoxelLoss(nn.Module):
    """
    Supervoxel-based loss function for training neural networks to perform
    instance segmentation.

    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.5,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device=0,
        threshold=0.5,
    ):
        """
        Instantiates SuperVoxelLoss module.

        Parameters
        ----------
        alpha : float, optional
            Scaling factor that controls the relative importance of voxel-
            versus structure-level mistakes. The default is 0.5.
        beta : float, optional
            Scaling factor that controls the relative importance of split
            versus merge mistakes. The default is 0.5.
        criterion : torch.nn.modules.loss
            Loss function used to penalize voxel- and structure-level
            mistakes. If provided, must set "reduction=None". The default is
            nn.BCEWithLogitsLoss.
        device : int, optional
            Device (e.g. cpu or gpu id) used to train model. The default is 0.
        threshold : float
            Theshold used to binarize predictions. The defulat is 0.5.

        Returns
        -------
        None

        """
        super(SuperVoxelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.device = device
        self.threshold = threshold

    def forward(self, y_pred, y_target):
        """
        Computes the loss with respect to "y_pred" and y_target".

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentations from batch.
        y_target : torch.Tensor
            Target segmentation from patch.

        Returns
        -------
        torch.Tensor
            Computed loss for the given batch.

        """
        # Detect critical components
        y_pred = np.array(y_pred.cpu().detach())
        if self.alpha > 0:
            neg_critical_mask, neg_critical_cnt = self.run_detection(
                y_pred, y_target
            )

        if self.beta > 0:
            pos_critical_mask, pos_critical_cnt = self.run_detection(
                y_pred, y_target
            )

        # Compute loss
        loss = self.criterion(y_pred, y_target)
        neg_loss = self.alpha * neg_critical_mask * loss if self.alpha else 0
        pos_loss = self.beta * pos_critical_mask * loss if self.beta else 0
        loss = (loss + neg_loss + pos_loss).mean()

        # Return
        if self.return_cnts:
            return loss, neg_critical_cnt, pos_critical_cnt
        else:
            return loss

    def run_detection(self, y_pred, y_target):
        cnts = []
        with cupy.cuda.Device(0):
            mask = cupy.zeros(y_pred.shape)
            for i in range(y_pred.shape[0]):
                mask[i, :, ...], cnt_i = detect_critical(
                    y_pred[i, 0, ...], y_target[i, 0, ...]
                )
                cnts.append(cnt_i)
        return from_dlpack(mask.toDlpack()), cnts
