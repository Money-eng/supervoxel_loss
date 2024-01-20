"""
Created on Sun November 05 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Supervoxel-based topological loss function for training neural networks to
perform instance segmentation.

"""

import cupy
import numpy as np
from supervoxel_loss.critical_detection_3d import detect_critical
from torch import nn
from torch.utils.dlpack import to_dlpack, from_dlpack


class SuperVoxelLoss(nn.Module):
    """
    Supervoxel-based topological loss function.

    """

    def __init__(
        self, alpha=10.0, beta=10.0, criterion=None, return_cnts=False
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
        alpha : float, optional
            Weight penalty applied to negatively critical components.
            The default value is 10.0.
        beta : float, optional
            Weight penalty appplied to positively critical components.
            The default value is 10.0.
        criterion : torch.nn.modules.loss
            Loss function that is used to penalize critical components.
            The default value is None.
        return_cnts : bool, optional
            Indicates whether to return the number of negatively and
            positively critical components.
            The default value is False.

        Returns
        -------
        None

        """
        super(SuperVoxelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, affs_pred, y_pred, y_target):
        """
        Computes the loss with respect to "y_pred" and y_target".

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation.
        y_target : torch.Tensor
            Target segmentation.

        Returns
        -------
        torch.Tensor
            Value of loss function.

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


# Utils
def to_cpu(arr):
    return np.array(arr.detach().cpu())


def to_cupy(arr):
    return cupy.from_dlpack(to_dlpack(arr))
