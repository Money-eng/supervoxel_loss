"""
Created on Fri November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Supervoxel-based topological loss function for training affinity-based neural
networks to perform instance segmentation.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import label
from supervoxel_loss.critical_detection_2d import detect_critical
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn


class SuperVoxel(nn.Module):
    """
    Supervoxel-based topological loss function for training neural networks to
    perform instance segmentation on 2D images.

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
        super(SuperVoxel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.device = device
        self.threshold = threshold

    def forward(self, preds, targets):
        # Initializations
        loss = self.criterion(preds, targets)
        targets = np.array(targets[:, 0, ...].cpu().detach())
        preds = np.array(preds[:, 0, ...].cpu().detach())
        binary_preds = (preds > self.threshold).astype(np.float32)

        # Compute loss
        masks = self.get_critical_masks(binary_preds, targets)
        for i in range(preds.shape[0]):
            term_1 = (1 - self.alpha) * loss[i, ...]
            term_2 = self.alpha * masks[i, ...] * loss[i, ...]
            loss[i, ...] = term_1 + term_2
        return loss.mean()

    def get_critical_masks(self, preds, targets):
        critical_masks = np.zeros(preds.shape)
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            for i in range(preds.shape[0]):
                pred, _ = label(preds[i, ...])
                target, _ = label(targets[i, ...])
                processes.append(
                    executor.submit(get_critical_mask, target, pred, i, -1)
                )
                processes.append(
                    executor.submit(get_critical_mask, pred, target, i, 1)
                )

            # Store results
            for process in as_completed(processes):
                i, mask_i, critical_type = process.result()
                if critical_type > 0:
                    critical_masks[i, ...] += (1 - self.beta) * mask_i
                else:
                    critical_masks[i, ...] += self.beta * mask_i
        return self.toGPU(critical_masks)

    def toGPU(self, arr):
        """
        Converts "arr" to a tensor and moves to GPU.

        Parameters
        ----------
        arr : torch.array
            Array to be converted to a tensor and moved to GPU.

        Returns
        -------
        torch.tensor
            Tensor on GPU.

        """
        if type(arr) == np.ndarray:
            arr[np.newaxis, ...] = arr
            arr = torch.from_numpy(arr)
        return Variable(arr).to(self.device, dtype=torch.float32)


def get_critical_mask(target_labels, pred_labels, process_id, critical_type):
    """
    Compute the critical mask for a given examples and returns associated
    metadata.

    Parameters
    ----------
    target : numpy.ndarray
        Ground truth labels.
    pred : numpy.ndarray
        Predicted labels.
    process_id : int
        A unique identifier for an example in a given batch.
    critical_type : int
        An integer indicating whether to compute positive or negative critical
        components based on whether its sign is positive or negative.

    Returns
    -------
    tuple
        A tuple containing:
        - "process_id" : A unique identifier for an example in a given batch.
        - "mask" : A binary mask indicating the critical components.
        - "n_criticals" : Number of detected critical components.
        - "critical_type" : Type of critical component computed.

    """
    mask = detect_critical(target_labels, pred_labels)
    return process_id, mask, critical_type
