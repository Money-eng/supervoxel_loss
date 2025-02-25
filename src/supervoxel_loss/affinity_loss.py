"""
Created on Fri November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of supervoxel-based loss function for training affinity-based
neural networks to perform instance segmentation.

Note: We use the term "labels" to refer to a segmentation.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.autograd import Variable
from waterz import agglomerate as run_watershed

import numpy as np
import torch
import torch.nn as nn

from supervoxel_loss.critical_detection_3d import detect_critical


class SuperVoxelAffinity(nn.Module):
    """
    Supervoxel-based loss function for training neural networks to perform
    affinity-based instance segmentation.

    """

    def __init__(
        self,
        edges,
        alpha=0.5,
        beta=0.5,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device=0,
        return_cnts=False,
        threshold=0.5,
    ):
        """
        Instantiates a SuperVoxelLoss object with the given parameters.

        Parameters
        ----------
        edges : List[Tuple[int]]
            Edge affinities learned by model (e.g. [[1, 0, 0], [0, 1, 0],
            [0, 0, 1]]).
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
            Device on which to train model. The default is "cuda".
        return_cnts : bool, optional
            Indication of whether to return the number of negatively and
            positively critical components. The default is False.
        threshold : float, optional
            Theshold used to binarize predictions. The defulat is 0.5.

        Returns
        -------
        None

        """
        # Call parent class
        super(SuperVoxelAffinity, self).__init__()

        # Instance attributes
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.decoder = SuperVoxelAffinity.Decoder(edges)
        self.device = device
        self.edges = list(edges)
        self.return_cnts = return_cnts
        self.threshold = threshold

    def forward(self, pred_affs, target_labels):
        """
        Computes the loss for a batch by comparing predicted and target
        affinites and scales loss per voxel for critical components.

        Parameters
        ----------
        pred_affs : torch.Tensor
            Predicted affinities with shape (batch_size, num_edges, height,
            width, depth).
        target_labels : torch.Tensor
            Target labels with shape (batch_size, height, width, depth)
            representing the ground truth labels. Note: target labels are
            converted into affinities to compute loss.

        Returns
        -------
        torch.Tensor
            Computed loss for the given batch.
        dict
            Stats related to the critical components for the batch, such as
            the number of positively and negatively critical components.

        """
        # Compute critical components
        pred_labels = self.affs_to_labels(pred_affs)
        masks, stats = self.get_critical_masks(pred_labels, target_labels)

        # Compute loss
        loss = 0
        for i in range(pred_affs.size(0)):
            mask_i = self.toGPU(masks[i, ...])
            target_labels_i = self.toGPU(target_labels[i, ...])
            for j, edge in enumerate(self.edges):
                # Compute affinities
                pred_affs_j = self.decoder(pred_affs[i, ...], j)
                target_affs_j = get_aff(target_labels_i, edge)
                mask_aff_j = get_aff(mask_i, edge)

                # Compute loss
                loss_j = self.criterion(pred_affs_j, target_affs_j)
                term_1 = (1 - self.alpha) * loss_j
                term_2 = self.alpha * mask_aff_j * loss_j
                loss += (term_1 + term_2).mean()
        return loss, stats

    # --- critical component detection ---
    def affs_to_labels(self, affs):
        """
        Converts predicted affinities to predicted labels by decoding the
        affinities.

        Parameters
        ----------
        affs : torch.Tensor
            Tensor containing predicted affinities from a batch.

        Returns
        -------
        List[numpy.ndarray]
            List of predicted labels for each example in the batch.

        """
        affs = np.array(affs.detach().cpu(), np.float32)
        labels = []
        for i in range(affs.shape[0]):
            binary_affs = (affs[i, ...] > self.threshold).astype(np.float32)
            iterator = run_watershed(binary_affs, [0])
            labels.append(next(iterator).astype(int))
        return labels

    def get_critical_masks(self, preds, targets):
        """
        Computes critical masks for predicted labels.

        Parameters
        ----------
        preds : List[torch.Tensor]
            List of predicted labels from a batch.
        targets : List[torch.Tensor]
            List of groundtruth labels from a batch.

        Returns
        -------
        tuple
            A tuple containing the following:
            - torch.Tensor: critical component masks for a batch.
            - dict: Dictionary containing the following stats:
              - "Splits": Average number of negatively critical components.
              - "Merges": Average number of positively critical components.

        """
        # Initializations
        masks = np.zeros((len(preds),) + preds[0].shape)
        stats = {"Splits": 0, "Merges": 0}
        targets = np.array(targets, dtype=int)

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            for i in range(len(preds)):
                processes.append(
                    executor.submit(
                        get_critical_mask, targets[i, 0, ...], preds[i], i, -1
                    )
                )
                processes.append(
                    executor.submit(
                        get_critical_mask, preds[i], targets[i, 0, ...], i, 1
                    )
                )

            # Store results
            for process in as_completed(processes):
                i, mask_i, n_criticals, crtitical_type = process.result()
                if crtitical_type == -1:
                    masks[i, ...] += self.beta * mask_i
                    stats["Splits"] += n_criticals / len(preds)
                else:
                    masks[i, ...] += (1 - self.beta) * mask_i
                    stats["Merges"] += n_criticals / len(preds)
        return self.toGPU(masks), stats

    def toGPU(self, arr):
        """
        Converts "arr" to a tensor and moves it to the GPU.

        Parameters
        ----------
        arr : numpy.array
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

    class Decoder(nn.Module):
        """
        Decoder module for processing edge affinities in the
        SuperVoxelAffinity loss function.

        """

        def __init__(self, edges):
            """
            Initializes Decoder object with the given edge affinities.

            Parameters
            ----------
            edges : List[Tuple[int]]
                Edge affinities learned by model (e.g. [[1, 0, 0], [0, 1, 0],
                [0, 0, 1]]).

            Returns
            -------
            None

            """
            super(SuperVoxelAffinity.Decoder, self).__init__()
            self.edges = list(edges)

        def forward(self, affs, i):
            """
            Extracts the predicted affinity for the i-th edge from the input
            tensor.

            Parameters
            ----------
            affs : torch.Tensor
                Predicted affinities for a single example.
            i : int
                Index of specific edge in "self.edges".

            Returns
            -------
            torch.Tensor
                Affinities corresponding to the i-th edge.

            """
            n_channels = affs.size(-4)
            assert n_channels == len(self.edges)
            assert i < n_channels and i >= 0
            return get_pair_first(affs[..., [i], :, :, :], self.edges[i])


# --- helpers ---
def get_critical_mask(target, pred, process_id, critical_type):
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
    mask, n_criticals = detect_critical(target, pred)
    return process_id, mask, n_criticals, critical_type


def get_aff(labels, edge):
    """
    Computes affinities for labels based on the given edge.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor containing the segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity.

    Returns
    -------
    torch.Tensor
        Binary tensor, where each element indicates the affinity for each
        voxel based on the given edge.

    """
    o1, o2 = get_pair(labels, edge)
    ret = (o1 == o2) & (o1 != 0)
    return ret.type(labels.type())


def get_pair(labels, edge):
    """
    Extracts two subarrays from "labels" by using the given edge affinity as
    an offset.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor containing the segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity.

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing two tensors:
        - "arr1": Subarray extracted based on the edge affinity.
        - "arr2": Subarray extracted based on the negative of the edge
                  affinity.

    """
    shape = labels.size()[-3:]
    edge = np.array(edge)
    offset1 = np.maximum(edge, 0)
    offset2 = np.maximum(-edge, 0)

    labels1 = labels[
        ...,
        offset1[0] : shape[0] - offset2[0],
        offset1[1] : shape[1] - offset2[1],
        offset1[2] : shape[2] - offset2[2],
    ]
    labels2 = labels[
        ...,
        offset2[0] : shape[0] - offset1[0],
        offset2[1] : shape[1] - offset1[1],
        offset2[2] : shape[2] - offset1[2],
    ]
    return labels1, labels2


def get_pair_first(labels, edge):
    """
    Gets subarray of "labels" based on the given edge affinity which defines
    an offset. Note this subarray will be used to compute affinities.

    Parameters
    ----------
    labels : torch.Tensor
        Segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity that defines the offset of the subarray.

    Returns
    -------
    torch.Tensor
        Subarray of "labels" based on the given edge affinity.

    """
    shape = labels.size()[-3:]
    edge = np.array(edge)
    offset1 = np.maximum(edge, 0)
    offset2 = np.maximum(-edge, 0)
    ret = labels[
        ...,
        offset1[0] : shape[0] - offset2[0],
        offset1[1] : shape[1] - offset2[1],
        offset1[2] : shape[2] - offset2[2],
    ]
    return ret
