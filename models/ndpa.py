"""
Normal Distribution-based Prior Assigner (NDPA)

Uses KL divergence between normal distributions to match priors with ground truth,
improving small object detection accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NDPA(nn.Module):
    """
    Normal Distribution-based Prior Assigner
    
    Models both ground truth bounding boxes and prior boxes as 2D normal distributions,
    then uses KL divergence to measure similarity for label assignment.
    """
    
    def __init__(self, pos_threshold=0.5, neg_threshold=0.3):
        """
        Args:
            pos_threshold: KL divergence threshold for positive samples (lower = positive)
            neg_threshold: KL divergence threshold for negative samples (higher = negative)
        """
        super(NDPA, self).__init__()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
    def bbox_to_gaussian(self, bboxes):
        """
        Convert bounding boxes to 2D normal distributions
        
        Args:
            bboxes: Tensor of shape (N, 4) with format [x_center, y_center, width, height]
        Returns:
            mu: Mean vectors of shape (N, 2) - [x_center, y_center]
            sigma: Covariance matrices of shape (N, 2, 2)
        """
        x_center = bboxes[:, 0]
        y_center = bboxes[:, 1]
        width = bboxes[:, 2]
        height = bboxes[:, 3]
        
        # Mean vector (center coordinates)
        mu = torch.stack([x_center, y_center], dim=1)  # (N, 2)
        
        # Covariance matrix: diagonal with half-width and half-height
        # Using half-width/height as standard deviation (kappa in paper)
        kappa_w = width / 2.0
        kappa_h = height / 2.0
        
        # Create diagonal covariance matrices
        # Sigma = [[kappa_w^2, 0], [0, kappa_h^2]]
        sigma = torch.zeros(bboxes.size(0), 2, 2, device=bboxes.device)
        sigma[:, 0, 0] = kappa_w ** 2
        sigma[:, 1, 1] = kappa_h ** 2
        
        # Add small epsilon for numerical stability
        sigma = sigma + torch.eye(2, device=bboxes.device).unsqueeze(0) * 1e-6
        
        return mu, sigma
    
    def kl_divergence_2d(self, mu1, sigma1, mu2, sigma2):
        """
        Compute KL divergence between two 2D normal distributions
        
        KL(P||Q) = 0.5 * [tr(Sigma2^-1 * Sigma1) + (mu2 - mu1)^T * Sigma2^-1 * (mu2 - mu1) 
                          - 2 + ln(det(Sigma2) / det(Sigma1))]
        
        Args:
            mu1, sigma1: Parameters of first distribution (priors)
            mu2, sigma2: Parameters of second distribution (ground truth)
        Returns:
            KL divergence of shape (N, M) where N=num_priors, M=num_gt
        """
        # Compute for all pairs: (N, 2) and (M, 2) -> (N, M, 2)
        mu_diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # (N, M, 2)
        
        # Inverse of sigma2: (M, 2, 2) -> (1, M, 2, 2)
        sigma2_inv = torch.inverse(sigma2)  # (M, 2, 2)
        sigma2_inv = sigma2_inv.unsqueeze(0)  # (1, M, 2, 2)
        
        # Trace term: tr(Sigma2^-1 * Sigma1)
        # sigma1: (N, 2, 2), sigma2_inv: (1, M, 2, 2)
        sigma1_expanded = sigma1.unsqueeze(1)  # (N, 1, 2, 2)
        trace_term = torch.diagonal(
            torch.matmul(sigma2_inv, sigma1_expanded), 
            dim1=-2, dim2=-1
        ).sum(dim=-1)  # (N, M)
        
        # Quadratic form: (mu2 - mu1)^T * Sigma2^-1 * (mu2 - mu1)
        mu_diff_expanded = mu_diff.unsqueeze(-1)  # (N, M, 2, 1)
        quadratic = torch.matmul(
            torch.matmul(mu_diff_expanded.transpose(-2, -1), sigma2_inv),
            mu_diff_expanded
        ).squeeze(-1).squeeze(-1)  # (N, M)
        
        # Determinant terms
        det_sigma1 = torch.det(sigma1)  # (N,)
        det_sigma2 = torch.det(sigma2)  # (M,)
        
        # Clamp determinants to avoid negative or zero values
        det_sigma1 = torch.clamp(det_sigma1, min=1e-10)
        det_sigma2 = torch.clamp(det_sigma2, min=1e-10)
        
        log_det = torch.log(det_sigma2.unsqueeze(0) / (det_sigma1.unsqueeze(1) + 1e-10))  # (N, M)
        
        # KL divergence
        kl = 0.5 * (trace_term + quadratic - 2 + log_det)
        
        # Clamp KL to reasonable range to prevent NaN
        kl = torch.clamp(kl, min=-100, max=100)
        
        return kl
    
    def assign_labels(self, priors, gt_boxes):
        """
        Assign positive/negative labels to priors based on KL divergence
        
        Args:
            priors: Tensor of shape (N, 4) - prior boxes [x, y, w, h]
            gt_boxes: Tensor of shape (M, 4) - ground truth boxes [x, y, w, h]
        Returns:
            labels: Tensor of shape (N,) - 1 for positive, 0 for negative, -1 for ignore
            matched_gt: Tensor of shape (N,) - index of matched GT box (-1 if no match)
        """
        if gt_boxes.size(0) == 0:
            # No ground truth, all negatives
            return torch.zeros(priors.size(0), dtype=torch.long, device=priors.device), \
                   torch.full((priors.size(0),), -1, dtype=torch.long, device=priors.device)
        
        # Convert to Gaussian distributions
        mu_p, sigma_p = self.bbox_to_gaussian(priors)
        mu_g, sigma_g = self.bbox_to_gaussian(gt_boxes)
        
        # Compute KL divergence for all pairs
        kl_matrix = self.kl_divergence_2d(mu_p, sigma_p, mu_g, sigma_g)  # (N, M)
        
        # Find best match for each prior (minimum KL divergence)
        min_kl, matched_indices = kl_matrix.min(dim=1)  # (N,)
        
        # Assign labels based on thresholds
        labels = torch.zeros(priors.size(0), dtype=torch.long, device=priors.device)
        labels[min_kl < self.pos_threshold] = 1  # Positive
        labels[min_kl > self.neg_threshold] = 0  # Negative
        # Between thresholds: ignore (already 0, but we can mark as -1 if needed)
        
        matched_gt = matched_indices.clone()
        matched_gt[min_kl >= self.pos_threshold] = -1  # No match
        
        return labels, matched_gt
    
    def forward(self, priors, gt_boxes):
        """
        Forward pass for NDPA
        
        Args:
            priors: Prior boxes (N, 4)
            gt_boxes: Ground truth boxes (M, 4)
        Returns:
            labels: Assigned labels (N,)
            matched_gt: Matched GT indices (N,)
            kl_matrix: KL divergence matrix (N, M) for visualization/debugging
        """
        if gt_boxes.size(0) == 0:
            mu_p, sigma_p = self.bbox_to_gaussian(priors)
            mu_g, sigma_g = self.bbox_to_gaussian(gt_boxes)
            kl_matrix = torch.zeros(priors.size(0), 0, device=priors.device)
        else:
            mu_p, sigma_p = self.bbox_to_gaussian(priors)
            mu_g, sigma_g = self.bbox_to_gaussian(gt_boxes)
            kl_matrix = self.kl_divergence_2d(mu_p, sigma_p, mu_g, sigma_g)
        
        labels, matched_gt = self.assign_labels(priors, gt_boxes)
        
        return labels, matched_gt, kl_matrix

