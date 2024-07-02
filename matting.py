from __future__ import division
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import torch

class MattingLaplacian(torch.nn.Module):
    def __init__(self, eps=1e-7, win_rad=1):
        super(MattingLaplacian, self).__init__()
        self.eps = eps
        self.win_rad = win_rad

    @staticmethod
    def rolling_block(A, block=(3, 3)):
        """Applies sliding window to given matrix."""
        shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
        strides = (A.strides[0], A.strides[1]) + A.strides
        return as_strided(A, shape=shape, strides=strides)

    def compute_laplacian(self, img, mask=None):
        """Computes Matting Laplacian for a given image."""
        win_rad = self.win_rad
        eps = self.eps
        win_size = (win_rad * 2 + 1) ** 2
        h, w, d = img.shape
        c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
        win_diam = win_rad * 2 + 1
        
        indsM = np.arange(h * w).reshape((h, w))
        ravelImg = img.reshape(h * w, d)
        win_inds = self.rolling_block(indsM, block=(win_diam, win_diam))
        win_inds = win_inds.reshape(c_h, c_w, win_size)
        
        if mask is not None:
            mask = cv2.dilate(
                mask.astype(np.uint8),
                np.ones((win_diam, win_diam), np.uint8)
            ).astype(bool)
            win_mask = np.sum(mask.ravel()[win_inds], axis=2)
            win_inds = win_inds[win_mask > 0, :]
        else:
            win_inds = win_inds.reshape(-1, win_size)

        winI = ravelImg[win_inds]
        win_mu = np.mean(winI, axis=1, keepdims=True)
        win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

        inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))
        X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
        vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

        nz_indsCol = np.tile(win_inds, win_size).ravel()
        nz_indsRow = np.repeat(win_inds, win_size).ravel()
        nz_indsVal = vals.ravel()
        L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
        return L

    def forward(self, img):
            """
            Compute the Matting Laplacian loss for a given image.
            
            Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
            
            Returns:
            torch.Tensor: Matting Laplacian loss
            """
            if img.dim() != 4:
                raise ValueError("Expected 4D tensor as input")
            
            batch_size, channels, height, width = img.shape
            loss = 0
            
            for b in range(batch_size):
                img_np = img[b].permute(1, 2, 0).detach().numpy()  # Convert to numpy array
                lap = self.compute_laplacian(img_np)
                
                # Convert scipy sparse matrix to torch sparse tensor more efficiently
                coo = lap.tocoo()
                indices = np.vstack((coo.row, coo.col))
                indices = torch.LongTensor(indices).to(img.device)
                values = torch.FloatTensor(coo.data).to(img.device)
                shape = torch.Size(coo.shape)
                
                lap_tensor = torch.sparse_coo_tensor(indices, values, shape)
                
                img_flat = img[b].view(channels, -1)
                loss += sum(torch.sum((img_flat[c] @ lap_tensor) * img_flat[c]) for c in range(channels))
            
            return loss / batch_size

def compute_lap(path_img):
    """
    Compute Laplacian matrix for an image.
    
    Args:
    path_img (str): Path to the input image
    
    Returns:
    torch.sparse.FloatTensor: Sparse tensor representation of the Laplacian matrix
    """
    image = cv2.imread(path_img, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    
    matting = MattingLaplacian()
    M = matting.compute_laplacian(image)
    
    indices = torch.LongTensor([M.row, M.col])
    values = torch.FloatTensor(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape).cuda()