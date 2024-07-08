import torch
import torch.nn.functional as F
import numpy as np
import pickle

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_mean_std_ignore_zeros(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    
    # Reshape the tensor and create a mask for non-zero elements
    feat_reshaped = feat.view(N, C, -1)
    mask = feat_reshaped != 0
    
    # Calculate mean ignoring zeros
    feat_sum = feat_reshaped.sum(dim=2)
    count_non_zero = mask.sum(dim=2)
    feat_mean = (feat_sum / (count_non_zero + eps)).view(N, C, 1, 1)
    

    # Calculate variance ignoring zeros
    feat_squared = (feat_reshaped ** 2) * mask
    feat_sum_squared = feat_squared.sum(dim=2)
    feat_var = (feat_sum_squared / (count_non_zero + eps)) - (feat_mean.squeeze(-1).squeeze(-1) ** 2)
    feat_std = (feat_var + eps).sqrt().view(N, C, 1, 1)
    
    return feat_mean, feat_std


def calc_weighted_mean_std(feat, weights, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    
    # Reshape the tensor and weights
    feat_reshaped = feat.view(N, C, -1)
    if weights.size()[0] != N:
        weights = weights.repeat(N, 1, 1, 1)
    weights_reshaped = weights.view(N, 1, -1)
    weights_reshaped = weights_reshaped.repeat(1, C, 1)  
    
    # Calculate weighted mean
    weighted_sum = (feat_reshaped * weights_reshaped).sum(dim=2)
    total_weights = weights_reshaped.sum(dim=2)
    feat_mean = (weighted_sum / (total_weights + eps)).view(N, C, 1, 1)
    
    # Calculate weighted variance
    squared_diff = (feat_reshaped - feat_mean.squeeze(-1).squeeze(-1).unsqueeze(-1)) ** 2
    weighted_squared_diff = (squared_diff * weights_reshaped).sum(dim=2)
    feat_var = weighted_squared_diff / (total_weights + eps)
    
    # Calculate weighted standard deviation
    feat_std = (feat_var + eps).sqrt().view(N, C, 1, 1)
    
    return feat_mean, feat_std
    


def adaptive_instance_normalization(content_feat, style_feat,content_sem,style_sem):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def adaptive_instance_normalization_by_segmentation(content_feat, style_feat, content_sem, style_sem):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    
    adaIN_feat = torch.zeros(size).to(content_feat.device)

    for class_id in torch.unique(content_sem):
        input_mask = F.interpolate((content_sem == class_id).float(), size = content_feat.shape[2:], mode = 'nearest')
        target_mask = F.interpolate((style_sem == class_id).float(), size = style_feat.shape[2:], mode = 'nearest')

        content_mean, content_std = calc_weighted_mean_std(content_feat,input_mask)

        if torch.any((style_sem == class_id).float() != 0):
            style_mean, style_std = calc_weighted_mean_std(style_feat,target_mask)
        else:
            style_mean, style_std = calc_mean_std(style_feat)

        print(style_mean.shape)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        normalized_feat = normalized_feat*input_mask
        adaIN_feat += normalized_feat * style_std.expand(size) + style_mean.expand(size)*input_mask

    return adaIN_feat


def adaptive_instance_normalization_precalculated(content_feat, style_feat, content_sem, style_sem):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    
    adaIN_feat = torch.zeros(size).to(content_feat.device)

    with open("mean_means.txt", "rb") as myFile:
        means = pickle.load(myFile)
    with open("mean_stds.txt", "rb") as myFile:
        stds = pickle.load(myFile)

    for class_id in torch.unique(content_sem):

        class_id_float = class_id.item()
        print(class_id_float)

        try:
            style_std = torch.from_numpy(stds[class_id_float]).to(content_feat.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        except KeyError:
            style_std = torch.zeros(size)
        try:
            style_mean = torch.from_numpy(means[class_id_float]).to(content_feat.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        except KeyError:
            style_mean = torch.zeros(size)
        
        # Calculate content mean and standard deviation for the current class
        input_mask = F.interpolate((content_sem == class_id).float(), size=size[2:], mode='nearest')
        content_mean, content_std = calc_weighted_mean_std(content_feat, input_mask)

        # Apply adaptive instance normalization
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        normalized_feat = normalized_feat * input_mask
        adaIN_feat += normalized_feat * style_std + style_mean*input_mask

    return adaIN_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
