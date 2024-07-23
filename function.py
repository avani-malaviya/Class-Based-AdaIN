import torch
import torch.nn.functional as F
import pickle

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



def calc_weighted_mean_std(feat, weights, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    
    feat_reshaped = feat.view(N, C, -1)
    if weights.size()[0] != N:
        weights = weights.repeat(N, 1, 1, 1)
    weights_reshaped = weights.view(N, 1, -1)
    weights_reshaped = weights_reshaped.repeat(1, C, 1)  

    weighted_sum = (feat_reshaped * weights_reshaped).sum(dim=2)
    total_weights = weights_reshaped.sum(dim=2)
    feat_mean = (weighted_sum / (total_weights + eps)).view(N, C, 1, 1)

    squared_diff = (feat_reshaped - feat_mean.squeeze(-1).squeeze(-1).unsqueeze(-1)) ** 2
    weighted_squared_diff = (squared_diff * weights_reshaped**2).sum(dim=2)
    feat_var = weighted_squared_diff / (total_weights + eps)

    feat_std = (feat_var + eps).sqrt().view(N, C, 1, 1)
    
    return feat_mean, feat_std, total_weights



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
    
    total_style_mean = torch.zeros(size).to(content_feat.device)
    total_style_var = torch.zeros(size).to(content_feat.device)
    total_normalized_feat = torch.zeros(size).to(content_feat.device)

    content_mean, content_std = calc_mean_std(content_feat)

    total_normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    for class_id in torch.unique(content_sem):
        input_mask = F.interpolate((content_sem == class_id).float(), size = content_feat.shape[2:], mode = 'bilinear')
        target_mask = F.interpolate((style_sem == class_id).float(), size = style_feat.shape[2:], mode = 'bilinear')

        if torch.any((style_sem == class_id).float() != 0):
            style_mean, style_std, _ = calc_weighted_mean_std(style_feat,target_mask)
        else:
            style_mean, style_std = calc_mean_std(style_feat)

        total_style_mean += style_mean*input_mask.to(style_mean.device)
        total_style_var += (style_std**2)*input_mask.to(style_mean.device)

    total_style_std = torch.sqrt(total_style_var)
    adaIN_feat = total_normalized_feat * total_style_std + total_style_mean
    return adaIN_feat




def adaptive_instance_normalization_saved_stats(content_feat, content_sem, style_means, style_stds):
    size = content_feat.size()
    
    total_style_mean = torch.zeros(size).to(content_feat.device)
    total_style_var = torch.zeros(size).to(content_feat.device)
    total_normalized_feat = torch.zeros(size).to(content_feat.device)

    with open(style_means, "rb") as myFile:
        means = pickle.load(myFile)
    with open(style_stds, "rb") as myFile:
        stds = pickle.load(myFile)

    content_mean, content_std = calc_mean_std(content_feat)
    total_normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    for class_id in torch.unique(content_sem):

        class_id_float = class_id.item()

        try:
            style_std = torch.from_numpy(stds[class_id_float]).to(content_feat.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        except KeyError:
            style_std = torch.zeros(size)
        try:
            style_mean = torch.from_numpy(means[class_id_float]).to(content_feat.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        except KeyError:
            style_mean = torch.zeros(size)

        input_mask = F.interpolate((content_sem == class_id).float(), size=size[2:], mode='bilinear')

        style_mean = style_mean.to(input_mask.device)
        style_std = style_std.to(input_mask.device)
        total_style_mean += style_mean*input_mask
        total_style_var += (style_std**2)*input_mask

    total_style_std = torch.sqrt(total_style_var)
    adaIN_feat = total_normalized_feat * total_style_std + total_style_mean
    return adaIN_feat
    
