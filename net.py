import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import scipy.sparse

from function import calc_mean_std


class SegmentNormalizeLayer(nn.Module):
    def __init__(self, num_features):
        super(SegmentNormalizeLayer, self).__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features)

    def forward(self, x, seg_map):
        # Ensure seg_map is the same spatial size as x
        seg_map = F.interpolate(seg_map.float(), size=x.shape[2:], mode='nearest')
        
        output = torch.zeros_like(x)
        unique_classes = torch.unique(seg_map)
        
        for class_id in unique_classes:
            # Create a mask for this class
            mask = (seg_map == class_id).float()
            
            # Apply the mask to the features
            class_features = x * mask
            
            # Normalize the features for this class
            normalized_features = self.norm(class_features)
            
            # Add to the output
            output += normalized_features

        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Layer 1
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(512, 256, (3, 3))
        self.relu1 = nn.ReLU()
        self.norm1 = SegmentNormalizeLayer(256)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Layer 2
        self.pad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(256, 256, (3, 3))
        self.relu2 = nn.ReLU()
        
        # Layer 3
        self.pad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(256, 256, (3, 3))
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.pad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(256, 256, (3, 3))
        self.relu4 = nn.ReLU()
        
        # Layer 5
        self.pad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(256, 128, (3, 3))
        self.relu5 = nn.ReLU()
        self.norm5 = SegmentNormalizeLayer(128)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Layer 6
        self.pad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 128, (3, 3))
        self.relu6 = nn.ReLU()
        
        # Layer 7
        self.pad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(128, 64, (3, 3))
        self.relu7 = nn.ReLU()
        self.norm7 = SegmentNormalizeLayer(64)
        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Layer 8
        self.pad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(64, 64, (3, 3))
        self.relu8 = nn.ReLU()
        
        # Final Layer
        self.pad_final = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv_final = nn.Conv2d(64, 3, (3, 3))

    def forward(self, x, seg_map):
        # Layer 1
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x, seg_map)
        x = self.up1(x)
        
        # Layer 2
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        
        # Layer 3
        x = self.pad3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Layer 4
        x = self.pad4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Layer 5
        x = self.pad5(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.norm5(x, seg_map)
        x = self.up5(x)
        
        # Layer 6
        x = self.pad6(x)
        x = self.conv6(x)
        x = self.relu6(x)
        
        # Layer 7
        x = self.pad7(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.norm7(x, seg_map)
        x = self.up7(x)
        
        # Layer 8
        x = self.pad8(x)
        x = self.conv8(x)
        x = self.relu8(x)
        
        # Final Layer
        x = self.pad_final(x)
        x = self.conv_final(x)
        
        return x

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, adain, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.adain = adain

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss_old(self, input, target, input_sem, target_sem):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        
        total_loss = 0 
        for class_id in torch.unique(input_sem):
            input_mask = F.interpolate((input_sem == class_id).float(), size = input.shape[2:], mode = 'bilinear')
            target_mask = F.interpolate((target_sem == class_id).float(), size = target.shape[2:], mode = 'bilinear')

            masked_input = input*input_mask 
            masked_target = target*target_mask 

            input_mean, input_std = calc_mean_std(masked_input)
            target_mean, target_std = calc_mean_std(masked_target)

            total_loss += self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

        return total_loss
    
    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)


    def forward(self, content, style, content_sem, style_sem, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.adain(content_feat, style_feats[-1], content_sem, style_sem)
        t = alpha * t + (1 - alpha) * content_feat
        
        g_t = t

        # Modify this part
        for layer in self.decoder:
            if isinstance(layer, SegmentNormalizeLayer):
                g_t = layer(g_t, content_sem)
            else:
                g_t = layer(g_t)

        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = 0
        for i in range(4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        
        return loss_c, loss_s