import torch
import torch.nn as nn
import numpy as np
from models.module.seg_oprs import ConvNorm
from models.module.fasterseg_ops_sub import ConvNorm, BasicResidual2x, BasicBlocks_Sub, Head, FeatureFusion
from torch.nn import functional as F
import copy


class FasterSeg_sub(nn.Module):
    def __init__(self, 
                 num_classes=19, 
                 layers=10, 
                 Fch=12, 
                 stem_head_width=(8./12, 8./12),
                 downs1_list = None,
                 downs2_list = None,
                 width_list= None, 
                 ops_list = None
                 ):
        
        super().__init__()

        self._stem_head_width = stem_head_width
        self._Fch = Fch
        self._layers = layers
        self._num_classes = num_classes

        # Fixed Stem
        self.stem = nn.Sequential(
            ConvNorm(3, self.num_filters(2, stem_head_width[0])*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            BasicResidual2x(self.num_filters(2, stem_head_width[0])*2, self.num_filters(4, stem_head_width[0])*2, kernel_size=3, stride=2, groups=1),
            BasicResidual2x(self.num_filters(4, stem_head_width[0])*2, self.num_filters(8, stem_head_width[0]), kernel_size=3, stride=2, groups=1)
        )
        
        shared_flag = True
        size1 = size2 = 8
        multiplier1 = multiplier2 = self._stem_head_width[0]
        self.shared_ops = nn.ModuleList()
        self.branch1_ops = nn.ModuleList()
        self.branch2_ops = nn.ModuleList()

        for idx, (down1, down2) in enumerate(zip(downs1_list, downs2_list)):
            if idx > 0:
                if shared_flag:
                    multiplier1 = multiplier2 = width_list[idx-1]
                else:
                    multiplier1 = width_list[idx-1][0]
                    multiplier2 = width_list[idx-1][1]
            if down1 == 1:
                size1 *= 2
            if down2 == 1:
                size2 *= 2
            shared_flag = shared_flag and (down1 == 0 and down2 == 0)
            if shared_flag:
                block = BasicBlocks_Sub(C_in=self.num_filters(size1, multiplier1),
                                         C_out=self.num_filters(size1, width_list[idx]),
                                         ops=ops_list[idx],
                                         stride=1)
                self.shared_ops.append(block)
            else:
                block1 = BasicBlocks_Sub(C_in=self.num_filters(size1 // (2 if down1 else 1), multiplier1),
                                        C_out=self.num_filters(size1, width_list[idx][0]),
                                        ops=ops_list[idx][0],
                                        stride= 2 if down1 else 1)
                self.branch1_ops.append(block1)

                block2 = BasicBlocks_Sub(C_in=self.num_filters(size2 // (2 if down2 else 1), multiplier2),
                                        C_out=self.num_filters(size2, width_list[idx][1]),
                                        ops=ops_list[idx][1],
                                        stride= 2 if down2 else 1)
                self.branch2_ops.append(block2)

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def build_arm_ffm_head(self):
        self.ch_8_1 = 32
        self.ch_8_2 = 32
        self.ch_16 = 64
        # Build SegHead
        if self.training:
            self.heads32 = Head(self.num_filters(32, self._stem_head_width[1]), self._num_classes, True)
            self.heads16 = Head(self.num_filters(16, self._stem_head_width[1])+self.ch_16, self._num_classes, True)


        self.heads8 = Head(self.num_filters(8, self._stem_head_width[1]) * 2, self._num_classes, Fch=self._Fch, scale=4, is_aux=False)

        
        self.arms32 = nn.ModuleList([
                ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]), 1, 1, 0, ),
                ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, ),
            ])
        
        self.refines32 = nn.ModuleList([
            ConvNorm(self.num_filters(16, self._stem_head_width[1])+self.ch_16, self.num_filters(16, self._stem_head_width[1]), 3, 1, 1, ), # 
            ConvNorm(self.num_filters(8, self._stem_head_width[1])+self.ch_8_2, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, ),
        ])
        
        self.arms16 = ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, )
        self.refines16 = ConvNorm(self.num_filters(8, self._stem_head_width[1])+self.ch_8_1, self.num_filters(8, self._stem_head_width[1]), 3, 1, 1, )
        self.ffm = FeatureFusion(self.num_filters(8, self._stem_head_width[1]) * 2, self.num_filters(8, self._stem_head_width[1]) * 2, reduction=1, Fch=self._Fch, scale=8)

   
    def agg_ffm(self, outputs8, outputs16_b1, outputs16_b2, outputs32):
        pred32 = []; pred16 = []; pred8 = []
        pred32.append(outputs32)
        pred16.append(outputs16_b2)
        pred16.append(outputs16_b1)
        out = self.arms32[0](outputs32) # 256 => 128
        out = F.interpolate(out, size=(outputs16_b2.size(2), outputs16_b2.size(3)), mode='bilinear', align_corners=True)
        
        # print(out.shape, outputs16_b2.shape)
        out = self.refines32[0](torch.cat([out, outputs16_b2], dim=1))
        out = self.arms32[1](out)
        out = F.interpolate(out, size=(outputs8.size(2), outputs8.size(3)), mode='bilinear', align_corners=True)
        out = self.refines32[1](torch.cat([out, outputs8], dim=1))
        pred8.append(out)
    
        out_2 = self.arms16(outputs16_b1)
        out_2 = F.interpolate(out_2, size=(outputs8.size(2), outputs8.size(3)), mode='bilinear', align_corners=True)
        out_2 = self.refines16(torch.cat([out_2, outputs8], dim=1))
        pred8.append(out_2)

        pred32 = self.heads32(torch.cat(pred32, dim=1))
        # print(pred16[0].shape, pred16[1].shape)
        pred16 = self.heads16(torch.cat(pred16, dim=1))
        pred8 = self.heads8(self.ffm(torch.cat(pred8, dim=1)))

        if self.training:
            return pred8, pred16, pred32
        else:
            return pred8
    
                
    def forward(self, input):
        _, _, H, W = input.size()
        stem = self.stem(input) # torch.Size([1, 64, 128, 256])
        shared_feat = stem
        for op in self.shared_ops:
            shared_feat = op(shared_feat)
        b1_feat = b2_feat = shared_feat

        for ops_b1 in self.branch1_ops:
            if ops_b1.stride == 2:
                os8_feat = b1_feat
            b1_feat = ops_b1(b1_feat)
        os16_feat_b1 = b1_feat

        for ops_b2 in self.branch2_ops:
            if ops_b2.stride == 2:
                os16_feat_b2 = b2_feat
            b2_feat = ops_b2(b2_feat)
        os32_feat = b2_feat

        # print(os8_feat.shape, os16_feat_b1.shape, os16_feat_b2.shape, os32_feat.shape)
        if self.training:
            pred8, pred16, pred32 = self.agg_ffm(os8_feat, os16_feat_b1, os16_feat_b2, os32_feat)
            pred8 = F.interpolate(pred8, scale_factor=8, mode='bilinear', align_corners=True)
            if pred16 is not None: pred16 = F.interpolate(pred16, scale_factor=16, mode='bilinear', align_corners=True)
            if pred32 is not None: pred32 = F.interpolate(pred32, scale_factor=32, mode='bilinear', align_corners=True)

            # return pred8, pred16, pred32
            return pred32, pred16, pred8
        else:
            pred8 = self.agg_ffm(os8_feat, os16_feat_b1, os16_feat_b2, os32_feat)
            out = F.interpolate(pred8, size=(int(pred8.size(2))*8, int(pred8.size(3))*8), mode='bilinear', align_corners=True)
            return out
        



