from dataclasses import dataclass
from typing import Sequence, Optional

from torch import nn

from models.arch.operations import Ops
from models.arch.utils import Flatten, conv3x3_bn, conv_1x1_bn


@dataclass(frozen=True)
class InvertResidualNetBlockMetaHyperparameters:
    num_channels: int
    num_repeat: int
    stride: int


@dataclass(frozen=True)
class InverseResidualMetaNetHyperparameters:
    init_channels: int
    blocks: Sequence[InvertResidualNetBlockMetaHyperparameters]
    last_channels: Optional[int]
    num_classes: Optional[int]
    last_pooled_channels: Optional[int] = None

    def __post_init__(self):
        if self.last_channels is None:
            assert self.num_classes is None


# InvertedResidual(genotype[gene_i], last_c, block_hyper.num_channels, stride)

class InvertedResidual(nn.Module):
    def __init__(self, op_type: Ops, c_in: int, c_out: int, stride: int):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # Use res_connect when stride=1 and c_in = c_out at the same time
        self.use_res_connect = self.stride == 1 and c_in == c_out
        self.op: nn.Module = op_type.value.__call__(c_in, c_out, stride, affine=True)
        self.op_type = op_type

    def forward(self, x):
        output = self.op(x)
        if self.use_res_connect:
            return x + output
        else:
            return output


class SqueezeNASNet(nn.Module):
    def __init__(self, hyperparams: InverseResidualMetaNetHyperparameters, genotype: Sequence[Ops], dropout=0):
        super().__init__()
        self.hyperparams = hyperparams
        self.conv1 = conv3x3_bn(c_in=3, c_out=hyperparams.init_channels, stride=2)
        self.residuals: nn.ModuleList[InvertedResidual] = nn.ModuleList()
        self.genotype = genotype

        gene_i = 0
        last_c = hyperparams.init_channels

        # TODO, Print the network architecture one by one, then bulid the supernet
        for block_hyper in hyperparams.blocks:
            for i in range(block_hyper.num_repeat):
                # The step of the first operation can be 2. Which means the downsample operation is operated at first
                if i == 0:
                    stride = block_hyper.stride
                else:
                    stride = 1
                self.residuals.append(InvertedResidual(genotype[gene_i], last_c, block_hyper.num_channels, stride))
                gene_i += 1
                last_c = block_hyper.num_channels

        assert gene_i == len(genotype)

        if hyperparams.last_channels is not None:
            self.last_conv = conv_1x1_bn(last_c, hyperparams.last_channels)
            if hyperparams.num_classes is not None:
                # building classifier
                if hyperparams.last_pooled_channels is None:
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        Flatten(),
                        nn.Dropout(dropout),
                        nn.Linear(hyperparams.last_channels, hyperparams.num_classes),
                    )
                else:
                    self.classifier = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        Flatten(),
                        nn.Linear(hyperparams.last_channels, hyperparams.last_pooled_channels),
                        nn.Dropout(dropout),
                        nn.Linear(hyperparams.last_pooled_channels, hyperparams.num_classes),
                    )
                # self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, gt=None):
        cur_feat = inputs
        cur_feat = self.conv1(cur_feat)
        # print('cur_feat', cur_feat.shape)
        residuals_outputs = []
        # the list "residuals_outputs" is used for skip_output_block_index
        for i, residual in enumerate(self.residuals):
            cur_feat = residual(cur_feat)
            residuals_outputs.append(cur_feat)
            # print(cur_feat.shape)

        if self.hyperparams.last_channels is None:
            return {'output': cur_feat, 'residuals_outputs': residuals_outputs}

        cur_feat = self.last_conv(cur_feat)
        
        if self.hyperparams.num_classes is None:
            # print('yes')
            return {'output': cur_feat, 'residuals_outputs': residuals_outputs}
        
        logits = self.classifier(cur_feat)

        # if gt is None:
        #     return {'preds': logits}
        # else:
        #     loss = self.criterion(logits.float(), gt)
        #     return {'loss': loss, 'preds': logits}
        return logits
