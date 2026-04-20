from models.arch.operations import Ops
from models.arch.hyperparameters import get_cityscapes_hyperparams_small, get_cityscapes_hyperparams_large
from models.arch.model_cityscapes import SqueezeNASNetCityscapes


def map_subnet_to_genotype(subnet):
    genotype = []
    for choice, group, expand_ratio in zip(subnet['choice'], subnet['groups'], subnet['expand_ratio']):
        if choice == 0:
            # 3x3 conv
            if group == 1 and expand_ratio == 1:
                op = Ops.inverse_residual_k3_e1_g1
            elif group == 1 and expand_ratio == 3:
                op = Ops.inverse_residual_k3_e3_g1
            elif group == 1 and expand_ratio == 6:
                op = Ops.inverse_residual_k3_e6_g1
            elif group == 2:
                op = Ops.inverse_residual_k3_e1_g2
        elif choice == 1:
            # 5x5 conv
            if group == 1 and expand_ratio == 1:
                op = Ops.inverse_residual_k5_e1_g1
            elif group == 1 and expand_ratio == 3:
                op = Ops.inverse_residual_k5_e3_g1
            elif group == 1 and expand_ratio == 6:
                op = Ops.inverse_residual_k5_e6_g1
            elif group == 2:
                op = Ops.inverse_residual_k5_e1_g2
        elif choice == 2:
            # 3x3 dilation conv
            if group == 1 and expand_ratio == 1:
                op = Ops.inverse_residual_k3_e1_g1_d2
            elif group == 1 and expand_ratio == 3:
                op = Ops.inverse_residual_k3_e3_g1_d2
            elif group == 1 and expand_ratio == 6:
                op = Ops.inverse_residual_k3_e6_g1_d2
            elif group == 2:
                op = Ops.inverse_residual_k3_e1_g2_d2
        elif choice == 3:
            # skip op
            op = Ops.residual_skipish
        else:
            raise ValueError(f"Invalid choice {choice}")
        genotype.append(op)
    return genotype


def get_custom_model(subnet, num_classes, size):
    genotype = map_subnet_to_genotype(subnet)
    if size == 'small':
        hyperparameters = get_cityscapes_hyperparams_small(num_classes=num_classes)
    elif size == 'large':
        hyperparameters = get_cityscapes_hyperparams_large(num_classes=num_classes)
    model = SqueezeNASNetCityscapes(hyperparameters, genotype, lr_aspp=True)
    return model

