import torch
import torch.nn as nn
import os
from collections import OrderedDict

from FWFormer import FWFormer
from RetCABFormer import RetCABFormer
from homoregionformer import HomoRegionFormer


def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from Shadowformer import ShadowFormer
    from homoregionformer import HomoRegionFormer
    from HistoSRFormer import HistoSRFormer
    from QAShadowNet import QAShadowFormer
    from BiSRFormer import BiSRFormer
    from RetSRFormer import RetSRFormer
    from RetSRMaskFormer import RetSRMaskFormer
    from RetMaskDepthFormer import RetMaskDethFormer
    from RetCABFormer import RetCABFormer
    from Ret2SRFormer import Ret2SRFormer
    from RetBoundaryFormer import RetBoundaryFormer
    from RetBFormer import RetBFormer
    from Ret3SRFormer import Ret3SRFormer
    from RetSARFormer import RetSARFormer
    from RetSRTFormer import RetSRTFormer
    from RetWinFormer import RetWinFormer
    from RetD2DFormer import RetD2DFormer
    from RetNoposFormer import RetNoPosFormer
    from RetMixGasFormer import RetMixGasFormer
    from RetPbPsFormer import RetPbPsFormer
    from RetGausFormer import RetGausFormer
    from Ret2DSRFormer import Ret2DSRFormer
    from FWFormer import FWFormer
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'ShadowFormer':
        model_restoration = ShadowFormer()
    elif arch == 'HomoFormer':
        model_restoration = HomoRegionFormer()
    elif arch == 'HistoSRFormer':
        model_restoration = HistoSRFormer()
    elif arch == 'QAShadowNet':
        model_restoration = QAShadowFormer()
    elif arch == 'BiSRFormer':
        model_restoration = BiSRFormer()
    elif arch == 'RetSRFormer':
        model_restoration = RetSRFormer()
    elif arch == 'RetSRMaskFormer':
        model_restoration = RetSRMaskFormer()
    elif arch == 'RetMaskDepthFormer':
        model_restoration = RetMaskDethFormer()
    elif arch == 'RetCABFormer':
        model_restoration = RetCABFormer()
    elif arch == 'Ret2SRFormer':
        model_restoration = Ret2SRFormer()
    elif arch == 'RetBoundaryFormer':
        model_restoration = RetBoundaryFormer()
    elif arch == 'RetBFormer':
        model_restoration = RetBFormer()
    elif arch == 'Ret3SRFormer':
        model_restoration = Ret3SRFormer()
    elif arch == 'RetSARFormer':
        model_restoration = RetSARFormer()
    elif arch == 'RetSRTFormer':
        model_restoration = RetSRTFormer()
    elif arch == 'RetWinFormer':
        model_restoration = RetWinFormer()
    elif arch == 'RetD2DFormer':
        model_restoration = RetD2DFormer()
    elif arch == 'RetNoPosFormer':
        model_restoration = RetNoPosFormer()
    elif arch == 'RetMixGasFormer':
        model_restoration = RetMixGasFormer()
    elif arch == 'RetPbPsFormer':
        model_restoration = RetPbPsFormer()
    elif arch == 'RetGausFormer':
        model_restoration = RetGausFormer()
    elif arch == 'Ret2DSRFormer':
        model_restoration = Ret2DSRFormer()
    elif arch == 'FWFormer':
        model_restoration = FWFormer()
    else:
        raise Exception("Arch error!")

    return model_restoration
