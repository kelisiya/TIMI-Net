from .resnet_dec import BasicBlock
from .res_timi_decoder import Res_TIMI_de


__all__ = ['resnet_timi_decoder']


def _res_timi_decoder(block, layers, **kwargs):
    model = Res_TIMI_de(block, layers, **kwargs)
    return model


def resnet_timi_decoder(**kwargs):
    return _res_timi_decoder(BasicBlock, [2, 3, 3, 2], **kwargs)
