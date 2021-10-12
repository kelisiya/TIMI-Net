from .resnet_enc import BasicBlock
from .res_timi_encoder import Res_TIMI


__all__ = ['resnet_timi_encoder']


def _res_timi_encoder(block, layers, **kwargs):
    model = Res_TIMI(block, layers, **kwargs)
    return model


def resnet_timi_encoder(**kwargs):
    return _res_timi_encoder(BasicBlock, [3, 4, 4, 2], **kwargs)


