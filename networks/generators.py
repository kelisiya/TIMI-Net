import torch
import torch.nn as nn

from   networks import encoders, decoders


class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha  = self.decoder(embedding, mid_fea)
        return alpha


def get_generator(encoder='resnet_timi_encoder', decoder='resnet_timi_decoder'):
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator


