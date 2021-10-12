import torch
from torch import nn
from torch.nn import Parameter



def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)



class memory_attention(nn.Module):

    def __init__(self, in_dim):
        super(memory_attention, self).__init__()
        self.chanel_in = in_dim
        self.image_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2 , kernel_size=1)
        self.image_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)

        self.trimap_query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.trimap_key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)


        self.softmax_image = nn.Softmax(dim=-1)  #
        self.softmax_trimap = nn.Softmax(dim=-1)  #

        self.gamma_image = nn.Parameter(torch.zeros(1))
        self.gamma_trimap = nn.Parameter(torch.zeros(1))

    def forward(self, mixture,image,trimap):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """


        m_batchsize, C, width, height = mixture.size()
        mixture_value = mixture.view(m_batchsize, -1, width * height)  # B X C X N

        image_query = self.image_query_conv(image).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        image_key = self.image_key_conv(image).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        image_energy = torch.bmm(image_query, image_key)  # transpose check
        image_attention = self.softmax_image(image_energy)  # BX (N) X (N)
        image_out = torch.bmm(mixture_value, image_attention.permute(0, 2, 1))
        image_out = image_out.view(m_batchsize, C, width, height)

        trimap_query = self.image_query_conv(trimap).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        trimap_key = self.image_key_conv(trimap).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        trimap_energy = torch.bmm(trimap_query, trimap_key)  # transpose check
        trimap_attention = self.softmax_trimap(trimap_energy)  # BX (N) X (N)
        trimap_out = torch.bmm(mixture_value, trimap_attention.permute(0, 2, 1))
        trimap_out = trimap_out.view(m_batchsize, C, width, height)


        out = self.gamma_image*image_out + self.gamma_trimap*trimap_out + mixture


        return out




class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)

