import torch.nn as nn
import torch
import math


class LFDA(nn.Module):
    def __init__(self, image_size=256, upbound=3+1, lowbound=1, fda_beta=None):
        super(LFDA, self).__init__()
        self.image_size = image_size
        self.upbound = upbound
        self.lowbound = lowbound
        self.activate = torch.nn.Sigmoid()
        if fda_beta is not None:
            self.fda_beta = torch.nn.Parameter(torch.FloatTensor(fda_beta))
        else:
            self.fda_beta = torch.nn.Parameter(torch.FloatTensor([math.log(1./9.)] * (upbound - lowbound)))
        assert self.fda_beta.shape[0] == upbound-lowbound

    def forward(self, input_S, input_T):
        zeros = torch.zeros(self.image_size//2-self.upbound).to(self.fda_beta.device)
        if self.lowbound > 0:
            ones = torch.ones(self.lowbound).to(self.fda_beta.device)
            vector = torch.cat((self.activate(self.fda_beta), ones), dim=0)
        else:
            vector = self.activate(self.fda_beta)
        vector = torch.cat((zeros, vector), dim=0)
        weight_1D = torch.cat((vector, torch.flip(vector, dims=[0])), dim=0).unsqueeze(0)
        weight_2D = torch.matmul(weight_1D.t(), weight_1D)
        weight_4D = weight_2D.unsqueeze(0).unsqueeze(0)
        CT_SS = self.FDA_torch(src_img=input_T, trg_img=input_S, mask=weight_4D.to(input_S.device))
        CS_ST = self.FDA_torch(src_img=input_S, trg_img=input_T, mask=weight_4D.to(input_S.device))
        return CT_SS, CS_ST

    def extract_ampl_phase(self, fft_im):
        # fft_im: size should be b x 3 x h x w
        fft_amp = torch.abs(fft_im)
        fft_pha = torch.angle(fft_im)
        return fft_amp, fft_pha

    def FDA_torch(self, src_img, trg_img, mask):
        # get fft of both source and target
        fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))
        fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))

        # extract amplitude and phase of both ffts
        amp_src, pha_src = self.extract_ampl_phase(fft_src)
        amp_trg, pha_trg = self.extract_ampl_phase(fft_trg)

        amp_src = torch.fft.fftshift(amp_src)
        amp_trg = torch.fft.fftshift(amp_trg)

        # replace the low frequency amplitude part of source with that from target
        amp_src_ = mask * amp_trg + (1. - mask) * amp_src
        amp_src_ = torch.fft.ifftshift(amp_src_)

        # recompose fft of source
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        # get the recomposed image: source content, target style
        _, _, imgH, imgW = src_img.size()
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg
