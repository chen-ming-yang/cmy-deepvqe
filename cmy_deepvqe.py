import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class FE(nn.Module):
    """Feature extraction"""
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c
    def forward(self, x):
        """x: (B,F,T,2)"""
        x_mag = torch.sqrt(x[...,[0]]**2 + x[...,[1]]**2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1-self.c) + 1e-12)
        return x_c.permute(0,3,2,1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4,3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
    def forward(self, x):
        """x: (B,C,T,F)"""
        y = self.elu(self.bn(self.conv(self.pad(x))))
        return y + x
    
        
class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, delay=100):
        super().__init__()
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.unfold = nn.Sequential(nn.ZeroPad2d([0,0,delay-1,0]),
                                    nn.Unfold((delay, 1)))
        self.conv = nn.Sequential(nn.ZeroPad2d([1,1,4,0]),
                                  nn.Conv2d(hidden_channels, 1, (5,3)))
        
        
    def forward(self, x_mic, x_ref):
        """
        x_mic: (B,C,T,F)
        x_ref: (B,C,T,F)
        """
        Q = self.pconv_mic(x_mic)  # (B,H,T,F)
        K = self.pconv_ref(x_ref)  # (B,H,T,F)
        Ku = self.unfold(K)        # (B, H*D, T*F)
        Ku = Ku.view(K.shape[0], K.shape[1], -1, K.shape[2], K.shape[3])\
            .permute(0,1,3,2,4).contiguous()  # (B,H,T,D,F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)      # (B,H,T,D)
        V = self.conv(V)           # (B,1,T,D)
        A = torch.softmax(V, dim=-1)[..., None]  # (B,1,T,D,1)
        
        y = self.unfold(x_ref).view(x_ref.shape[0], x_ref.shape[1], -1, x_ref.shape[2], x_ref.shape[3])\
                .permute(0,1,3,2,4).contiguous()  # (B,C,T,D,F)
        y = torch.sum(y * A, dim=-2)  # (B,C,T,F)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), stride=(1,2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)
    def forward(self, x):
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """x : (B,C,T,F)"""
        y = rearrange(x, 'b c t f -> b t (c f)')
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, 'b t (c f) -> b c t f', c=x.shape[1])
        return y
    

class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1,1,3,0])
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size)
        
    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, 'b (r c) t f -> b c t (r f)', r=2)
        return y
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.is_last = is_last
    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
    

class CCM(nn.Module):
    """Complex convolving mask block"""
    def __init__(self):
        super().__init__()
        self.v = torch.tensor([[1,        -1/2,           -1/2],
                               [0, np.sqrt(3)/2, -np.sqrt(3)/2]], dtype=torch.float32)  # (2,3)
        self.unfold = nn.Sequential(nn.ZeroPad2d([1,1,2,0]),
                                    nn.Unfold(kernel_size=(3,3)))
    
    def forward(self, m, x):
        """
        m: (B,27,T,F)
        x: (B,F,T,2)"""
        m = rearrange(m, 'b (r c) t f -> b r c t f', r=3)
        H_real = torch.sum(self.v[0].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)
        H_imag = torch.sum(self.v[1].to(m.device)[None,:,None,None,None] * m, dim=1)  # (B,C/3,T,F)

        M_real = rearrange(H_real, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        M_imag = rearrange(H_imag, 'b (m n) t f -> b m n t f', m=3)  # (B,3,3,T,F)
        
        x = x.permute(0,3,2,1).contiguous()  # (B,2,T,F)
        x_unfold = self.unfold(x)
        x_unfold = rearrange(x_unfold, 'b (c m n) (t f) -> b c m n t f', m=3,n=3,f=x.shape[-1])

        x_enh_real = torch.sum(M_real * x_unfold[:,0] - M_imag * x_unfold[:,1], dim=(1,2))  # (B,T,F)
        x_enh_imag = torch.sum(M_real * x_unfold[:,1] + M_imag * x_unfold[:,0], dim=(1,2))  # (B,T,F)
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1,2).contiguous()
        return x_enh


class DeepVQE(nn.Module):
    def __init__(
            self,
            mic_channels=None,
            ref_channels=None,
            dec_channels=None,
            gru_hidden=256,
            align_hidden=16,
            dmax=100,
            fe_compress=0.3,
            n_fft=512,
    ):
        super().__init__()

        if mic_channels is None:
            mic_channels = [64, 128, 128, 128, 128]
        if ref_channels is None:
            ref_channels = [32, 128]
        if dec_channels is None:
            dec_channels = [128, 128, 128, 64, 27]
        
        self.mic_channels = mic_channels
        self.ref_channels = ref_channels
        self.dec_channels = dec_channels

        # Compute the frequency dimension reaching the bottleneck.
        # Each EncoderBlock uses stride=(1,2) on F, giving ceil(F/2) each stage.
        import math
        _f = n_fft // 2 + 1          # e.g. 257 for n_fft=512
        for _ in range(len(mic_channels)):  # 5 encoder stages
            _f = math.ceil(_f / 2)   # 129→65→33→17→9
        gru_input_size = mic_channels[-1] * _f  # e.g. 128*9 = 1152



        self.fe = FE(c=fe_compress)
        self.mic_enc1 = EncoderBlock(2, mic_channels[0])
        self.mic_enc2 = EncoderBlock(mic_channels[0], mic_channels[1])
        self.mic_enc3 = EncoderBlock(
            mic_channels[1] + ref_channels[1], mic_channels[2]
        )
        self.mic_enc4 = EncoderBlock(mic_channels[2], mic_channels[3])
        self.mic_enc5 = EncoderBlock(mic_channels[3], mic_channels[4])

        self.ref_enc1 = EncoderBlock(2, ref_channels[0])
        self.ref_enc2 = EncoderBlock(ref_channels[0], ref_channels[1])

        self.align_block = AlignBlock(
            in_channels=ref_channels[1],
            hidden_channels=align_hidden,
            delay=dmax
        )

        self.bottle = Bottleneck(input_size=gru_input_size, hidden_size=gru_hidden)

        self.dec1 = DecoderBlock(mic_channels[4], dec_channels[0])
        self.dec2 = DecoderBlock(dec_channels[0], dec_channels[1])
        self.dec3 = DecoderBlock(dec_channels[1], dec_channels[2])
        self.dec4 = DecoderBlock(dec_channels[2], dec_channels[3])
        self.dec5 = DecoderBlock(dec_channels[3], dec_channels[4], is_last=True)

        self.skip_adapt2 = (
            nn.Conv2d(mic_channels[3], dec_channels[0], 1) 
            if mic_channels[3] != dec_channels[0] 
            else nn.Identity()
        )

        self.skip_adapt3 = (
            nn.Conv2d(mic_channels[2], dec_channels[1], 1)
            if mic_channels[2] != dec_channels[1]
            else nn.Identity()
        )

        self.skip_adapt4 = (
            nn.Conv2d(mic_channels[1], dec_channels[2], 1)
            if mic_channels[1] != dec_channels[2]
            else nn.Identity()
        )

        self.skip_adapt5 = (
            nn.Conv2d(mic_channels[0], dec_channels[3], 1)
            if mic_channels[0] != dec_channels[3]
            else nn.Identity()
        )

        self.ccm = CCM()
        
    def forward(self, mic_spec, ref_spec):
        """x: (B,F,T,2)"""

        mic_feat = self.fe(mic_spec)
        ref_feat = self.fe(ref_spec)

        mic_e1 = self.mic_enc1(mic_feat)
        mic_e2 = self.mic_enc2(mic_e1)

        ref_e1 = self.ref_enc1(ref_feat)
        ref_e2 = self.ref_enc2(ref_e1)

        ref_align = self.align_block(mic_e2, ref_e2)

        mic_cat = torch.cat([mic_e2, ref_align], dim=1)

        mic_e3 = self.mic_enc3(mic_cat)
        mic_e4 = self.mic_enc4(mic_e3)
        mic_e5 = self.mic_enc5(mic_e4)

        bn_out = self.bottle(mic_e5)

        dec_out = self.dec1(bn_out, mic_e5)
        dec_out = dec_out[..., :mic_e4.shape[-1]]
        dec_out = self.dec2(dec_out, self.skip_adapt2(mic_e4))
        dec_out = dec_out[..., :mic_e3.shape[-1]]
        dec_out = self.dec3(dec_out, self.skip_adapt3(mic_e3))
        dec_out = dec_out[..., :mic_e2.shape[-1]]
        dec_out = self.dec4(dec_out, self.skip_adapt4(mic_e2))
        dec_out = dec_out[..., :mic_e1.shape[-1]]
        dec_out = self.dec5(dec_out, self.skip_adapt5(mic_e1))
        dec_out = dec_out[..., :mic_spec.shape[1]]  # trim to original F

        clean_spec = self.ccm(dec_out, mic_spec)
        
        return clean_spec



if __name__ == "__main__":
    model = DeepVQE().eval()

    # Register hooks to print intermediate shapes
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                print(f"{name}: {output.shape}")
        return hook

    hooks = []
    for name, module in model.named_children():
        hooks.append(module.register_forward_hook(make_hook(name)))

    mic = torch.randn(1, 257, 63, 2)
    ref = torch.randn(1, 257, 63, 2)
    print(f"mic input: {mic.shape}")
    print(f"ref input: {ref.shape}")
    y = model(mic, ref)
    print(f"output: {y.shape}")

    # Remove hooks
    for h in hooks:
        h.remove()

    
    from ptflops import get_model_complexity_info
    def input_constructor(input_res):
        return {"mic_spec": torch.randn(1, *input_res), "ref_spec": torch.randn(1, *input_res)}
    flops, params = get_model_complexity_info(model, (257, 63, 2), input_constructor=input_constructor,
                                           as_strings=True, print_per_layer_stat=False, verbose=True)
    print(flops, params)




        