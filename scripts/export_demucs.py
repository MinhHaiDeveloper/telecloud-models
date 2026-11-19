# How htdemucs-vocals.onnx was made (demucs 4.1.0, torch 2.14). Also checks the ONNX file against the original model.

# Export htdemucs (MIT) to ONNX without its STFT: the app does the STFT / inverse STFT itself (JS), the network
# runs in ONNX Runtime. One fixed 7.8 s segment (343980 samples at 44.1 kHz), vocals only.
#   inputs : mix (1, 2, 343980) normalised waveform, spec (1, 4, 2048, 336) = STFT as complex-as-channels
#   outputs: spec_out (1, 4, 2048, 336) vocals spectrogram (cac), wave_out (1, 2, 343980) vocals time branch
import os, math, torch
from demucs.pretrained import get_model
HERE = os.path.dirname(os.path.abspath(__file__))
bag = get_model('htdemucs')
m = bag.models[0].eval()
V = m.sources.index('vocals')
SEG = int(m.segment * m.samplerate)
print('sources', m.sources, 'segment', SEG, 'nfft', m.nfft, 'hop', m.hop_length, 'cac', m.cac, 'bag weights', bag.weights)

class Core(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, mix, mag):
        self = s.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = self.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt)
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = self.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = self.channel_downsampler_t(xt)
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        return x[:, V], xt[:, V]

core = Core(m).eval()
mix = torch.randn(1, 2, SEG) * 0.3
with torch.no_grad():
    z = m._spec(mix)
    mag = m._magnitude(z)
    ref = m(mix)[:, V]
    so, wo = core(mix, mag)
    out = m._ispec(m._mask(z, so[:, None]), SEG)[:, 0] + wo
print('core vs model, max abs diff', (out - ref).abs().max().item(), 'mag shape', tuple(mag.shape))
p = os.path.join(HERE, 'onnx', 'htdemucs-vocals.onnx')
torch.onnx.export(core, (mix, mag), p, input_names=['mix', 'spec'], output_names=['spec_out', 'wave_out'], opset_version=17, dynamo=False)
print('saved', p, os.path.getsize(p) >> 20, 'MB')
import onnxruntime as ort, numpy as np
s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
a, b = s.run(None, {'mix': mix.numpy(), 'spec': mag.numpy()})
print('onnx vs torch core: spec', np.abs(a - so.numpy()).max(), 'wave', np.abs(b - wo.numpy()).max())
