# How w2v-en-int8.onnx and w2v-ja-int8.onnx were made (Python 3.12, torch 2.14, transformers 5.23, onnxruntime 1.32).
# Put the original Hugging Face models under models/<org>/<name>/ next to this script, then run it.

# Export the two wav2vec2 CTC models to ONNX (float32) and an int8 copy (dynamic quantisation of the matmuls).
# Input: audio (1, samples) float32 at 16 kHz, already zero-mean unit-variance. Output: logits (1, frames, vocab).
import os, json, shutil, torch
from transformers import Wav2Vec2ForCTC
from onnxruntime.quantization import quantize_dynamic, QuantType
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'onnx')
os.makedirs(OUT, exist_ok=True)
for short, name in [('ja', 'jonatasgrosman/wav2vec2-large-xlsr-53-japanese'), ('en', 'facebook/wav2vec2-large-960h-lv60-self')]:
    src = os.path.join(HERE, 'models', name)
    m = Wav2Vec2ForCTC.from_pretrained(src).eval()
    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, audio): return s.m(audio).logits
    f32 = os.path.join(OUT, f'w2v-{short}.onnx')
    x = torch.randn(1, 16000 * 4)
    torch.onnx.export(Wrap(m), (x,), f32, input_names=['audio'], output_names=['logits'],
                      dynamic_axes={'audio': {1: 'samples'}, 'logits': {1: 'frames'}}, opset_version=17, dynamo=False)
    q8 = os.path.join(OUT, f'w2v-{short}-int8.onnx')
    quantize_dynamic(f32, q8, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
    vocab = json.load(open(os.path.join(src, 'vocab.json'), encoding='utf-8'))
    json.dump(vocab, open(os.path.join(OUT, f'w2v-{short}-vocab.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print(short, 'f32', os.path.getsize(f32) >> 20, 'MB', 'int8', os.path.getsize(q8) >> 20, 'MB')
