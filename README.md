# TeleCloud models

Model files that the [TeleCloud](https://github.com/MinhHaiDeveloper/telecloud) desktop app downloads when you need
them. Nothing here is in the app's installer. You download each pack from inside the app, and you can delete it and
download it again whenever you like.

## Cadence (word timing)

Cadence works out when each word of a song is sung, for lyrics that only have a time per line. It runs on your
computer. The models live in the releases tagged `cadence-N`.

| Pack | Size | What it is for | Files |
|---|---|---|---|
| English and other languages | 339 MB | Needed. Fits the words of every line that is not Japanese | `w2v-en-int8.onnx`, `w2v-en-vocab.json` |
| Japanese | 341 MB | Japanese lines (kana and kanji as written) | `w2v-ja-int8.onnx`, `w2v-ja-vocab.json` |
| Voice separation | 166 MB | Optional. Takes the music away first, so words land better under loud music | `htdemucs-vocals.onnx` |

`manifest.json` lists every file with its size and SHA-256. The app checks each download against it.

### Where the models come from

| File | Original | Licence | Changed |
|---|---|---|---|
| `w2v-en-int8.onnx` | [facebook/wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self) | Apache-2.0 | Exported to ONNX, int8 weights |
| `w2v-ja-int8.onnx` | [jonatasgrosman/wav2vec2-large-xlsr-53-japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) | Apache-2.0 | Exported to ONNX, int8 weights |
| `htdemucs-vocals.onnx` | [Demucs v4 htdemucs](https://github.com/facebookresearch/demucs) | MIT | The network without its STFT, vocals only, ONNX |

The full licence text for each model is in `licenses/`, and each release carries it next to the model. The int8
copies time words as well as the originals: on 25 test songs the median error was 41 ms against 42 ms.

### Making the files again

`scripts/export_w2v.py` and `scripts/export_demucs.py` are the scripts that made these files. The Demucs export also
compares its output with the original model. `scripts/manifest.js <tag>` writes `manifest.json` from the files in
`release/`.

## Licence

The files written for TeleCloud (this README, `manifest.json` and the scripts) are under the MIT licence in
`LICENSE`. Each model keeps its own licence, listed above and in `licenses/`.
