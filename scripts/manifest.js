// Writes manifest.json for a release from the files in release/: name, size and SHA-256 of each, grouped in packs.
// node scripts/manifest.js cadence-1
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const tag = process.argv[2] || 'cadence-1';
const dir = path.join(__dirname, '..', 'release');
const OWNER = 'MinhHaiDeveloper', REPO = 'telecloud-models';
const packs = {
  en: { title: 'English and other languages', required: true, files: ['w2v-en-int8.onnx', 'w2v-en-vocab.json', 'wav2vec2-en.Apache-2.0.txt'] },
  ja: { title: 'Japanese', required: false, files: ['w2v-ja-int8.onnx', 'w2v-ja-vocab.json', 'wav2vec2-ja.Apache-2.0.txt'] },
  voice: { title: 'Voice separation', required: false, files: ['htdemucs-vocals.onnx', 'demucs.MIT.txt'] },
};
const sha = (f) => crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex');
const out = {
  name: 'cadence',
  version: Number(tag.split('-')[1]),
  tag,
  base: `https://github.com/${OWNER}/${REPO}/releases/download/${tag}/`,
  packs: Object.fromEntries(Object.entries(packs).map(([id, p]) => [id, {
    title: p.title,
    required: p.required,
    files: p.files.map(f => ({ name: f, size: fs.statSync(path.join(dir, f)).size, sha256: sha(path.join(dir, f)) })),
  }])),
};
for (const p of Object.values(out.packs)) p.size = p.files.reduce((a, f) => a + f.size, 0);
const json = JSON.stringify(out, null, 2) + '\n';
fs.writeFileSync(path.join(dir, 'manifest.json'), json);
fs.writeFileSync(path.join(__dirname, '..', 'manifest.json'), json);
for (const [id, p] of Object.entries(out.packs)) console.log(id.padEnd(6), (p.size / 1048576).toFixed(0).padStart(4), 'MB', p.title);
