# Pretrained Med-VLMs

CalibPrompt evaluates four medical foundation models. The backbone **code** is vendored under `models/`; only the **weight files** need to be downloaded. Place them under a directory and set `MODEL_ROOT` to it (default `./models`).

| Model | Original source | Used with | Local weight path (under `MODEL_ROOT`) |
|:-- |:-- |:-- |:-- |
| MedCLIP | [github.com/RyanWangZf/MedCLIP](https://github.com/RyanWangZf/MedCLIP) | radiology (COVID, RSNA18) | `medclip/pretrained/medclip-vit/pytorch_model.bin` |
| BioMedCLIP | [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | radiology (COVID, RSNA18) | *auto-downloaded from the Hub at runtime* |
| PLIP | [github.com/PathologyFoundation/plip](https://github.com/PathologyFoundation/plip) | histopathology (Kather, PanNuke, DigestPath) | `plip/plip_vit_b32.pt` |
| QuiltNet | [quilt1m.github.io](https://quilt1m.github.io/) ([HF](https://huggingface.co/wisdomik/QuiltNet-B-32)) | histopathology (Kather, PanNuke, DigestPath) | `quiltnet/quiltnet_b32.pt` |
| CLIP (optional) | [OpenAI CLIP](https://github.com/openai/CLIP) | zero-shot CLIP baseline | `clip/openai_clip_vit_b32.pt` |

We use the **same models as [BAPLe](https://github.com/asif-hanif/baple)**; BAPLe also provides mirrored download links for the packaged weights in its [Models section](https://github.com/asif-hanif/baple#models).

## Directory layout

```bash
models/                              # = MODEL_ROOT
    ├── clip/
    │   └── openai_clip_vit_b32.pt           # optional (zero-shot CLIP)
    ├── medclip/
    │   └── pretrained/medclip-vit/
    │       └── pytorch_model.bin
    ├── plip/
    │   └── plip_vit_b32.pt
    └── quiltnet/
        └── quiltnet_b32.pt
```

## Notes
- **BioMedCLIP** requires no manual download — `models/open_clip` fetches `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` from the HuggingFace Hub on first use (internet access required).
- `MODEL_ROOT` can be the repo's `models/` directory (weights sit next to the loader code) or any separate location (e.g. a shared `med-vlms/` directory) — just set the env var accordingly.
- Each weight file is subject to its original model license; download from the official sources above.
