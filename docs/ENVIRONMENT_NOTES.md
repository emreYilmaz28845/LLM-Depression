# Environment Notes

The target runtime is the MareNostrum5-compatible `qwen_mn5_rebuilt` environment.

Important points:
- The transferred local copy at `/media/emre/Backup/AudioLLM/qwen_mn5_rebuilt` is a copied virtual environment.
- Its package metadata is readable, but it is not the authoritative place to run `pip freeze` or `conda env export`.
- The local copy has broken absolute interpreter references, so the final authoritative environment capture should be generated on MareNostrum5 or from a repaired clone of that environment.

Files in this repository:
- `requirements_mn5_freeze.txt`: conservative package versions recovered from transferred metadata
- `environment_mn5_no_builds.yml`: conservative environment template recovered from transferred metadata
- `scripts/capture_environment.sh`: the command sequence to regenerate all captures on the actual runtime environment

Commands to run on MareNostrum5:

```bash
python -V
pip freeze
conda env export --no-builds
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)"
```

Recovered package versions from transferred metadata:
- `torch==2.3.0+cu121`
- `transformers==4.55.0`
- `accelerate==1.8.1`
- `peft==0.17.0`
- `deepspeed==0.14.5`
- `librosa==0.11.0`
- `soundfile==0.13.1`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `scikit-learn==1.7.0`
