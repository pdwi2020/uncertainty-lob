# Google Colab Setup — DL Paper

## Overview

Colab provides T4 GPU (free) or L4 (Pro). Use for:
- Quick calibration experiments (temperature scaling, MC Dropout)
- Single-model training runs when Kaggle quota is exhausted
- Interactive debugging of Neural CDE

## MCP Tools

| Tool | Use Case |
|------|---------|
| `colab_execute` | Quick GPU test, short training runs |
| `colab_execute_file` | Run full training script |
| `colab_execute_notebook` | Run + collect model checkpoints + plots |

## Quick GPU Test

```python
colab_execute("""
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import torchcde
print(f"torchcde: {torchcde.__version__}")
""")
```

## Install Neural CDE Dependencies

```python
colab_execute("""
import subprocess
subprocess.run(["pip", "install", "torchcde", "torchdiffeq", "hmmlearn", "mapie", "-q"])
import torchcde, torchdiffeq
print("Packages installed successfully")
""")
```

## Run Training on Colab

```python
# Upload script via base64 if needed
import base64
with open("src/models/neural_cde.py", "rb") as f:
    script_b64 = base64.b64encode(f.read()).decode()

colab_execute(f"""
import base64
script = "{script_b64}"
with open("/tmp/neural_cde.py", "wb") as f:
    f.write(base64.b64decode(script))
exec(open("/tmp/neural_cde.py").read())
""", timeout=600)
```

## Download Model Checkpoint

```python
# After training, export checkpoint as base64
colab_execute("""
import base64, torch
# (model must be trained first)
torch.save(model.state_dict(), "/tmp/neural_cde_best.pt")
with open("/tmp/neural_cde_best.pt", "rb") as f:
    print("CKPT_START")
    print(base64.b64encode(f.read()).decode())
    print("CKPT_END")
""")
```

## Quota Notes

- Free tier: session time limit, usage cap (varies)
- Pro ($10/month): L4 GPU, longer sessions, priority access
- For DL paper: Kaggle is better (GPU T4, 30h/week, no session interruption)
- Use Colab only for quick experiments or when Kaggle quota exhausted
