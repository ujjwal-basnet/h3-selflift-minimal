# H3 SelfLift

Minimal Python/PyTorch workflow: H3 hybrid INT8, eight-step Turbo and experimental SelfLift-zero. No ComfyUI runtime.

Requires Linux, Python 3.12, an NVIDIA CUDA GPU, system `ffmpeg` and [uv](https://docs.astral.sh/uv/getting-started/installation/). Budget at least 60 GB of disk. The tested T4 run peaked at 14.54 GiB GPU memory; 12 GB operation is unverified.

```bash
git clone https://github.com/ujjwal-basnet/h3-selflift-minimal.git
cd h3-selflift-minimal
uv sync --locked
uv run python download_models.py
uv run python main.py --prompt-file prompts/storm-guardian-scene-1.txt --output output/scene.mp4
```

For a notebook:

```bash
uv sync --locked --extra notebook
uv run python -m ipykernel install --user --name h3-selflift --display-name 'H3 SelfLift'
uv run jupyter lab studio.ipynb
```

Select the **H3 SelfLift** kernel. Optional Colab setup is inside the notebook; choose a GPU runtime and run the uv subprocess cells.

Edit the prompt files to change the video. `main.py` renders one scene; `uv run python render_story.py` renders three scenes in fresh workers and joins them into 15 seconds. Output is 800×480 at 24 fps with audio. The tested three-scene T4 run took about 73 minutes. Run one job at a time.

```python
from pathlib import Path
from settings import RenderSettings
from pipeline import render

render('A quiet forest stream, steady camera, flowing water ambience.',
       Path('output/stream.mp4'), RenderSettings(frames=39, seed=9175))
```

Frames must be 17n+5 between 22 and 345. Model files download into `models/`; generated files go into `output/`. Both are ignored by Git. `vram_limit` controls staging, not a hard total memory cap.

Upstream: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [hybrid checkpoint](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models), [DiffSynth-Studio and NF4 components](https://github.com/modelscope/DiffSynth-Studio), [LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [SelfLift research](https://arxiv.org/abs/2609.02036), and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen). Their licenses and model terms apply. This is an experimental H3 adaptation of SelfLift-zero; no trained SelfLift LoRA or upstream weights are included.
