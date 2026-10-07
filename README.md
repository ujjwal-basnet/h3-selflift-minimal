# H3 SelfLift

Minimal Python/PyTorch workflow: H3 hybrid INT8, four/eight-step Turbo and experimental SelfLift-zero. No ComfyUI runtime.

Requires Linux, Python 3.12, an NVIDIA CUDA GPU, system `ffmpeg` and [uv](https://docs.astral.sh/uv/getting-started/installation/). Budget at least 60 GB of disk. The `vram8` short test on a Colab T4 used 4.49 GiB sampled device memory and 9.46 GiB process RAM. It took 9 minutes 52 seconds, excluding setup. Longer clips and a physical 8 GB GPU remain unverified.

```bash
git clone https://github.com/ujjwal-basnet/h3-selflift-minimal.git
cd h3-selflift-minimal
uv sync --locked
uv run python download_models.py --steps 4
uv run python monitor_run.py --profile vram8 --frames 39 --output output/scene.mp4
```

For a notebook:

```bash
uv sync --locked --extra notebook
uv run python -m ipykernel install --user --name h3-selflift --display-name 'H3 SelfLift'
uv run jupyter lab studio.ipynb
```

Select the **H3 SelfLift** kernel. Optional Colab setup is inside the notebook; choose a GPU runtime and run the uv subprocess cells.

Edit the prompt files to change the video. `main.py` renders one scene. The tested `vram8` recipe produces 640×384 at 24 fps with audio and keeps Turbo weights in CPU RAM. Start with 39 frames (1.625 seconds); 124 frames requests about 5.17 seconds, with memory use still to be measured. Run one job at a time.

The default `quality` profile uses eight steps and 800×480. Download its adapter with `download_models.py --steps 8`. `render_story.py` uses that larger profile to assemble a 15-second film; the earlier T4 film took about 73 minutes and peaked at 14.54 GiB device memory.

```python
from pathlib import Path
from settings import RenderSettings
from pipeline import render

render('A quiet forest stream, steady camera, flowing water ambience.',
       Path('output/stream.mp4'), RenderSettings(profile='vram8', frames=39, seed=9175))
```

Frames must be 17n+5 between 22 and 345. Model files download into `models/`; generated files go into `output/`. Both are ignored by Git. `vram_limit` controls staging, not a hard total memory cap.

Upstream: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [hybrid checkpoint](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models), [DiffSynth-Studio and NF4 components](https://github.com/modelscope/DiffSynth-Studio), [LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [SelfLift research](https://arxiv.org/abs/2609.02036), and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen). Their licenses and model terms apply. This is an experimental H3 adaptation of SelfLift-zero; no trained SelfLift LoRA or upstream weights are included.

## Generated outputs

[Under-8-GB short test](samples/vram8-colab-39.mp4) · [15-second film](samples/storm-guardian-15s.mp4) · [Earlier four-step test](samples/lowmem-39.mp4). The notebook includes playback cells; see [sample settings](samples/README.md).

![Under-8-GB test frames](samples/vram8-preview.jpg)
