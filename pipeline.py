"""Standalone, disk-staged H3 inference with the validated precision settings."""
import json
import os
import resource
import time
from pathlib import Path
from settings import ROOT, RenderSettings

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")


def load_pipeline(settings: RenderSettings):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from download_models import COMPANIONS, TURBO
    from hybrid_checkpoint import PATH, SHA256
    from hybrid_lora import HybridTurboLoader, turbo_metadata
    from safe_attention import enable_memory_efficient_attention
    from selflift import attach_selflift

    receipt = json.loads((ROOT / "hybrid-download.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("sha256") != SHA256:
        raise ValueError("Run download_models.py to verify the hybrid checkpoint")
    enable_memory_efficient_attention()
    offload = dict(offload_dtype="disk", offload_device="disk",
                   onload_dtype="disk", onload_device="disk",
                   preparing_dtype="disk", preparing_device="disk",
                   computation_device="cuda", computation_dtype=torch.float32)
    paths = [PATH, *(ROOT / "models/nf4" / name for name in COMPANIONS)]
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.float32, device="cuda",
        model_configs=[ModelConfig(path=str(path), **offload) for path in paths],
        processor_config=ModelConfig(path=str(ROOT / "models/h3/FL2VA/processor")),
        vram_limit=settings.vram_limit)
    video_decode, audio_decode = pipe.video_vae.decode_video, pipe.audio_vae.decode_audio

    def decode_video(latents, **kwargs):
        kwargs.update(dtype=torch.float32, tile_size=256)
        return video_decode(latents.float(), **kwargs)

    def decode_audio(latents, **kwargs):
        kwargs.update(dtype=torch.float32)
        return audio_decode(latents.float(), **kwargs)

    pipe.video_vae.decode_video, pipe.audio_vae.decode_audio = decode_video, decode_audio
    turbo_path = ROOT / "models/turbo" / TURBO
    metadata = turbo_metadata(turbo_path)
    pipe.lora_loader = HybridTurboLoader
    pipe.load_lora(pipe.dit, ModelConfig(path=str(turbo_path)), alpha=metadata["scale"])
    attach_selflift(pipe, height=480, width=800, transition_step=6, rho=0.4,
                    seed=9174, vae_tile_size=256)
    return pipe, metadata


def render(prompt: str, output: Path, settings: RenderSettings | None = None):
    """Render one scene and write an adjacent JSON receipt. Requires a CUDA GPU."""
    import torch
    from diffsynth.utils.data.audio_video import write_video_audio

    settings = settings or RenderSettings()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = dict(status="started", settings=settings.model_dump(), prompt=prompt,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  recipe="hybrid-int8-turbo8-selflift-zero", width=800, height=480)
    receipt = output.with_suffix(".json")
    receipt.write_text(json.dumps(report, indent=2))
    try:
        pipe, metadata = load_pipeline(settings)
        video, audio = pipe(prompt=prompt, width=640, height=384,
                            num_frames=settings.frames, num_inference_steps=8,
                            seed=settings.seed, flow_shift=6.0, cfg_scale=1,
                            tiled=True, tile_size=256, tile_overlap=32)
        if not torch.isfinite(torch.as_tensor(audio)).all():
            raise ValueError("Generated audio contains non-finite samples")
        write_video_audio(video=video, audio=audio, output_path=str(output),
                          fps=24, audio_sample_rate=32000)
        report.update(status="success", video_file=output.name, turbo_metadata=metadata,
                      selflift_diagnostics=pipe.selflift_diagnostics,
                      peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.time() - started,
                      peak_resident_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        receipt.write_text(json.dumps(report, indent=2))
    return output
