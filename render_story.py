"""Render the original three-scene film, using fresh monitored scene workers."""
import json
import shutil
import subprocess
import sys
import time
from settings import ROOT


def render_story(profile="quality", frames=124):
    from settings import RenderSettings
    settings = RenderSettings(profile=profile, frames=frames)
    if frames * 3 / 24 < 15:
        raise ValueError("Three scenes need at least 124 frames each for a 15-second film")
    from imageio_ffmpeg import get_ffmpeg_exe
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    started = time.time()
    report = dict(status="started", settings=settings.model_dump(), scenes=[])
    receipt = output / "story-job.json"
    try:
        scenes = json.loads((ROOT / "prompts/scenes.json").read_text())
        for scene in scenes:
            target = output / f"scene-{scene['scene']}.mp4"
            subprocess.run([sys.executable, str(ROOT / "monitor_run.py"),
                            "--prompt-file", str(ROOT / "prompts" / scene["prompt_file"]),
                            "--profile", profile, "--frames", str(frames),
                            "--output", str(target), "--seed", str(scene["seed"])], check=True)
            data = json.loads(target.with_suffix(".json").read_text())
            if data["status"] != "success":
                raise RuntimeError("Scene export failed")
            shutil.copyfile(ROOT / "memory-observation.json", target.with_suffix(".memory.json"))
            data["memory_observation"] = json.loads(target.with_suffix(".memory.json").read_text())
            report["scenes"].append(data)
            receipt.write_text(json.dumps(report, indent=2))
        listing = output / "concat.txt"
        listing.write_text("".join(f"file '{scene['video_file']}'\n" for scene in report["scenes"]))
        final = output / "storm-guardian-15s.mp4"
        subprocess.run([get_ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-t", "15", "-c:v", "libx264", "-crf", "18",
                        "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(final)], check=True)
        report.update(status="success", video_file=final.name)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.time() - started
        receipt.write_text(json.dumps(report, indent=2))
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quality", "fast", "lowmem", "vram8"], default="quality")
    parser.add_argument("--frames", type=int, default=124)
    args = parser.parse_args()
    print(render_story(args.profile, args.frames))
