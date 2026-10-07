"""CLI for one scene. Story rendering starts a fresh process for each scene."""
import argparse
from pathlib import Path
from settings import ROOT, RenderSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "prompts/storm-guardian-scene-1.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output/scene.mp4")
    parser.add_argument("--cache-weights", action="store_true", help="Experiment: retain some DiT weights in GPU memory")
    parser.add_argument("--profile", choices=["quality", "fast", "lowmem", "vram8"], default="quality")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=9175)
    parser.add_argument("--vram-limit", type=float, default=None)
    args = parser.parse_args()
    settings = RenderSettings(profile=args.profile, cache_weights=args.cache_weights, frames=args.frames, seed=args.seed, vram_limit=args.vram_limit)
    from pipeline import render
    print(render(args.prompt_file.read_text().strip(), args.output, settings), flush=True)


if __name__ == "__main__":
    main()
