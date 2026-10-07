# Generated samples

- [Colab low-VRAM film](vram8-film-15s.mp4): 15 seconds, 360 frames, 640×384 at 24 fps with generated audio. Three fresh 124-frame scene workers, joined and trimmed to 15 seconds. Hybrid INT8 + four-step Turbo + SelfLift, `vram8` profile at code `18217af`. Colab T4: 3230.67 seconds for the story job excluding setup/downloads, 6.44 GiB sampled device peak, 9.41 GiB process RAM high-water mark. Full video/audio decode passed. A physical 8 GB card and continuous 15-second generation remain unverified.
- [Colab under-8-GB test](vram8-colab-39.mp4): 39 frames / 1.625 seconds, 640×384, 24 fps with audio. Hybrid INT8 + four-step Turbo + SelfLift, `vram8` profile at code `1eca7d5`. Colab T4: 592.13 seconds excluding setup, 4.49 GiB sampled device peak, 4.35 GiB peak PyTorch allocation, 9.46 GiB process RAM high-water mark. Device polling is once per second. See the film result above for 124-frame scenes; a physical 8 GB GPU remains unverified.
- [Storm Guardian film](storm-guardian-15s.mp4): 15 seconds, 800×480, 24 fps, generated audio. The original working eight-step H3 Turbo + SelfLift recipe.
- [Four-step test](lowmem-39.mp4): 39 frames / 1.625 seconds, 640×384, generated audio. Generated on the experimental branch at `b1b2702`; 441.21 seconds on T4, 9.41 GiB sampled device peak. This clip does **not** demonstrate under-8-GB operation.

These are generated outputs. Models and research retain the upstream attribution in the root README.
