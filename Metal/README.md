# ArtCNN on Metal

Native Metal (MSL 4.1) versions of the ArtCNN GLSL shaders, generated from the official
`GLSL/*.glsl` files so the weights stay identical. Only `C4F16_DS` is generated so far.

- `translate.py` converts each mpv/libplacebo pass into a Metal compute kernel, line for line.
- `winograd.py` builds on it and replaces the five 16->16 convolutions with Winograd
  F(2x2, 3x3) kernels using weights extracted by `weights.py`. This is the shipped version.
- `ArtCNNMetal.h/.m` compiles a generated shader and encodes its passes into a command
  buffer. Input is a single-channel luma texture; output is an R16Float texture at 2x.

Regenerate:

```
python winograd.py ../GLSL/ArtCNN_C4F16_DS.glsl ArtCNN_C4F16_DS.metal
```

## Verification

`tools/` holds the checks. `reference.py` prepares BT.709 luma inputs and runs the official
ONNX model; `artcnn_metal_cli.m` runs a generated shader; `plbench.c` runs the original GLSL
through libplacebo on MoltenVK, the way Moonlight's Vulkan renderer does;
`numpy_forward.py` runs the network from the extracted weights.

Results on an M4 Pro (20-core GPU), macOS 27, `Images/artcnn_examples.png` luma (1600x1300):

| Implementation | PSNR vs ONNX (float32) | Max error (8-bit codes) |
|---|---|---|
| `translate.py --float32` | 138.6 dB | 1 (rounding) |
| NumPy from extracted weights | 137.6 dB | 1 (rounding) |
| `winograd.py` (half, shipped) | 59.6 dB | 2 |
| `translate.py` (half) | 59.5 dB | 2 |
| Original GLSL via libplacebo/MoltenVK | 56.0 dB | 3 |

Wall-clock time per frame, 300 pipelined frames, median of 3 runs:

| Input | Metal Winograd | Metal direct | libplacebo/MoltenVK |
|---|---|---|---|
| 1280x800 | 4.18 ms | 4.51 ms | 4.40 ms |
| 1512x982 | 6.04 ms | 6.48 ms | 6.37 ms |
| 1920x1080 | 8.34 ms | 9.00 ms | 8.82 ms |

libplacebo's own per-pass GPU timers report about 0.5 ms at 1512x982 on MoltenVK, which
would exceed the GPU's peak arithmetic rate by 9x, so only wall-clock time is compared.

`tools/mpsgraph_cli.m` runs the same network through Apple's MPSGraph convolutions
(float16, weights from `tools/export_weights.py`): 6.26 ms at 1512x982 against 6.05 ms for
the Winograd shader in the same session, with matching output (59.5 dB, at most 2 codes).

Per-pass profiling (`ARTCNN_PROFILE=1`) shows each 16->16 pass at about 1.05 ms whether it
uses Winograd or direct convolution, so these passes are limited by moving the 16-channel
activations through memory rather than by arithmetic. Fusing consecutive convolutions to
keep activations in threadgroup memory is the next optimization to try.
