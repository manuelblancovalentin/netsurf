# Modern ML, quantization, and hardware-backend boundary

Validated on Kona on 2026-10-22:

| Component | Version | M2 role |
|---|---:|---|
| Python | 3.11.17 | canonical runtime |
| NumPy | 2.4.6 | core arrays and exact codec |
| TensorFlow | 2.21.0 | modern MNIST training/inference |
| Keras | 3.15.1 | source-model construction/training |
| HGQ2 | 0.2.0 | future Keras 3 quantized/HLS investigation |
| hls4ml | 1.3.0 | future conversion investigation |
| Matplotlib | 3.11.2 | report plots from saved measurements |
| SciPy | 1.17.1 | Student-t intervals when available |

TensorFlow publishes Python 3.11 wheels for the selected release. hls4ml 1.3
documents Keras 3, QKeras-v3, and HGQ2 frontends. Its support matrix marks
multi-head attention as experimental, supported by the Vitis backend and not
the Catapult backend. The Keras frontend documents HGQ2 `QMultiHeadAttention`
and `QEinsum`; original QKeras is not Keras-3 compatible. These are capability
signals, not a validated NetSuRF hardware path.

Primary references:

- TensorFlow pip support: <https://www.tensorflow.org/install/pip>
- hls4ml setup and tool versions: <https://fastmachinelearning.org/hls4ml/intro/setup.html>
- hls4ml frontend/backend support matrix: <https://fastmachinelearning.org/hls4ml/intro/status.html>
- hls4ml Keras/HGQ2 frontend: <https://fastmachinelearning.org/hls4ml/frontend/keras.html>
- HGQ2 package: <https://pypi.org/project/hgq2/>
- QKeras-v3 repository: <https://github.com/fastmachinelearning/qkerasV3>
- Keras native quantization overview: <https://keras.io/guides/quantization_overview/>

## Decision for M2

The immediate benchmark uses Keras for training and NetSuRF `BitCodec` for an
explicit six-bit two's-complement storage/inference representation. This makes
every injected bit independently traceable. Keras native quantization currently
targets built-in int4/int8 schemes rather than the required `q<6,2,1>` contract,
so it is not substituted silently. HGQ2 is the preferred next investigation for
quantized attention and hls4ml, but the current MNIST checkpoint is not claimed
to use HGQ2 quantization-aware training.

For dense networks, both Vitis and Catapult remain plausible hls4ml backends.
For attention, the documented hls4ml 1.3 matrix currently favors Vitis; Catapult
cannot be claimed compatible until the relevant operators appear and pass
NetSuRF numerical tests. No HLS conversion, C simulation, RTL, synthesis, or
licensed-tool run is part of M2.

Acceptance before a hardware claim requires fixed-vector equivalence across:

1. NetSuRF quantized Python inference.
2. The selected HGQ2 or explicit fixed-point Keras representation.
3. hls4ml C simulation.
4. RTL simulation for a named backend/tool release.
5. A machine-readable logical-bit-to-storage mapping.

## Optional Hugging Face boundary

Hugging Face remains optional. A future adapter may use the `hf` CLI to obtain a
model or dataset pinned to an immutable commit, for example:

```bash
hf download <repository> --revision <commit-sha> --local-dir <shared-input-directory>
```

NetSuRF will then explicitly implement the architecture, map tensors, and test
numerical equivalence before registration. It will not clone arbitrary Hub
models into fault-injectable architectures, and Hugging Face is not a required
M2 dependency.
