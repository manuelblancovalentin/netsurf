# Modern ML dependency boundary

The validated M2 environment is Python 3.11.17 with the direct versions in
`modern-ml-direct.txt`. NetSuRF requires NumPy 2 because TensorFlow 2.21's
validated `ml-dtypes` release requires it. The older `numpy<2` project bound
was a conservative M1.5 constraint, not a scientific requirement, and has
been replaced by `numpy>=2.0,<3` after the complete core suite passed.

The current MNIST benchmark is trained with TensorFlow/Keras, then exported to
NetSuRF's explicit `q<6,2,1>` codec. Faulted inference consumes those exact
encoded/decodeable checkpoint values. It does not claim to be HGQ2
quantization-aware training or HLS-equivalent inference.

HGQ2 and hls4ml are optional forward-looking dependencies. They are present to
evaluate a Keras 3 quantized hardware path, including attention. No M2 command
performs HLS conversion or synthesis.

Original QKeras is not part of this environment: it targets the older Keras
stack. QKeras-v3 is also not installed alongside HGQ2 because both currently
publish the `qkeras` import namespace. HGQ2 is the selected investigation path
because it exposes Keras 3 quantized dense, einsum, and multi-head-attention
layers. That choice does not establish hardware equivalence.

After installation, the acceptance checks are:

```bash
python -m pip check
python -c "import numpy, tensorflow, keras, hgq, hls4ml; print(numpy.__version__, tensorflow.__version__, keras.__version__, hgq.__version__, hls4ml.__version__)"
netsurf doctor
python -m pytest -q
```

CPU-only TensorFlow messages saying that CUDA drivers are absent are expected
on a CPU run. oneDNN may warn about floating-point operation ordering; stable
scientific comparison is checked from the fixed quantized checkpoint and
repeated payload digest. A TensorFlow `NodeDef` unknown-attribute warning was
observed during the bounded preparation run but did not stop training or the
numerical integration checks; it remains a compatibility warning to revisit
before changing TensorFlow versions.
