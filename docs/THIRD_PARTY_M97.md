# M97 third-party source attribution

The streaming feature and event-counting implementation is adapted from
MLCommons Tiny, commit `4addd0fa08d216e20637637874e084895f289da4`:

- `benchmark/training/streaming_wakeword/get_dataset.py`, `get_lfbe_func`
- `benchmark/training/streaming_wakeword/str_ww_util.py`, event/debounce functions

Source: https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4
License: Apache License, Version 2.0, https://www.apache.org/licenses/LICENSE-2.0
The full pinned license is included in `licenses/MLCommons-Tiny-Apache-2.0.txt`.
The two original Python files have no file-header copyright notice. NetSuRF
changes are a bounded NumPy/TensorFlow adapter, exact input/output gates, private
TFLite XOR storage, immutable probe receipts and explicit task/capability limits.
The optional tests execute the original source functions from a local checkout.

FKeras formulas are independently implemented and numerically tested against
the pinned source method, not imported as a runtime package. Source and paper
attribution, revision, hashes and implementation differences are documented in
`M97_FKERAS_REPRODUCTION.md`. No pretrained weights, datasets or proprietary
assets are redistributed by this change.
