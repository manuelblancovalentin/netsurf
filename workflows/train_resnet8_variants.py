#!/usr/bin/env python3
"""Run J: functional but "marginal" ResNet-8 checkpoints for the ranking ablation.

Two families, both converted to full INT8 exactly as in retrain_resnet8.py (reference converter, reference
representative set) and both with the architecture of the pinned checkpoint:

  undertrained  the reference training recipe (mlcommons/tiny at 4addd0fa: batch 32, Adam, learning rate
                1e-3 * 0.99**epoch, rotation 15, shifts 0.1, horizontal flip, all 50,000 images), stopped
                after --epochs epochs instead of 500.
  pruned        the pinned pretrained FP32 model (pretrainedResnet.h5), magnitude-pruned per layer to
                --sparsity (Conv2D and Dense kernels only; the smallest |w| of each kernel are set to zero),
                then fine-tuned for --epochs epochs with the same augmentation, Adam with learning rate
                1e-3 * 0.95**epoch, the pruned positions forced back to zero after every batch.  This pruning recipe is ours;
                the reference repository has none.

Each run writes a new directory with the HDF5 model, the INT8 TFLite model, a log and a manifest (hashes,
FP32 and INT8 test accuracy, sparsity of the INT8 kernels).  Nothing pinned is written.

Fixed before any ranking comparison: a variant counts as "functional" if its INT8 test accuracy is at least
80% (the pinned checkpoint has 87%); a variant below that is reported and excluded from conclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

REF = Path("/projects/netsurf/manuelbv/workspace/implementation-worktrees/paper1-tecs-revision-v1/workspace/external/mlcommons_tiny/benchmark/training/image_classification")
PINNED_H5 = Path("/data/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/image_classification/trained_models/pretrainedResnet.h5")
CIFAR = Path("/data/datasets/cifar10/cifar-10-batches-py")
BS = 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cifar(np):
    def unpickle(name):
        with open(CIFAR / name, "rb") as f:
            return pickle.load(f, encoding="bytes")
    parts = [unpickle(f"data_batch_{i}") for i in range(1, 6)]
    train = np.rollaxis(np.vstack([p[b"data"] for p in parts]).reshape(-1, 3, 32, 32), 1, 4)
    train_y = np.array(sum((p[b"labels"] for p in parts), []))
    t = unpickle("test_batch")
    return train, train_y, np.rollaxis(t[b"data"].reshape(-1, 3, 32, 32), 1, 4), np.array(t[b"labels"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["undertrained", "pruned"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--sparsity", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only")
    ap.add_argument("--int8-eval", type=int, default=10000)
    args = ap.parse_args()
    assert (args.variant == "pruned") == (0 < args.sparsity < 1)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    sys.path.insert(0, str(REF))
    import keras_model

    train_x, train_y, test_x, test_y = load_cifar(np)
    train_labels = tf.keras.utils.to_categorical(train_y)
    pinned = tf.keras.models.load_model(PINNED_H5, compile=False)
    callbacks = [tf.keras.callbacks.CSVLogger(str(args.out / "training_log.csv"))]
    masks = {}
    if args.variant == "undertrained":
        model = keras_model.resnet_v1_eembc(conv_filters=16)
        assert [tuple(w.shape) for w in model.weights] == [tuple(w.shape) for w in pinned.weights]
        optimizer = tf.keras.optimizers.Adam()
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.99 ** epoch)))
    else:
        model = pinned
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                w = layer.kernel.numpy()
                thr = np.quantile(np.abs(w), args.sparsity)
                masks[layer.name] = (np.abs(w) > thr).astype(np.float32)
                layer.kernel.assign(w * masks[layer.name])

        class KeepPruned(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                for layer in self.model.layers:
                    if layer.name in masks:
                        layer.kernel.assign(layer.kernel * masks[layer.name])

        optimizer = tf.keras.optimizers.Adam()
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * (0.95 ** epoch)))
        callbacks.append(KeepPruned())
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.2)
    datagen.fit(train_x)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics="accuracy")
    steps = args.max_steps or math.ceil(len(train_x) / BS)
    started = time.time()
    history = model.fit(datagen.flow(train_x, train_labels, batch_size=BS, seed=args.seed), steps_per_epoch=steps,
                        epochs=args.epochs, verbose=2, callbacks=callbacks)
    train_seconds = time.time() - started
    model.save(args.out / "trained.h5")
    fp32_acc = float(np.mean(np.argmax(model.predict(test_x.astype(np.float32), batch_size=500, verbose=0), axis=1) == test_y))
    fp32_sparsity = {name: float(1.0 - m.mean()) for name, m in masks.items()}

    idx = np.load(REF / "calibration_samples_idxs.npy")

    def representative():
        for i in idx:
            yield [np.expand_dims(np.array(test_x[i], dtype=np.float32), axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quant = converter.convert()
    (args.out / "trained_quant.tflite").write_bytes(quant)
    interp = tf.lite.Interpreter(model_content=quant, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    kernels = [d for d in interp.get_tensor_details() if d["dtype"] == np.int8 and len(d["shape"]) in (2, 4) and d["quantization_parameters"]["scales"].size >= 1
               and "bias" not in d["name"].lower()]
    zero_codes = total_codes = 0
    for d in kernels:
        try:
            t = interp.get_tensor(d["index"])
        except ValueError:
            continue
        if t.size >= 100 and d["quantization_parameters"]["scales"].size in (1, t.shape[0]) and d["name"] and "Conv2D" in d["name"] or "MatMul" in d["name"]:
            zero_codes += int((t == 0).sum()); total_codes += int(t.size)
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    n = min(args.int8_eval, len(test_x)); correct = 0
    for i in range(n):
        interp.set_tensor(inp["index"], (test_x[i].astype(np.int16) - 128).astype(np.int8)[None])
        interp.invoke()
        correct += int(np.argmax(interp.get_tensor(outp["index"])[0]) == test_y[i])
    manifest = dict(schema="paper1-runJ-variant/1.0", model="mlperf_tiny_resnet8", variant=args.variant, epochs=args.epochs,
                    sparsity_target=args.sparsity, seed=args.seed, steps_per_epoch=steps, smoke=bool(args.max_steps),
                    reference_commit="4addd0fa08d216e20637637874e084895f289da4", pinned_h5_sha256=sha256(PINNED_H5),
                    fp32_kernel_sparsity=fp32_sparsity, int8_kernel_zero_fraction=(zero_codes / total_codes if total_codes else None),
                    int8_kernel_codes_counted=total_codes, train_seconds=train_seconds,
                    final_train_accuracy=float(history.history["accuracy"][-1]), fp32_test_accuracy=fp32_acc,
                    int8_test_accuracy=correct / n, int8_test_images=n, tensorflow=tf.__version__, threads=args.threads,
                    trained_h5_sha256=sha256(args.out / "trained.h5"), trained_quant_tflite_sha256=sha256(args.out / "trained_quant.tflite"),
                    script_sha256=sha256(Path(__file__)))
    with open(args.out / "manifest.json", "x") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("variant", "epochs", "sparsity_target", "fp32_test_accuracy", "int8_test_accuracy", "int8_kernel_zero_fraction")}), flush=True)


if __name__ == "__main__":
    main()
