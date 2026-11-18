#!/usr/bin/env python3
"""Run D: retrain the MLPerf Tiny ResNet-8 from a recorded seed and convert it to full INT8.

Recipe = reference scripts of mlcommons/tiny at commit 4addd0fa
(benchmark/training/image_classification/train.py and model_converter.py):
  500 epochs, batch 32, Adam, learning rate 1e-3 * 0.99**epoch, categorical cross-entropy,
  ImageDataGenerator(rotation 15, width/height shift 0.1, horizontal flip), all 50,000 CIFAR-10
  training images as raw 0-255 pixels; post-training full-INT8 quantization with the reference
  representative set (test images at calibration_samples_idxs.npy), int8 input and output.
Differences from the reference, all recorded in the manifest:
  * a fixed seed (the reference sets none);
  * conv_filters = 16, the architecture of the pinned pretrained checkpoint (the reference train.py
    at this commit defaults to 26); the weight shapes are asserted equal to the pinned HDF5 model;
  * no plots; a thread limit; steps_per_epoch = ceil(50000 / 32); batches are augmented by several
    threads, so a rerun with the same seed gives a statistically equivalent, not bit-identical, model.
The model definition is imported from the reference keras_model.py, unmodified.
Outputs go to a new directory (refuses an existing one).  Nothing pinned is written.
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
REF_COMMIT = "4addd0fa08d216e20637637874e084895f289da4"
PINNED_H5 = Path("/data/models/models--mlcommons--tiny/snapshots/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/image_classification/trained_models/pretrainedResnet.h5")
CIFAR = Path("/data/datasets/cifar10/cifar-10-batches-py")
EPOCHS, BS = 500, 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cifar(np):
    """Same arrays as the reference load_cifar_10_data: uint8 NHWC images, integer labels."""
    def unpickle(name):
        with open(CIFAR / name, "rb") as f:
            return pickle.load(f, encoding="bytes")
    parts = [unpickle(f"data_batch_{i}") for i in range(1, 6)]
    train = np.rollaxis(np.vstack([p[b"data"] for p in parts]).reshape(-1, 3, 32, 32), 1, 4)
    train_y = np.array(sum((p[b"labels"] for p in parts), []))
    t = unpickle("test_batch")
    test = np.rollaxis(t[b"data"].reshape(-1, 3, 32, 32), 1, 4)
    return train, train_y, test, np.array(t[b"labels"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only: steps per epoch")
    ap.add_argument("--int8-eval", type=int, default=10000)
    ap.add_argument("--loader-threads", type=int, default=1,
                    help="threads that augment batches; batch order is kept, augmentation draws are not bit-reproducible above 1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    sys.path.insert(0, str(REF))
    import keras_model  # reference model definition, unmodified

    train_x, train_y, test_x, test_y = load_cifar(np)
    train_labels = tf.keras.utils.to_categorical(train_y)
    model = keras_model.resnet_v1_eembc(conv_filters=16)
    pinned = tf.keras.models.load_model(PINNED_H5, compile=False)
    shapes = [tuple(w.shape) for w in model.weights]
    assert shapes == [tuple(w.shape) for w in pinned.weights], "architecture differs from the pinned checkpoint"
    del pinned

    def lr_schedule(epoch):
        return 0.001 * (0.99 ** epoch)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, validation_split=0.2)
    datagen.fit(train_x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics="accuracy")
    steps = args.max_steps or math.ceil(len(train_x) / BS)
    started = time.time()
    history = model.fit(datagen.flow(train_x, train_labels, batch_size=BS, seed=args.seed), steps_per_epoch=steps,
                        epochs=args.epochs, verbose=2, workers=args.loader_threads, max_queue_size=64,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                                   tf.keras.callbacks.CSVLogger(str(args.out / "training_log.csv"))])
    train_seconds = time.time() - started
    model.save(args.out / "trained.h5")
    fp32_acc = float(np.mean(np.argmax(model.predict(test_x.astype(np.float32), batch_size=500, verbose=0), axis=1) == test_y))

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
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    n = min(args.int8_eval, len(test_x)); correct = 0
    for i in range(n):
        interp.set_tensor(inp["index"], (test_x[i].astype(np.int16) - 128).astype(np.int8)[None])
        interp.invoke()
        correct += int(np.argmax(interp.get_tensor(outp["index"])[0]) == test_y[i])
    manifest = dict(
        schema="paper1-runD-retrain/1.0", model="mlperf_tiny_resnet8", seed=args.seed, epochs=args.epochs,
        steps_per_epoch=steps, batch_size=BS, smoke=bool(args.max_steps or args.epochs != EPOCHS),
        reference_commit=REF_COMMIT, reference_keras_model_sha256=sha256(REF / "keras_model.py"),
        reference_calibration_index_sha256=sha256(REF / "calibration_samples_idxs.npy"),
        differences_from_reference=["fixed seed", "conv_filters=16 (pinned checkpoint architecture)", "no plots",
                                    "thread limit", "steps_per_epoch=ceil(50000/32)",
                                    f"{args.loader_threads} augmentation threads (not bit-reproducible when above 1)"],
        weight_shapes=[list(s) for s in shapes], tensorflow=tf.__version__, numpy=np.__version__, threads=args.threads,
        train_seconds=train_seconds, final_train_accuracy=float(history.history["accuracy"][-1]),
        fp32_test_accuracy=fp32_acc, int8_test_accuracy=correct / n, int8_test_images=n,
        int8_input_quantization=[float(inp["quantization"][0]), int(inp["quantization"][1])],
        trained_h5_sha256=sha256(args.out / "trained.h5"), trained_quant_tflite_sha256=sha256(args.out / "trained_quant.tflite"),
        script_sha256=sha256(Path(__file__)))
    with open(args.out / "manifest.json", "x") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("seed", "epochs", "train_seconds", "fp32_test_accuracy", "int8_test_accuracy")}), flush=True)


if __name__ == "__main__":
    main()
