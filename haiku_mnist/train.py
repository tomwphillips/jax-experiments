import argparse
from functools import partial
import time
from typing import Any, Iterator, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


def load_mnist(split: str, batch_size: int) -> Iterator[Tuple[np.array, np.array]]:
    dataset = tfds.load("mnist", as_supervised=True, split=split)

    if split == "train":
        dataset = dataset.shuffle(10 * batch_size)

    dataset = dataset.batch(batch_size)

    for image, label in tfds.as_numpy(dataset):
        yield image.astype(jnp.float32) / 255, label


def softmax_cross_entropy(logits: jnp.array, label: jnp.array) -> jnp.array:
    ohe_label = hk.one_hot(label, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * ohe_label, axis=-1)


def accuracy(
    network: hk.Transformed, params: hk.Params, images: jnp.array, labels: jnp.array
) -> jnp.array:
    predictions = network.apply(params, images)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == labels)


def network_fn(images: jnp.array) -> jnp.array:
    network = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(512),
            jax.nn.relu,
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return network(images)


# TODO: check why @jax.jit fails here but speeds up first batch significantly
def loss_fn(
    network: hk.Transformed, params: hk.Params, images: jnp.array, labels: jnp.array,
) -> jnp.array:
    logits = network.apply(params, images)
    return jnp.mean(softmax_cross_entropy(logits, labels))


def sgd(param: hk.Params, gradient: hk.Params, learning_rate: float) -> hk.Params:
    return param - learning_rate * gradient


def main(epochs: int, batch_size: int, learning_rate: float) -> None:
    train_data = partial(load_mnist, "train", batch_size)
    train_eval_data = partial(load_mnist, "train", batch_size=10000)
    test_eval_data = partial(load_mnist, "test", batch_size=10000)

    network = hk.transform(network_fn)
    key = jax.random.PRNGKey(42)
    images, _ = next(train_data())
    params = network.init(key, images)

    sgd_ = partial(sgd, learning_rate=learning_rate)
    loss_fn_ = partial(loss_fn, network)

    for epoch in range(epochs):
        start = time.time()

        for images, labels in train_data():
            grads = jax.grad(loss_fn_)(params, images, labels)
            params = jax.tree_multimap(
                sgd_, params, grads
            )  # TODO: check understanding of tree_multimap

        elapsed = time.time() - start

        metrics = {
            "epoch": epoch + 1,
            "time": elapsed,
            "train_loss": loss_fn_(params, *next(train_eval_data())),
            "train_accuracy": accuracy(network, params, *next(train_eval_data())),
            "test_loss": loss_fn_(params, *next(test_eval_data())),
            "test_accuracy": accuracy(network, params, *next(test_eval_data())),
        }
        print(metrics)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--batch-size", default=128)
    parser.add_argument("--learning-rate", default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
