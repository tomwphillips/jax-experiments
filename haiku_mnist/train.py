import argparse
from functools import partial
import time
from typing import Iterator, Tuple

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


# TODO: check why @jax.jit fails here but speeds up first batch significantly
def loss_fn(images: jnp.array, labels: jnp.array) -> jnp.array:
    model = hk.Sequential(
        [
            hk.Flatten(),
            hk.Linear(512),
            jax.nn.relu,
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )

    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))


def sgd(param, gradient, learning_rate):
    return param - learning_rate * gradient


def main(epochs: int, batch_size: int, learning_rate: float) -> None:
    train_data = partial(load_mnist, "train", batch_size)
    train_eval_data = partial(load_mnist, "train", batch_size=10000)
    test_eval_data = partial(load_mnist, "test", batch_size=10000)

    loss_obj = hk.transform(loss_fn)
    key = jax.random.PRNGKey(42)
    params = loss_obj.init(key, *next(train_data()))

    sgd_ = partial(sgd, learning_rate=learning_rate)

    for epoch in range(epochs):
        start = time.time()

        for images, labels in train_data():
            grads = jax.grad(loss_obj.apply)(params, images, labels)
            params = jax.tree_multimap(
                sgd_, params, grads
            )  # TODO: check understanding of tree_multimap

        elapsed = time.time() - start

        # TODO: compute accuracy - need to move model out of loss_fn?
        train_loss = loss_obj.apply(params, *next(train_eval_data()))
        test_loss = loss_obj.apply(params, *next(test_eval_data()))
        print(
            f"Epoch: {epoch + 1} \t Time: {elapsed:.2f} s \t Training loss: {train_loss:.2E} \t Test loss: {test_loss:.2E}"
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--batch-size", default=128)
    parser.add_argument("--learning-rate", default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
