import haiku as hk
import jax.numpy as jnp
import jax.random

from haiku_mnist.train import load_mnist, loss_fn, softmax_cross_entropy


def test_load_mnist_train():
    batch_size = 64
    train_data = load_mnist("train", batch_size=batch_size)
    images, labels = next(train_data)
    assert images.shape == (batch_size, 28, 28, 1)
    assert images.dtype == jnp.float32
    assert jnp.min(images) == 0
    assert jnp.max(images) == 1
    assert labels.shape == (batch_size,)


def test_load_mnist_test():
    batch_size = 64
    test_data = load_mnist("test", batch_size=batch_size)
    images, labels = next(test_data)
    assert images.shape == (batch_size, 28, 28, 1)
    assert labels.shape == (batch_size,)


def test_softmax_cross_entropy():
    logits = jnp.array([[1.0, 10.0, 100.0]])
    label = jnp.array([2])
    assert softmax_cross_entropy(logits, label).shape == (1,)


def test_loss_fn():
    key = jax.random.PRNGKey(0)
    images = jax.random.normal(key, (128, 28, 28))
    labels = jax.random.randint(key, (128,), 0, 9)
    loss_obj = hk.transform(loss_fn)
    params = loss_obj.init(key, images, labels)
    loss = loss_obj.apply(params, images, labels)
    assert loss.shape == ()
    assert loss > 0
