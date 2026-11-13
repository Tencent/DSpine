"""Neutral target rows outside each request's verified prefix."""


def logits_padding(controller, value, shape):
    """The caller overwrites every verified row; unused rows cannot emit tokens.

    A one-point distribution stays normalized under temperature, top-k,
    top-p and min-p, without filtering a full-vocabulary uniform dummy row.
    """
    padded = value.new_full(shape, float("-inf"))
    padded[..., 0] = 0.0
    return padded
