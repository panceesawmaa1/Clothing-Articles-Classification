"""
This module contains a lookup table with each implemented model string as a key
and its corresponding operation of the last convolution layer. This dict is useful
when calculating the receptive field of the model.
"""

last_node_lookup = {
    "baseline": "max_pooling2d_1/MaxPool",
    "classifier": "max_pooling2d_1/MaxPool",
    "deeperclassifier": "max_pooling2d_2/MaxPool"
}
