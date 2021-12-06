"""
Receptive Field Calculation Module.
"""

from clothing_classifier.models.base_model import BaseModel
import libraries.receptive_field.receptive_field as rf
import tensorflow._api.v2.compat.v1 as tf
from clothing_classifier.utils.lookup import last_node_lookup


def calculate_receptive_field(model: BaseModel):
    """
    This function calculates the receptive field, effective
    stride, and effective pad in both dimensions.
    :param model: Model instance
    :return: overall receptive field
    """
    g = tf.Graph()
    with g.as_default():
        tf.keras.backend.set_learning_phase(0)
        model.build()

    last_node = last_node_lookup[model.__str__()]

    rf_x, rf_y, eff_stride_x, eff_stride_y, eff_pad_x, eff_pad_y = \
        rf.compute_receptive_field_from_graph_def(
            g.as_graph_def(), 'input_layer_input', last_node)

    print()
    print("Receptive field report:")
    print("==============================================")

    print("Receptive field size (horizontal) = ", rf_x)
    print("Receptive field size (vertical) = ", rf_y)
    print("Effective stride (horizontal) = ", eff_stride_x)
    print("Effective stride (vertical) = ", eff_stride_y)
    print("Effective padding (horizontal) = ", eff_pad_x)
    print("Effective padding (vertical) = ", eff_pad_y)

    print("==============================================")

    return rf_x
