import keras_applications as ka
import os
from .__version__ import __version__

import sys
sys.path.append(os.getcwd()+'/keras_classification_models')
import keras_classification_models.BiFPN
import keras_classification_models._layers


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils
