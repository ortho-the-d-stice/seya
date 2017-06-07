# encoding: utf-8
"""Test seya.layers.recurrent module"""

from __future__ import print_function

import unittest
import numpy as np
import theano
import tensorflow as tf
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import Model

from seya.utils import apply_model
floatX = K.common._FLOATX


class TestApply(unittest.TestCase):
    """Test apply methods"""

    def test_apply_model(self):
        """Test keras.models.Sequential.__call__"""
        nb_samples, input_dim, output_dim = 3, 10, 5
        model = Sequential()
        model.add(Dense(input_dim=10, units=output_dim))
        model.compile('sgd', 'mse')

        F = K.function(inputs=[model.layers[-1].input],
                       outputs=[model.layers[-1].output])

        x = np.random.randn(nb_samples, input_dim).astype(floatX)
        y1 = F([x])
        y2 = [model.predict(x)]

        # results of __call__ should match model.predict
        assert_allclose(y1, y2)


if __name__ == '__main__':
    theano.config.exception_verbosity = 'high'
    unittest.main(verbosity=2)
