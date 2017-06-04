import numpy as np
import theano
import theano.tensor as T

from keras.layers.core import Layer, Lambda
from keras import backend as K
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


class MaskedLayer(Layer):
    '''If your layer trivially supports masking
    (by simply copying the input mask to the output),
    then subclass MaskedLayer instead of Layer,
    and make sure that you incorporate the input mask
    into your calculation of get_output().
    '''
    def supports_masked_input(self):
        return True

    def get_input_mask(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output_mask(train)
        else:
            return None

    def get_output_mask(self, train=False):
        ''' The default output mask is just the input mask unchanged.
        Override this in your own implementations if,
        for instance, you are reshaping the input'''
        return self.get_input_mask(train)

class LambdaMerge(Lambda):
    '''LambdaMerge layer for evaluating an arbitrary Theano / TensorFlow
    function over multiple inputs.

    # Output shape
        Specified by output_shape argument

    # Arguments
        layers - Input layers. Similar to layers argument of Merge
        function - The function to be evaluated. Takes one argument:
            list of outputs from input layers
        output_shape - Expected output shape from function.
            Could be a tuple or a function of list of input shapes
        arguments: optional dictionary of keyword arguments to be passed
            to the function.
    '''
    def __init__(self, layers, function, output_shape=None, arguments={}):
        if len(layers) < 2:
            raise Exception('Please specify two or more input layers '
                            '(or containers) to merge.')
        self.layers = layers
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        self.arguments = arguments
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.trainable_weights:
                    self.trainable_weights.append(p)
                    self.constraints.append(c)
        self.function = function
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            assert hasattr(output_shape, '__call__'), 'In LambdaMerge, `output_shape` must be a list, a tuple, or a function.'
            self._output_shape = output_shape
        super(Lambda, self).__init__()

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        if self._output_shape is None:
            return input_shapes[0]
        elif type(self._output_shape) in {tuple, list}:
            return (input_shapes[0][0],) + self._output_shape
        else:
            shape = self._output_shape(input_shapes)
            if type(shape) not in {list, tuple}:
                raise Exception('In LambdaMerge, the `output_shape` function must return a tuple.')
            return tuple(shape)

    def get_params(self):
        return self.trainable_weights, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        inputs = [layer.get_output(train) for layer in self.layers]
        arguments = self.arguments
        arg_spec = inspect.getargspec(self.function)
        if 'train' in arg_spec.args:
            arguments['train'] = train
        return self.function(inputs, **arguments)

    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res

    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].trainable_weights) + len(self.non_trainable_weights)
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        # note: not serializable at the moment.
        config = {'name': self.__class__.__name__,
                  'layers': [l.get_config() for l in self.layers],
                  'function': self.function,
                  'output_shape': self._output_shape,
                  'arguments': self.arguments}
        base_config = super(LambdaMerge, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WinnerTakeAll2D(Layer):
    def __init__(self, **kwargs):
        super(WinnerTakeAll2D, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if train:
            M = K.max(X, axis=(2, 3), keepdims=True)
            R = K.switch(K.equal(X, M), X, 0.)
            return R
        else:
            return X


class Lambda(MaskedLayer):
    def __init__(self, func, output_shape, ndim=2, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=ndim)
        self.func = func
        self._output_shape = output_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.func(X)

    @property
    def output_shape(self):
        return self._output_shape


class Pass(MaskedLayer):
    ''' Do literally nothing
        It can the first layer
    '''
    def __init__(self, ndim=2, **kwargs):
        super(Pass, self).__init__(**kwargs)
        self.input = K.placeholder(ndim=ndim)

    def get_output(self, train=False):
        X = self.get_input(train)
        return X


class GaussianProd(MaskedLayer):
    '''
        Multiply by Gaussian noise.
        Similar to dropout but with gaussians instead of binomials.
        The way they have this at Keras is not the way we need for
        Variational AutoEncoders.
    '''
    def __init__(self, avg=0., std=1., **kwargs):
        super(GaussianProd, self).__init__(**kwargs)
        self.std = std
        self.avg = avg
        self.srng = RandomStreams(seed=np.random.randint(10e6))

    def get_output(self, train=False):
        X = self.get_input(train)
        X *= self.srng.normal(size=X.shape,
                              avg=self.avg,
                              std=self.std,
                              dtype=floatX)
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "avg": self.avg,
                "std": self.std}


class Replicator(MaskedLayer):
    '''
        WARN: use `keras.layer.RepeatVector` instead.

        Replicates an input matrix across a new second dimension.
        Originally useful for broadcasting a fixed input into a scan loop.
        Think conditional RNNs without the need to rewrite the RNN class.
    '''
    def __init__(self, leng):
        super(Replicator, self).__init__()
        raise ValueError("Deprecated. Use `keras.layers.RepeatVector instead`")
        self.ones = T.ones((leng,))
        self.input = T.matrix()

    def get_output(self, train=False):
        X = self.get_input(train)
        output = X[:, None, :] * self.ones[None, :, None]
        return output


class Unpool(Layer):
    '''Unpooling layer for convolutional autoencoders
    inspired by: https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.py

    Parameter:
    ----------
    ds: list with two values each one defines how much that dimension will
    be upsampled.
    '''
    def __init__(self, ds):
        raise ValueError("Deprecated. Use `keras.layers.convolutional.UpSample instead`")
        super(Unpool, self).__init__()
        self.input = T.tensor4()
        self.ds = ds

    def get_output(self, train=False):
        X = self.get_input(train)
        output = X.repeat(self.ds[0], axis=2).repeat(self.ds[1], axis=3)
        return output


class TimePicker(MaskedLayer):
    def __init__(self, time=-1):
        '''Picks a single value in time from a recurrent layer
           without forgeting its input mask'''
        super(TimePicker, self).__init__()
        self.time = time
        self.input = T.tensor3()

    def get_output(self, train=False):
        X = self.get_input(train)
        return X[:, self.time, :]

    @property
    def output_shape(self):
        return self.input_shape[0, 2]
