import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from pythor3.model.slm.plugins.passthrough.passthrough import (
        SequentialLayeredModelPassthrough,
        )
from pythor3.operation import fbcorr_
from pythor3.operation import lnorm_


def get_into_shape(x):
    if hasattr(x,'__iter__'):
        x = np.array(x)
        assert x.ndim == 1
        x = x[np.newaxis, :, np.newaxis, np.newaxis]
        x = x.astype(np.float32)
    return x


def dict_add(a, b):
    rval = dict(a)
    rval.update(b)
    return rval


class InvalidDescription(Exception):
    """Model description was invalid"""


class TheanoSLM(object):
    """
    SequentialLayeredModel clone implemented with Theano
    """

    def __init__(self, in_shape, description,
            dtype='float32', rng=888):

        # -- transpose shape to theano-convention (channel major)

        if len(in_shape) == 2:
            self.theano_in_shape = (1, 1,) +  in_shape
            self.pythor_in_shape = in_shape
        elif len(in_shape) == 3:
            self.theano_in_shape = (1, in_shape[2], in_shape[0], in_shape[1])
            self.pythor_in_shape = in_shape
        else:
            self.theano_in_shape = (in_shape[0],
                    in_shape[3],
                    in_shape[1],
                    in_shape[2])
            self.pythor_in_shape = in_shape[1:]

        assert len(self.theano_in_shape) == 4
        print 'TheanoSLM.theano_in_shape', self.theano_in_shape
        print 'TheanoSLM Description', description

        # This guy is used to generate filterbanks
        pythor_safe_description = get_pythor_safe_description(description)
        try:
            self.SLMP = SequentialLayeredModelPassthrough(
                    self.pythor_in_shape,
                    pythor_safe_description,
                    dtype=dtype)
        except ValueError, e:
            if 'negative dimensions' in str(e):
                print 'pythor_in_shape', self.pythor_in_shape
                print 'in_shape', in_shape
                raise InvalidDescription()
            raise

        del in_shape

        self.s_input = tensor.ftensor4('arr_in')
        self.rng = np.random.RandomState(rng)  # XXX check for rng being int

        x = self.s_input
        x_shp = self.theano_in_shape
        for layer_idx, layer_desc in enumerate(description):
            for op_name, op_params in layer_desc:
                init_fn = getattr(self, 'init_' + op_name)
                _D = dict_add(
                            op_params.get('kwargs', {}),
                            op_params.get('initialize', {}))
                x, x_shp = init_fn(x, x_shp, **_D)
                print 'TheanoSLM added layer', op_name, 'shape', x_shp

        if 0 == np.prod(x_shp):
            raise InvalidDescription()

        self.theano_out_shape = x_shp
        self.pythor_out_shape = x_shp[2], x_shp[3], x_shp[1]
        self.s_output = x

    def init_fbcorr_h(self, x, x_shp, **kwargs):
        min_out = kwargs.get('min_out', fbcorr_.DEFAULT_MIN_OUT)
        max_out = kwargs.get('max_out', fbcorr_.DEFAULT_MAX_OUT)
        kwargs['max_out'] = get_into_shape(max_out)
        kwargs['min_out'] = get_into_shape(min_out)
        return self.init_fbcorr(x, x_shp, **kwargs)

    def init_fbcorr(self, x, x_shp, n_filters,
            filter_shape,
            min_out=fbcorr_.DEFAULT_MIN_OUT,
            max_out=fbcorr_.DEFAULT_MAX_OUT,
            stride=fbcorr_.DEFAULT_STRIDE,
            mode=fbcorr_.DEFAULT_MODE,
            generate=None):
        # Reference implementation:
        # ../pythor3/pythor3/operation/fbcorr_/plugins/scipy_naive/scipy_naive.py
        if stride != fbcorr_.DEFAULT_STRIDE:
            raise NotImplementedError('stride is not used in reference impl.')
        fake_x = np.empty((x_shp[2], x_shp[3], x_shp[1]),
                x.dtype)
        kerns = self.SLMP._get_filterbank(fake_x,
                dict(n_filters=n_filters,
                    filter_shape=filter_shape,
                    generate=generate))
        kerns = kerns.transpose(0, 3, 1, 2).copy()[:,:,::-1,::-1]
        x = conv.conv2d(
                x,
                kerns,
                image_shape=x_shp,
                filter_shape=kerns.shape,
                border_mode=mode)
        if mode == 'valid':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] - filter_shape[0] + 1,
                    x_shp[3] - filter_shape[1] + 1)
        elif mode == 'full':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] + filter_shape[0] - 1,
                    x_shp[3] + filter_shape[1] - 1)
        else:
            raise NotImplementedError('fbcorr mode', mode)

        if min_out is None and max_out is None:
            return x, x_shp
        elif min_out is None:
            return tensor.minimum(x, max_out), x_shp
        elif max_out is None:
            return tensor.maximum(x, min_out), x_shp
        else:
            return tensor.clip(x, min_out, max_out), x_shp

    def boxconv(self, x, x_shp, kershp, channels=False):
        """
        channels: sum over channels (T/F)
        """
        kershp = tuple(kershp)
        if channels:
            rshp = (   x_shp[0],
                        1,
                        x_shp[2] - kershp[0] + 1,
                        x_shp[3] - kershp[1] + 1)
            kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
        else:
            rshp = (   x_shp[0],
                        x_shp[1],
                        x_shp[2] - kershp[0] + 1,
                        x_shp[3] - kershp[1] + 1)
            kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
            x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
            x = x.reshape(x_shp)
        try:
            rval = tensor.reshape(
                    conv.conv2d(x,
                        kerns,
                        image_shape=x_shp,
                        filter_shape=kerns.shape,
                        border_mode='valid'),
                    rshp)
        except Exception, e:
            if "Bad size for the output shape" in str(e):
                raise InvalidDescription()
            else:
                raise
        return rval, rshp

    def init_lnorm_h(self, x, x_shp, **kwargs):
        threshold = kwargs.get('threshold', lnorm_.DEFAULT_THRESHOLD)
        stretch = kwargs.get('stretch', lnorm_.DEFAULT_STRETCH)
        kwargs['threshold'] = get_into_shape(threshold)
        kwargs['stretch'] = get_into_shape(stretch)
        return self.init_lnorm(x, x_shp, **kwargs)

    def init_lnorm(self, x, x_shp,
            inker_shape=lnorm_.DEFAULT_INKER_SHAPE,    # (3, 3)
            outker_shape=lnorm_.DEFAULT_OUTKER_SHAPE,  # (3, 3)
            remove_mean=lnorm_.DEFAULT_REMOVE_MEAN,    # False
            div_method=lnorm_.DEFAULT_DIV_METHOD,      # 'euclidean'
            threshold=lnorm_.DEFAULT_THRESHOLD,        # 0.
            stretch=lnorm_.DEFAULT_STRETCH,            # 1.
            mode=lnorm_.DEFAULT_MODE,                  # 'valid'
            ):
        # Reference implementation:
        # ../pythor3/pythor3/operation/lnorm_/plugins/scipy_naive/scipy_naive.py
        EPSILON = lnorm_.EPSILON
        if mode != 'valid':
            raise NotImplementedError('lnorm requires mode=valid', mode)

        if outker_shape == inker_shape:
            size = np.asarray(x_shp[1] * inker_shape[0] * inker_shape[1],
                    dtype=x.dtype)
            ssq, ssqshp = self.boxconv(x ** 2, x_shp, inker_shape,
                    channels=True)
            xs = inker_shape[0] // 2
            ys = inker_shape[1] // 2
            if div_method == 'euclidean':
                if remove_mean:
                    arr_sum, _shp = self.boxconv(x, x_shp, inker_shape,
                            channels=True)
                    arr_num = x[:, :, xs:-xs, ys:-ys] - arr_sum / size
                    arr_div = EPSILON + tensor.sqrt(
                            tensor.maximum(0,
                                ssq - (arr_sum ** 2) / size))
                else:
                    arr_num = x[:, :, xs:-xs, ys:-ys]
                    arr_div = EPSILON + tensor.sqrt(ssq)
            else:
                raise NotImplementedError('div_method', div_method)
        else:
            raise NotImplementedError('outker_shape != inker_shape',outker_shape, inker_shape)
        if (hasattr(stetch, '__iter__') and (stretch != 1).any()) or stretch != 1:
            arr_num = arr_num * stretch
            arr_div = arr_div * stretch
        arr_div = tensor.switch(arr_div < (threshold + EPSILON), 1.0, arr_div)

        r = arr_num / arr_div
        r_shp = x_shp[0], x_shp[1], ssqshp[2], ssqshp[3]
        return r, r_shp

    def init_lpool_h(self, x, x_shp, **kwargs):
        order = kwargs.get('order', 1)
        kwargs['order'] = get_into_shape(order)
        return self.init_lpool(x, x_shp, **kwargs)

    def init_lpool(self, x, x_shp,
            ker_shape=(3, 3),
            order=1,
            stride=1,
            mode='valid'):

        if hasattr(order, '__iter__'):
            o1 = (order == 1).all()
            o2 = (order == order.astype(np.int)).all()
        else:
            o1 = order == 1
            o2 = (order == int(order))

        if o1:
            r, r_shp = self.boxconv(x, x_shp, ker_shape)
        elif o2:
            r, r_shp = self.boxconv(x ** order, x_shp, ker_shape)
            r = tensor.maximum(r, 0) ** (1.0 / order)
        else:
            r, r_shp = self.boxconv(abs(x) ** order, x_shp, ker_shape)
            r = tensor.maximum(r, 0) ** (1.0 / order)

        if stride > 1:
            r = r[:, :, ::stride, ::stride]
            # intdiv is tricky... so just use numpy
            r_shp = np.empty(r_shp)[:, :, ::stride, ::stride].shape
        return r, r_shp

    def get_theano_fn(self):
        try:
            fn = self._fn
        except AttributeError:
            fn = self._fn = theano.function([self.s_input], self.s_output,
                allow_input_downcast=True)
        return fn

    def process_batch(self, arr_in):
        fn = self.get_theano_fn()
        if arr_in.ndim == 4:
            channel_major_in = arr_in.transpose(0, 3, 1, 2)
        elif arr_in.ndim == 3:
            channel_major_in = arr_in[:,:,:,None].transpose(0, 3, 1, 2)
        else:
            raise NotImplementedError()
        return fn(channel_major_in).transpose(0, 2, 3, 1)

    def process(self, arr_in):
        """Return something like SequentialLayeredModel would have
        """
        rval = self.process_batch(arr_in[None,None,:,:])[0]
        if rval.shape[2] == 1:
            # -- drop the colour channel for single-channel images
            return rval[:, :, 0]
        else:
            return rval


    def init_fbcorr2_h(self, x, x_shp, **kwargs):
        min_out = kwargs.get('min_out', fbcorr_.DEFAULT_MIN_OUT)
        max_out = kwargs.get('max_out', fbcorr_.DEFAULT_MAX_OUT)
        kwargs['max_out'] = get_into_shape(max_out)
        kwargs['min_out'] = get_into_shape(min_out)
        exp1 = kwargs.get('exp1', 1)
        exp2 = kwargs.get('exp2', 1)
        kwargs['exp1'] = get_into_shape(exp1)
        kwargs['exp2'] = get_into_shape(exp2)
        return self.init_fbcorr2(x, x_shp, **kwargs)

    def init_fbcorr2(self, x, x_shp, n_filters,
            filter_shape,
            min_out=fbcorr_.DEFAULT_MIN_OUT,
            max_out=fbcorr_.DEFAULT_MAX_OUT,
            mode=fbcorr_.DEFAULT_MODE,
            exp1 = 1,
            exp2 = 1,
            generate1=None,
            generate2=None
            ):
        # Reference implementation:
        # ../pythor3/pythor3/operation/fbcorr_/plugins/scipy_naive/scipy_naive.py

        fake_x = np.empty((x_shp[2], x_shp[3], x_shp[1]),
                x.dtype)
        kerns1 = self.SLMP._get_filterbank(fake_x,
                dict(n_filters=n_filters,
                    filter_shape=filter_shape,
                    generate=generate1))
        kerns1 = kerns1.transpose(0, 3, 1, 2).copy()[:,:,::-1,::-1]
        kerns2 = self.SLMP._get_filterbank(fake_x,
                dict(n_filters=n_filters,
                    filter_shape=filter_shape,
                    generate=generate2))
        kerns2 = kerns2.transpose(0, 3, 1, 2).copy()[:,:,::-1,::-1]

        if (hasattr(exp1, '__iter__') and (exp1 != 1).any()) or exp1 != 1:
            x1 = x ** exp1
        if (hasattr(exp2, '__iter__') and (exp2 != 1).any()) or exp2 != 1:
            x2 = x ** exp2
        x1 = conv.conv2d(
                x1,
                kerns1,
                image_shape=x_shp,
                filter_shape=kerns1.shape,
                border_mode=mode)
        x2 = conv.conv2d(
                x2,
                kerns2,
                image_shape=x_shp,
                filter_shape=kerns2.shape,
                border_mode=mode)
        if (hasattr(exp1, '__iter__') and (exp1 != 1).any()) or exp1 != 1:
            x1 = tensor.maximum(x1, 0) ** (1.0 / exp1)
        if (hasattr(exp2, '__iter__') and (exp2 != 1).any()) or exp2 != 1:
            x2 = tensor.maximum(x2, 0) ** (1.0 / exp2)
        x = x1/x2

        if mode == 'valid':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] - filter_shape[0] + 1,
                    x_shp[3] - filter_shape[1] + 1)
        elif mode == 'full':
            x_shp = (x_shp[0], n_filters,
                    x_shp[2] + filter_shape[0] - 1,
                    x_shp[3] + filter_shape[1] - 1)
        else:
            raise NotImplementedError('fbcorr mode', mode)

        if min_out is None and max_out is None:
            return x, x_shp
        elif min_out is None:
            return tensor.minimum(x, max_out), x_shp
        elif max_out is None:
            return tensor.maximum(x, min_out), x_shp
        else:
            return tensor.clip(x, min_out, max_out), x_shp

    def init_rescale(self, x, x_shp,
            stride=1,
            mode='valid'):

        if stride > 1:
            r = x[:, :, ::stride, ::stride]
            # intdiv is tricky... so just use numpy
            r_shp = np.empty(x_shp)[:, :, ::stride, ::stride].shape
        else:
            r = x
            r_shp = x_shp

        return r, r_shp
