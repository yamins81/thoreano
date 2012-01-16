import copy
import sys
import time
import os
import itertools
import tempfile
import os.path as path
import cPickle
import hashlib

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
from bson import BSON, SON


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


def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()


def get_pythor_safe_description(description):
    description = copy.deepcopy(description)
    for layer_idx, layer_desc in enumerate(description):
        for (op_idx,(op_name, op_params)) in enumerate(layer_desc):
            if op_name.endswith('_h'):
                newname = op_name[:-2]
                layer_desc[op_idx] = (newname,op_params)
            if op_name == 'fbcorr2':
                newname = 'fbcorr'
                op_params['initialize'].pop('exp1')
                op_params['initialize'].pop('exp2')
                op_params['initialize']['generate'] = op_params['initialize'].pop('generate1')
                op_params['initialize'].pop('generate2')
                layer_desc[op_idx] = (newname,op_params)
            elif op_name == 'rescale':
                newname = 'lpool'
                op_params['kwargs']['ker_shape'] = (1,1)
                op_params['kwargs']['order'] = 1
                layer_desc[op_idx] = (newname,op_params)
            elif op_name == 'activ':
                layer_desc[op_idx] = None
        description[layer_idx] = [_x for _x in layer_desc if _x is not None]
    return description


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

    def init_fbcorr(self, x, x_shp, n_filters,
            filter_shape,
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

        return x, x_shp

    def init_fbcorr2_h(self, x, x_shp, **kwargs):
        exp1 = kwargs.get('exp1', 1)
        exp2 = kwargs.get('exp2', 1)
        kwargs['exp1'] = get_into_shape(exp1)
        kwargs['exp2'] = get_into_shape(exp2)
        return self.init_fbcorr2(x, x_shp, **kwargs)

    def init_fbcorr2(self, x, x_shp, n_filters,
            filter_shape,
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
        else:
            x1 = x
        if (hasattr(exp2, '__iter__') and (exp2 != 1).any()) or exp2 != 1:
            x2 = x ** exp2
        else:
            x2 = x
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
            x1 = tensor.real(x1 ** (1.0 / exp1))
        if (hasattr(exp2, '__iter__') and (exp2 != 1).any()) or exp2 != 1:
            x2 = tensor.real(x2 ** (1.0 / exp2))
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

        return x, x_shp

    def init_activ_h(self, x, x_shp, **kwargs):
        min_out = kwargs.get('min_out', fbcorr_.DEFAULT_MIN_OUT)
        max_out = kwargs.get('max_out', fbcorr_.DEFAULT_MAX_OUT)
        kwargs['max_out'] = get_into_shape(max_out)
        kwargs['min_out'] = get_into_shape(min_out)
        return self.init_activ(x, x_shp, **kwargs)

    def init_activ(self, x, x_shp,
                   min_out=fbcorr_.DEFAULT_MIN_OUT,
                   max_out=fbcorr_.DEFAULT_MAX_OUT):
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
        EPSILON = lnorm_.EPSILON  # 0.0001
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
        if (hasattr(stretch, '__iter__') and (stretch != 1).any()) or stretch != 1:
            arr_num = arr_num * stretch
            arr_div = arr_div * stretch
        if threshold is not None:
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

        if ker_shape != (1,1):
            if o1:
                r, r_shp = self.boxconv(x, x_shp, ker_shape)
            elif o2:
                r, r_shp = self.boxconv(x ** order, x_shp, ker_shape)
                r = tensor.maximum(r, 0) ** (1.0 / order)
            else:
                r, r_shp = self.boxconv(abs(x) ** order, x_shp, ker_shape)
                r = tensor.maximum(r, 0) ** (1.0 / order)
        else:
            r, r_shp = x, x_shp

        return r, r_shp

    def init_rescale(self, x, x_shp, stride=1):
        if stride > 1:
            r = x[:, :, ::stride, ::stride]
            # intdiv is tricky... so just use numpy
            r_shp = np.empty(x_shp)[:, :, ::stride, ::stride].shape
        else:
            r, r_shp = x, x_shp
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



TEST = False
TEST_NUM = 40
DEFAULT_TLIMIT = None

class ExtractedFeatures(object):
    def __init__(self, X, feature_shps, batchsize, slms, filenames,
                 tlimit=DEFAULT_TLIMIT, file_out = False):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """

        self.filenames = []
        self.features = []
        self.feature_shps = feature_shps

        for feature_shp, filename, slm in zip(feature_shps, filenames, slms):
            size = 4 * np.prod(feature_shp)
            print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
            memmap = file_out or use_memmap(size)
            if memmap:
                print('Creating memmap %s for features of shape %s' % (
                                                      filename, str(feature_shp)))
                features_fp = np.memmap(filename,
                    dtype='float32',
                    mode='w+',
                    shape=feature_shp)
            else:
                print('Using memory for features of shape %s' % str(feature_shp))
                features_fp = np.empty(feature_shp,dtype='float32')

            if TEST:
                print('TESTING')

            i = 0
            t0 = time.time()
            while not TEST or i < 10:
                if i + batchsize >= len(X):
                    assert i < len(X)
                    xi = np.asarray(X[-batchsize:])
                    done = True
                else:
                    xi = np.asarray(X[i:i+batchsize])
                    done = False
                t1 = time.time()
                feature_batch = slm.process_batch(xi)
                if TEST:
                    print('compute: ', time.time() - t1)
                t2 = time.time()
                delta = max(0, i + batchsize - len(X))
                assert np.all(np.isfinite(feature_batch))
                features_fp[i:i+batchsize-delta] = feature_batch[delta:]
                if TEST:
                    print('write: ', time.time() - t2)
                if done:
                    break

                i += batchsize
                if (i // batchsize) % 50 == 0:
                    t_cur = time.time() - t0
                    t_per_image = (time.time() - t0) / i
                    t_tot = t_per_image * X.shape[0]
                    if tlimit is not None and t_tot / 60.0 > tlimit:
                        raise TooLongException(t_tot/60.0, tlimit)
                    print 'get_features_fp: %i / %i  mins: %.2f / %.2f ' % (
                            i , len(X),
                            t_cur / 60.0, t_tot / 60.0)
            # -- docs hers:
            #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
            #    say that deletion is the way to flush changes !?
            if memmap:
                del features_fp
                self.filenames.append(filename)
                features_fp = np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=feature_shp)
                self.features.append(features_fp)
            else:
                self.filenames.append('')
                self.features.append(features_fp)

    def __enter__(self):
        return self.features

    def __exit__(self, *args):
        for filename in self.filenames:
            if filename:
                os.remove(filename)


class TheanoExtractedFeatures(ExtractedFeatures):
    def __init__(self, X, batchsize, configs, filenames, tlimit=DEFAULT_TLIMIT):
        slms = [slm_from_config(config, X.shape, batchsize) for config in configs]
        feature_shps = [(X.shape[0],) + slm.pythor_out_shape for slm in slms]
        super(TheanoExtractedFeatures, self).__init__(X, feature_shps,
                batchsize,
                slms,
                filenames,
                tlimit=tlimit)


class FeatureExtractor(object):
    def __init__(self, X, slm,
            tlimit=float('inf'),
            batchsize=4,
            filename=None,
            indices=None,
            verbose=False):
        """
        X - 4-tensor of images
        feature_shp - 4-tensor of output feature shape (len matches X)
        batchsize - number of features to extract in parallel
        slm - feature-extraction module (with .process_batch() fn)
        filename - store features to memmap here

        returns - read-only memmap of features
        """
        if filename is None:
            rand_id = random_id()
            filename = 'FeatureExtractor_' + rand_id + '.npy'
        self.filename = filename
        self.batchsize = batchsize
        self.tlimit = tlimit
        self.X = X
        self.slm = slm
        self.verbose = verbose
        if indices is None:
            indices = range(len(X))
        self.indices = indices
        self.n_to_extract = len(self.indices)
        if TEST:
            print('FeatureExtractor running in TESTING mode')
            self.verbose = True
            self.n_to_extract = min(TEST_NUM, self.n_to_extract)
        assert self.n_to_extract <= len(self.indices)

        # -- convenience
        self.feature_shp = (len(self.indices),) + self.slm.pythor_out_shape

    def __enter__(self):
        if self.filename:
            self.features = self.compute_features(use_memmap=True)
        else:
            self.features = self.compute_features(use_memmap=False)
        return self.features

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)
        del self.features

    def extract_to_memmap(self):
        """
        Allocate a memmap, fill it with extracted features, return r/o view.
        """
        filename = self.filename
        feature_shp = self.feature_shp
        print('Creating memmap %s for features of shape %s' % (
                                              filename,
                                              str(feature_shp)))
        features_fp = np.memmap(filename,
            dtype='float32',
            mode='w+',
            shape=feature_shp)
        info = open(filename+'.info', 'w')
        cPickle.dump(('float32', feature_shp), info)
        del info

        self.extract_to_storage(features_fp)

        # -- docs here:
        #    http://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        #    say that deletion is the way to flush changes !?
        del features_fp
        rval = np.memmap(self.filename,
            dtype='float32',
            mode='r',
            shape=feature_shp)
        return rval

    def extract_to_storage(self, arr):
        """
        Fill arr with the first len(arr) features of self.X.
        """
        assert len(arr) <= len(self.indices)
        batchsize = self.batchsize
        tlimit = self.tlimit
        print('Total size: %i bytes (%.2f GB)' % (
            arr.size * arr.dtype.itemsize,
            arr.size * arr.dtype.itemsize / float(1e9)))
        i = 0
        t0 = time.time()
        print('about to extract %d features' % self.n_to_extract)
        while True:
            if i + batchsize >= len(self.indices):
                assert i < len(self.indices)
                inds = self.indices[-batchsize:]
                done = True
            else:
                inds = self.indices[i:i+batchsize]
                if i + batchsize < self.n_to_extract:
                    done = False
                else:
                    done = True
            t_load = time.time()
            xi = np.asarray(self.X[inds])
            if self.verbose:
                print('load: ', time.time() - t_load)
            t1 = time.time()
            feature_batch = self.slm.process_batch(xi)
            if self.verbose:
                print('compute: ', time.time() - t1)
            t2 = time.time()
            delta = max(0, i + batchsize - len(arr))
            assert np.all(np.isfinite(feature_batch))
            arr[i:i + batchsize - delta] = feature_batch[delta:]
            if self.verbose:
                print('write: ', time.time() - t2)
            if done:
                break

            i += batchsize
            if (i // batchsize) % 50 == 0:
                t_cur = time.time() - t0
                t_per_image = (time.time() - t0) / i
                t_tot = t_per_image * len(arr)
                if tlimit is not None and t_tot / 60.0 > tlimit:
                    raise TooLongException(t_tot/60.0, tlimit)
                print 'extraction: %i / %i  mins: %.2f / %.2f ' % (
                        i , len(arr),
                        t_cur / 60.0, t_tot / 60.0)

    def compute_features(self, use_memmap=None):
        if use_memmap is None:
            size = np.prod(self.feature_shp) * 4
            use_memmap = (size > 3e8)  # 300MB cutoff

        if use_memmap:
            return self.extract_to_memmap()
        else:
            print('Using memory for features of shape %s' % str(self.feature_shp))
            arr = np.empty(self.feature_shp, dtype='float32')
            self.extract_to_storage(arr)
            return arr


class ThoreanoFeatureExtractor(FeatureExtractor):
    def __init__(self, X, config,
            tlimit=float('inf'),
            batchsize=4,
            filename=None,
            indices=None,
            verbose=False):
        self.config = config
        if indices is None:
            indices = range(len(X))
        relevant_shape = (len(indices),) + X.shape[1:]
        slm = slm_from_config(config, relevant_shape, batchsize=batchsize)
        super(ThoreanoFeatureExtractor, self).__init__(X, slm,
                tlimit=tlimit,
                batchsize=batchsize,
                filename=filename,
                indices=indices,
                verbose=verbose)


def slm_from_config(config, X_shape, batchsize):
    return slm_from_config_base(config, X_shape[1:], batchsize)


def slm_from_config_base(config, M_shape, batchsize):
    config = son_to_py(config)
    desc = copy.deepcopy(config['desc'])
    interpret_model(desc)
    if len(M_shape) == 2:
        t_slm = TheanoSLM(
                in_shape=(batchsize,) + M_shape + (1,),
                description=desc)
    elif len(M_shape) == 3:
        t_slm = TheanoSLM(
                in_shape=(batchsize,) + M_shape,
                description=desc)
    else:
        raise NotImplementedError()
    slm = t_slm
    # -- pre-compile the function to not mess up timing later
    slm.get_theano_fn()
    return slm


def son_to_py(son):
    """ turn son keys (unicode) into str
    """
    if isinstance(son, SON):
        return dict([(str(k), son_to_py(v)) for k, v in son.items()])
    elif isinstance(son, list):
        return [son_to_py(s) for s in son]
    elif isinstance(son, basestring):
        return str(son)
    else:
        return son


def use_memmap(size):
    if size < 3e8:
        memmap = False
    else:
        memmap = True
    return memmap


class TooLongException(Exception):
    """model takes too long to evaluate"""
    def msg(tot, cutoff):
        return 'Would take too long to execute model (%f mins, but cutoff is %s mins)' % (tot, cutoff)


def interpret_model(desc):
    for layer in desc:
        for (ind,(opname,opparams)) in enumerate(layer):
            if opname not in ['fbcorr', 'lpool', 'lnorm', 'rescale', 'activ']:
                raise NotImplementedError('Op %s not implemented in this branch' % opname)

