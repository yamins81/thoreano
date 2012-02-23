"""
These tests try to ensure that the thoreano's implementations are
consistent with pythor3.
"""
import time
import unittest

import numpy as np

from pythor3.model import SequentialLayeredModel

# required for fuzz-testing
import pyll

try:
    from scipy.misc import lena
except ImportError:
    from scipy import lena

from thoreano.slm import TheanoSLM, InvalidDescription, SLMFunction

def test_foo():
    import pythor3
    print pythor3.plugin_library['model.slm']


def match_single(desc, downsample=4):

    arr_in = (1.0 * lena())[::downsample, ::downsample]
    try:
        pythor_model = SequentialLayeredModel(arr_in.shape, desc)
        pythor_out = pythor_model.process(arr_in)
        if pythor_out.size == 0:
            pythor_accepted = False
        else:
            pythor_accepted = True
    except ValueError:
        pythor_accepted = False


    try:
        theano_model = TheanoSLM(arr_in.shape, desc)
        theano_accepted = True
    except InvalidDescription:
        theano_accepted = False

    if pythor_accepted and not theano_accepted:
        print 'PYTHOR_OUT shape', pythor_out.shape
    assert theano_accepted == pythor_accepted

    if not theano_accepted:
        raise InvalidDescription()

    theano_out = theano_model.process(arr_in)

    #fbcorr leaves in color channel of size 1
    if pythor_out.ndim == 3 and pythor_out.shape[2] == 1:
        pythor_out = pythor_out[:,:,0]
    assert theano_out.shape == pythor_out.shape, (
            theano_out.shape, pythor_out.shape)

    absdiff = abs(theano_out - pythor_out)
    absdiffmax = absdiff.max()

    if absdiffmax > .001:
        print 'theano_out', theano_out
        print 'pythor_out', pythor_out
        #
        assert 0, ('too much error: %s' % absdiffmax)


def match_single_color(desc):
    raise NotImplementedError()


def match_batch_color(desc):
    raise NotImplementedError()


def test_basic_lnorm():
    match_single(
            desc=[[('lnorm', {'kwargs': {'inker_shape': (3, 3)}})]])


def test_basic_lpool():
    match_single(desc=[[('lpool', {'kwargs': {'ker_shape': (3, 3)}})]])
    match_single(desc=[[('lpool', {'kwargs': {'ker_shape': (5, 5)}})]])
    # XXX: order
    # XXX: stride


def test_basic_fbcorr_1():
    match_single(
            desc=[[('fbcorr', {'kwargs': {'min_out': 0},
                 'initialize': {
                     'n_filters': 1,
                     'filter_shape': (3, 3),
                     'generate': ('random:uniform', {'rseed': 42}),
                 },
                })]])


def test_basic_fbcorr_16():
    match_single(
            desc=[[('fbcorr', {'kwargs': {'min_out': 0},
                 'initialize': {
                     'n_filters': 16,
                     'filter_shape': (3, 3),
                     'generate': ('random:uniform', {'rseed': 42}),
                 },
                })]])


def test_full_layer():
    match_single(
        desc=[[('fbcorr', {'kwargs': {'min_out': 0},
                     'initialize': {
                         'n_filters': 2,
                         'filter_shape': (3, 3),
                         'generate': ('random:uniform', {'rseed': 42}),
                     },
                    }),
         ('lpool', {'kwargs': {'ker_shape': (3, 3)}}),
         ('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
        ]],
        downsample=4)


class L3Basic(unittest.TestCase):
    desc = [
        # -- Layer 0
        [('lnorm', {'kwargs': {'inker_shape': (3, 3)}})],

        # -- Layer 1
        [('fbcorr', {'kwargs': {'min_out': 0},
                     'initialize': {
                         'n_filters': 16,
                         'filter_shape': (3, 3),
                         'generate': ('random:uniform', {'rseed': 42}),
                     },
                    }),
         ('lpool', {'kwargs': {'ker_shape': (3, 3)}}),
         ('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
        ],

        # -- Layer 2
        [('fbcorr', {'kwargs': {'min_out': 0},
                     'initialize': {
                         'n_filters': 16,
                         'filter_shape': (3, 3),
                         'generate': ('random:uniform', {'rseed': 42}),
                     },
                    }),
         ('lpool', {'kwargs': {'ker_shape': (3, 3)}}),
         ('lnorm', {'kwargs': {'inker_shape': (3, 3)}}),
        ],
    ]

    def test_0(self):
        match_single(desc=self.desc[:1])

    def test_1(self):
        match_single(desc=self.desc[1:2], downsample=64)

    def test_0and1(self):
        match_single(desc=self.desc[:2])

    def test_all(self):
        match_single(desc=self.desc)


def test_slm_function_on_lfw():
    from skdata import lfw, larray

    imgs, labels = lfw.Aligned().img_classification_task()

    feat_fn = SLMFunction(L3Basic.desc, lfw.Aligned().img_shape)
    feats = larray.lmap(feat_fn, imgs)

    if 0:
        # this uses a lot of memory, save re-computation of features
        feats = larray.cache_memory(feats)
    else:
        feats = larray.cache_memmap(feats, name='lfw_test_foo')

    # nothing has been computed yet, we just set up the pipeline
    try:

        # this computes the 0'th features
        t0 = time.time()
        f0a = feats[0]
        compute_t = time.time() - t0

        # this uses the cached features if cache is in use
        t0 = time.time()
        f0b = feats[0]
        cache_t = time.time() - t0

        assert np.all(f0a == f0b), (f0a.sum(), f0b.sum())
        
        print 'compute', compute_t, 'cache', cache_t
        assert compute_t > 2 * cache_t, (compute_t, cache_t)

        if 0:
            # this would compute *all* the feature if we're caching
            feats.cache_populate(batchsize=8)
        
        if 0:
            # indexing triggers the computation of uncomputed elements
            # in the cache, or the computation of all elements when no
            # cache is used.
            feats[:]

    finally:
        if hasattr(feats, 'delete_files'):
            # if feats is a memmap, this cleans up.
            feats.delete_files()


def fuzz_verify_runs(desc, downsample=4):
    arr_in = (1.0 * lena())[::downsample, ::downsample]
    theano_model = TheanoSLM(arr_in.shape, desc)
    theano_out = theano_model.process(arr_in)


class ForInts(object):
    def test_1(self):
        self.forint(1)

    def test_2(self):
        self.forint(2)

    def test_3(self):
        self.forint(3)

    def test_many(self):
        for seed in range(100, 150):
            self.forint(seed)

class TestLNormFuzz(unittest.TestCase, ForInts):
    def forint(self, seed):
        size = pyll.scope.int(
                pyll.scope.quniform(1.01, 10, 1))
        lnorm = ('lnorm', {'kwargs':
                {'inker_shape' : (size, size),
                 'outker_shape' : (size, size),
                 'remove_mean' : pyll.scope.one_of(0, 1),
                 'stretch' : pyll.scope.uniform(0, 10),
                 'threshold' : pyll.scope.uniform(0, 10),
             }})
        config = pyll.stochastic.sample(
                    [[lnorm]],
                np.random.RandomState(seed))
        fuzz_verify_runs(config)

if 0:
    import cvpr_params
    def test_cvpr_many():
        passing_seeds = []
        seed = 1
        while len(passing_seeds) < 20:
            template = gd.gDist(
                    repr(cvpr_params.config).replace("'",'"'))
            config = template.sample(seed)
            try:
                match_single(config['desc'])
                passing_seeds.append(seed)
            except InvalidDescription:
                print "Skipping invalid description from seed", seed
            except Exception:
                print config['desc']
                raise
            seed += 1
        print 'passing seeds', passing_seeds

