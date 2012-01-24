"""
These tests try to ensure that the thoreano's implementations are consistent with pythor3.
"""
import unittest
from nose.tools import assert_equals

import scipy as sp
import numpy as np
import hyperopt.gdist as gd
from pythor3.model import SequentialLayeredModel

try:
    from scipy.misc import lena
except:
    from scipy import lena

from thoreano.slm import TheanoSLM, InvalidDescription

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
        #
        #
        assert 0, ('too much error: %s' % absdiffmax)


def match_single_color(desc):
    raise NotImplementedError()


def match_batch(desc, batchsize=16):
    """
    test that pythor and theano match in process_batch
    """
    raise NotImplementedError()
    if 0:
        batch_major = np.random.randn(*((16, 3,) + arr_in.shape)).astype('float32')
        batch_minor = batch_major.transpose(0, 2, 3, 1).copy()

        theano_model = TheanoSLM(batch_major.shape, desc)
        theano_out = theano_model.process_batch(batch_major)
        # This doesn't run
        pythor_out = pythor_model.process_batch(batch_minor)


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

