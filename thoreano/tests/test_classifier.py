"""
Tests of thoreanoclassifier
"""

__authors__   = "Dan Yamins"
__copyright__ = "Dan Yamins"
__license__   = "3-clause BSD License"

import unittest

import numpy as np
import thoreano.classifier as classifier


class TestScikitsRegression(unittest.TestCase):
    def setUp(self):
        self.labels = np.arange(0,1000,.1) + \
                 .001*np.dot((np.random.random((10000,)) - .5),
                             np.arange(0,1000,.1))

    def test_basic(self):
        labels = self.labels
        features1 = 2*labels + .1*(np.random.random((len(labels),)) - .5)
        features2 = -3*labels + .4*(np.random.random((len(labels),)) - .5)
        features = np.column_stack([features1, features2])
        train_Xy = (features[:100],labels[:100])
        test_Xy = (features[9000:], labels[9000:])
        model, result = classifier.train_scikits(train_Xy,
                                          test_Xy,
                                          'linear_model.LassoCV',
                                          regression=True,
                                          model_kwargs={'fit_intercept':False},
                                          normalization=False)
        assert result['test_rsquared'] > .99, result['test_rsquared']
        v = np.array([0, -0.33])
        c = model.coef_
        diff = np.abs(c - v)
        assert (diff < .005).all(), np.abs(c - v)


