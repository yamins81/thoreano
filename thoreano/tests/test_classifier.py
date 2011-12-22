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
        self.labels = labels = np.arange(0,1000,.1) + \
                 .001*np.dot((np.random.random((10000,)) - .5),
                             np.arange(0,1000,.1))
        features1 = 2*labels + .1*(np.random.random((len(labels),)) - .5)
        features2 = -3*labels + .4*(np.random.random((len(labels),)) - .5)
        self.features = features = np.column_stack([features1, features2])
        self.train_Xy = (features[:100],labels[:100])
        self.test_Xy = (features[9000:], labels[9000:])

    def test_linear_regression(self):
        model, result = classifier.train_scikits(self.train_Xy,
                                          self.test_Xy,
                                          'linear_model.LinearRegression',
                                          regression=True,
                                          normalization=False)
        assert result['test_rsquared'] > .99, result['test_rsquared']

    def test_lassoCV(self):
        model, result = classifier.train_scikits(self.train_Xy,
                                          self.test_Xy,
                                          'linear_model.LassoCV',
                                          regression=True,
                                          normalization=False)
        assert result['test_rsquared'] > .99, result['test_rsquared']
        assert np.abs(model.intercept_) < .01, model.intercept_
        v = np.array([0, -0.33])
        c = model.coef_
        diff = np.abs(c - v)
        assert (diff < .005).all(), np.abs(c - v)
