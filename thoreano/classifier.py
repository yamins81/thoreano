import numpy as np
import scipy as sp

import asgd  # use master branch from https://github.com/jaberg/asgd
from early_stopping import fit_w_early_stopping, EarlyStopping
from asgd import utils

from scikits.learn import svm as sklearn_svm
from scikits.learn import linear_model as sklearn_linear_model
from scikits.learn.linear_model.logistic import LogisticRegression


#############
#####sgd#####
#############

def train_asgd_classifier_normalize(trainXy, testXy, verbose=False, batchsize=10, trace_normalize=False):

    train_features, train_labels = trainXy
    test_features, test_labels = testXy
    train_features, test_features, train_mean, train_std, trace = normalize([train_features, test_features],
                                                              trace_normalize=trace_normalize)
    trainXy = (train_features, train_labels)
    testXy = (test_features, test_labels)
    model, es, result = train_asgd_classifier(trainXy, testXy, verbose=verbose, batchsize=batchsize,
                                       step_sizes=step_sizes)

    result['train_mean'] = train_mean
    result['train_std'] = train_std
    result['trace'] = trace
    return model, es, result


def train_asgd_classifier(train_Xy, test_Xy,
        verbose=False,
        batchsize=10,
        validation_interval=100):
    """
    batchsize = 10                 # unit: examples
    validation_interval = 100      # unit: batches
    """
    train_X, train_y = train_Xy
    test_X, test_y = test_Xy
    if train_X.ndim != 2: raise TypeError()
    if test_X.ndim != 2: raise TypeError()
    n_examples, n_features = train_X.shape
    labelset = set(train_y)
    labelset.update(test_y)

    if verbose:
        print 'training classifier: n_train=%i n_test=%i n_features=%i' % (
                n_examples, len(test_X), n_features)

    assert labelset == set(range(len(labelset))), labelset

    if labelset == set([0, 1]):
        labels = [1, 1]
        # BINARY CLASSIFICATION
        # -- change labels to -1, +1
        train_y = train_y * 2 - 1
        test_y = test_y * 2 - 1
        def model_fn():
            return asgd.naive_asgd.NaiveBinaryASGD(
                    n_features=n_features)

        if use_theano:
            print "train_classifier not using theano for binary classif."
    else:
        # MULTI-CLASS CLASSIFICATION
        labels = range(len(labelset))
        def model_fn(step_size0):
            return asgd.naive_asgd.NaiveMulticlassASGD(
                    n_features=n_features,
                    n_classes=len(labelset))

    # -- repeat training for several learning rates
    #    take model that was best on held-out data
    model, es, test_prediction, train_prediction = fit_w_early_stopping(
                model_fn(),
                es=EarlyStopping(warmup=50, max_iters=1000), # unit: validation intervals
                train_X=train_X,
                train_y=train_y,
                validation_X=test_X,
                validation_y=test_y,
                batchsize=batchsize,
                validation_interval=validation_interval,
                verbose=verbose
                )

    result = get_result(train_y, test_y, train_prediction, test_prediction, labels)
    return model, es, result


#############
###scikits###
#############

def train_scikits(train_Xy,
                  test_Xy,
                  model_type,
                  regression=False,
                  model_kwargs=None,
                  fit_kwargs=None,
                  normalization=True,
                  trace_normalize=False):

    model, train_data = train_only_scikits(train_Xy,
                  model_type,
                  regression=regression,
                  model_kwargs=model_kwargs,
                  fit_kwargs=fit_kwargs,
                  normalization=normalization,
                  trace_normalize=trace_normalize)

    model, train_result = evaluate(model,
            train_Xy,
            train_data,
            regression=regression,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='train')
            
    model, test_result = evaluate(model,
            test_Xy,
            train_data,
            regression=regression,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='test')
    train_result.update(test_result)
            
    return model, train_result


def train_only_scikits(train_Xy,
                  model_type,
                  regression=False,
                  model_kwargs=None,
                  fit_kwargs=None,
                  normalization=True,
                  trace_normalize=False):

    """

    """

    train_features, train_labels = train_Xy

    if not regression:
        labels = sp.unique(train_labels)
        label_to_id = dict([(k,v) for v, k in enumerate(labels)])
        train_ids = sp.array([label_to_id[i] for i in train_labels])
    else:
        train_ids = train_labels
        labels = None

    #do normalization
    if normalization:
        train_features, train_mean, train_std, trace = normalize([train_features],
                                                                 trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None
    model = train_scikits_core(train_features, train_ids, model_type, model_kwargs,
                              fit_kwargs)
    train_data = {'train_mean':train_mean, 'train_std': train_std, 'trace': trace, 'labels':labels}

    return model, train_data


def train_scikits_core(train_features,
                     train_labels,
                     model_type,
                     model_kwargs,
                     fit_kwargs
                     ):
    """ Classifier training using SVMs

    Input:
    train_features = training features (both positive and negative)
    train_labels = corresponding label vector
    svm_eps = eps of svm
    svm_C = C parameter of svm
    model_type = liblinear or libsvm"""
    if model_kwargs is None:
        model_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    if model_type == 'liblinear':
        cls = sklearn_svm.LinearSVC
    elif model_type == 'libSVM':
        cls = sklearn_svm.SVC
    elif model_type == 'LRL':
        cls = LogisticRegression
    elif model_type == 'MCC':
        cls = CorrelationClassifier
    elif model_type.startswith('svm.'):
        ct = model_type.split('.')[-1]
        cls = getattr(sklearn_svm,ct)
    elif model_type.startswith('linear_model.'):
        ct = model_type.split('.')[-1]
        cls = getattr(sklearn_linear_model,ct)
    else:
        raise ValueError('Model type %s not recognized' % model_type)

    clf = cls(**model_kwargs)
    clf.fit(train_features, train_labels, **fit_kwargs)
    return clf



############
#evaluation#
############

def evaluate(model,
            test_Xy,
            train_data,
            regression=False,
            normalization=True,
            trace_normalize=False,
            prefix=None):

    test_features, test_labels = test_Xy
    if normalization:
        test_features, train_mean, train_std, trace = normalize([test_features],
                                                                data=train_data,
                                                trace_normalize=trace_normalize)
    test_prediction = model.predict(test_features)
    if regression:
        result = regression_stats(test_labels,test_prediction, prefix=prefix)
    else:
        labels = train_data['labels']
        test_prediction = labels[test_prediction]
        result = get_test_result(test_labels, test_prediction, labels, prefix=prefix)
    result.update(train_data)
    return model, result


def evaluate_classifier_normalize(model, test_Xy, data, trace_normalize=False, verbose=False, batchsize=10):
    test_X, test_y = test_Xy
    test_X = normalize([test_X], data=data, trace_normalize=trace_normalize)
    return evaluate_batch_classifier(model, (test_X, test_y), batchsize=batchsize, verbose=verbose)


def evaluate_batch(model, test_Xy, labels,
        batchsize=10,
        verbose=0):

    test_X, test_y = test_Xy

    if set(test_y) == set([0, 1]):
        test_y = test_y * 2 - 1
    assert set(test_y) == set(labels)

    tpos = 0
    vpos = 0
    errs = []
    test_prediction = []
    while vpos < len(test_X):
        xi = test_X[vpos:vpos + batchsize]
        yi = test_y[vpos:vpos + batchsize]
        pi = model.predict(xi)
        test_prediction.extend(pi.tolist())
        assert np.all(np.isfinite(pi))
        errs.append((yi != pi).astype('float64'))
        vpos += batchsize
    test_prediction = np.array(test_prediction)

    vscore = np.mean(errs)
    # -- std dev appropriate for classification
    vscore_std = np.sqrt(vscore * (1.0 - vscore) / len(test_X))
    result = get_test_result(test_y, test_prediction, labels)
    return result





#########
##stats##
#########

def get_regression_result(train_actual, test_actual, train_predicted, test_predicted):
    test_results = regression_stats(test_actual, test_predicted)
    train_results = regression_stats(train_actual, train_predicted, prefix='train')
    test_results.update(train_results)
    return test_results


def regression_stats(actual, predicted, prefix='test'):
    return {prefix+'_rsquared' : utils.rsquared(actual, predicted)}


def get_result(train_labels, test_labels, train_prediction, test_prediction, labels):
    result = {'train_errors': (train_labels != train_prediction).tolist(),
     'test_errors': (test_labels != test_prediction).tolist(),
     'train_prediction': train_prediction.tolist(),
     'test_prediction' : test_prediction.tolist(),
     'labels' : labels,
     }
    stats = multiclass_stats(test_labels, test_prediction, train_labels, train_prediction, labels)
    result.update(stats)
    return result


def get_test_result(test_labels, test_prediction, labels, prefix='test'):
    result = {
     prefix + '_errors': (test_labels != test_prediction).tolist(),
     prefix + '_prediction' : test_prediction.tolist(),
     'labels': labels
     }
    stats = multiclass_test_stats(test_labels, test_prediction, labels, prefix=prefix)
    result.update(stats)
    return result


def multiclass_stats(test_actual, test_predicted, train_actual, train_predicted,labels):
    test_results = multiclass_test_stats(test_actual, test_predicted, labels)
    train_results = multiclass_test_stats(train_actual, train_predicted, labels, prefix='train')
    test_results.update(train_results)
    return test_results


def multiclass_test_stats(test_actual, test_predicted, labels, prefix='test'):
    accuracy, ap, auc = utils.multiclass_stats(test_actual, test_predicted, labels)
    return {prefix+'_accuracy' : accuracy,
            prefix+'_ap' : ap,
            prefix+'_auc' : auc}