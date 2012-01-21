import numpy as np
import scipy as sp


#############
#####asgd#####
#############
try:
    import asgd  # use master branch from https://github.com/jaberg/asgd
except ImportError:
    print("Can't import asgd.")

def train_asgd(train_Xy,
               test_Xy,
               normalization=True,
               trace_normalize=False):
               
    
    model, train_data = train_only_asgd(train_Xy,
                                        normalization=normalization,
                                        trace_normalize=trace_normalize)
                                        
    model, train_result = evaluate(model,
            train_Xy,
            train_data,
            regression=False,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='train')
            
    model, test_result = evaluate(model,
            test_Xy,
            train_data,
            regression=False,
            normalization=normalization,
            trace_normalize=trace_normalize,
            prefix='test')
    train_result.update(test_result)
            
    return model, train_result


def train_only_asgd(train_Xy, 
                    normalization=True,
                    trace_normalize=False,
                    margin_biases=None):
    """
    """
    train_features, train_labels = train_Xy
    if train_features.ndim != 2: raise TypeError()
    n_examples, n_features = train_features.shape
    labelset = set(train_y)
    
    #do normalization
    if normalization:
        train_features, train_mean, train_std, trace = normalize(
                              [train_features], trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None
                                
    if labelset == set([-1, 1]):
        labels = [-1, 1]
        model = asgd.naive_asgd.NaiveBinaryASGD(n_features=n_features)
    else:
        # MULTI-CLASS CLASSIFICATION
        labels = range(len(labelset))
        assert labelset <= set(labels)
        model = asgd.naive_asgd.NaiveMulticlassASGD(n_features=n_features,
                                                    n_classes=len(labelset))
                              
    
    model.fit(train_X, train_y, margin_biases=margin_biases)
    train_data = {'train_mean':train_mean,
                  'train_std': train_std,
                  'trace': trace,
                  'labels': labels}

    return model, train_data



#############
###scikits###
#############
try:
    from scikits.learn import svm as sklearn_svm
    from scikits.learn import linear_model as sklearn_linear_model
    from scikits.learn.linear_model.logistic import LogisticRegression
except ImportError:
    print("Can't import scikits stuff")

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
        train_features, train_mean, train_std, trace = normalize(
                              [train_features], trace_normalize=trace_normalize)
    else:
        train_mean = None
        train_std = None
        trace = None
    model = train_scikits_core(train_features, train_ids, model_type, model_kwargs,
                              fit_kwargs)
    train_data = {'train_mean':train_mean,
                  'train_std': train_std,
                  'trace': trace,
                  'labels': labels}

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
            prefix=None,
            batchsize=None):

    test_features, test_labels = test_Xy
    if normalization:
        test_features, train_mean, train_std, trace = normalize(
              [test_features], data=train_data, trace_normalize=trace_normalize)
    
    if batchsize is None:
        test_prediction = model.predict(test_features)
    else:
        test_prediction = batch_prediction(model, test_features, batchsize)
    if regression:
        result = regression_stats(test_labels, test_prediction, prefix=prefix)
    else:
        labels = train_data['labels']
        test_prediction = labels[test_prediction]
        result = get_test_result(test_labels, test_prediction, labels, prefix=prefix)
    result.update(train_data)
    return model, result


def batch_prediction(model, test_X, batchsize):
    pos = 0
    errs = []
    test_prediction = []
    while pos < len(test_X):
        xi = test_X[pos:pos + batchsize]
        pi = model.predict(xi)
        test_prediction.extend(pi.tolist())
        assert np.all(np.isfinite(pi))
        pos += batchsize
    return np.array(test_prediction)



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
    accuracy, ap, auc = classification_stats(test_actual, test_predicted, labels)
    return {prefix+'_accuracy' : accuracy,
            prefix+'_ap' : ap,
            prefix+'_auc' : auc}


def classification_stats(actual, predicted, labels):
    accuracy = float(100*(predicted == actual).sum() / float(len(predicted)))
    aps = []
    aucs = []
    if len(labels) == 2:
        labels = labels[1:]
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    ap = np.array(aps).mean()
    auc = np.array(aucs).mean()
    return accuracy, ap, auc
    

def average_precision(actual, predicted, labels):
    if len(labels) == 2:
        labels = labels[1:]
    aps = []
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        ap = ap_from_prec_and_rec(prec, rec)
        aps.append(ap)
    ap = np.array(aps).mean()
    return ap
    

def ap_from_prec_and_rec(prec, rec):
    ap = 0
    rng = np.arange(0, 1.1, .1)
    for th in rng:
        parray = prec[rec>=th]
        if len(parray) == 0:
            p = 0
        else:
            p = parray.max()
        ap += p / rng.size
    return ap


def area_under_curve(actual, predicted, labels):
    if len(labels) == 2:
        labels = labels[1:]
    aucs = []
    for label in labels:
        prec, rec = precision_and_recall(actual, predicted, label)
        auc = auc_from_prec_and_rec(prec, rec)
        aucs.append(auc)
    auc = np.array(aucs).mean()
    return auc
    

def auc_from_prec_and_rec(prec, rec):
    #area under curve
    h = np.diff(rec)
    auc = np.sum(h * (prec[1:] + prec[:-1])) / 2.0
    return auc


def rsquared(actual, predicted):
    a_mean = actual.mean()
    num = np.linalg.norm(actual - predicted) ** 2
    denom = np.linalg.norm(actual - a_mean) ** 2
    return 1 -  num / denom


def precision_and_recall(actual, predicted, cls):
    c = (actual == cls)
    si = np.argsort(-c)
    tp = np.cumsum(np.single(predicted[si] == cls))
    fp = np.cumsum(np.single(predicted[si] != cls))
    rec = tp /np.sum(predicted == cls)
    prec = tp / (fp + tp)
    return prec, rec



#########
##utils##
#########

def normalize(feats, trace_normalize=False, data=None):
    """Performs normalizations before training on a list of feature array/label
    pairs. first feature array in list is taken by default to be training set
    and norms are computed relative to that one.
    """

    if data is None:
        train_f = feats[0]
        m = train_f.mean(axis=0)
        s = np.maximum(train_f.std(axis=0), 1e-8)
    else:
        m = data['train_mean']
        s = data['train_std']
    feats = [(f - m) / s for f in feats]
    if trace_normalize:
        if data is None:
            train_f = feats[0]
            tr = np.maximum(np.sqrt((train_f**2).sum(axis=1)).mean(), 1e-8)
        else:
            tr = data['trace']
    else:
        tr = None
    if trace_normalize:
        feats = [f / tr for f in feats]
    feats = tuple(feats)
    return feats + (m, s, tr)


def mean_and_std(X, min_std):
    # XXX: this loop is more memory efficient than numpy but not as
    # numerically accurate. It would be better to look at the train_mean,
    # and then either use the msq for getting unit norms if the train_means
    # are small-ish, or else use numpy.std if the mean is large enough to
    # cause numerical trouble
    m = np.zeros(X.shape[1], dtype='float64')
    msq = np.zeros(X.shape[1], dtype='float64')
    for i in xrange(X.shape[0]):
        alpha = 1.0 / (i + 1)
        v = X[i]
        m = (alpha * v) + (1 - alpha) * m
        msq = (alpha * v * v) + (1 - alpha) * msq

    train_mean = np.asarray(m, dtype=X.dtype)
    train_std = np.sqrt(np.maximum(
            msq - m * m,
            min_std ** 2)).astype(X.dtype)
    return train_mean, train_std


def split_center_normalize(X, y,
        validset_fraction=.2,
        validset_max_examples=5000,
        inplace=False,
        min_std=1e-4,
        batchsize=1):
    n_valid = int(min(
        validset_max_examples,
        validset_fraction * X.shape[0]))

    # -- increase n_valid to a multiple of batchsize
    while n_valid % batchsize:
        n_valid += 1

    n_train = X.shape[0] - n_valid

    # -- decrease n_train to a multiple of batchsize
    while n_train % batchsize:
        n_train -= 1

    if not inplace:
        X = X.copy()

    train_features = X[:n_train]
    valid_features = X[n_train:n_train + n_valid]
    train_labels = y[:n_train]
    valid_labels = y[n_train:n_train + n_valid]

    train_mean, train_std = mean_and_std(X, min_std=min_std)

    # train features and valid features are aliased to X
    X -= train_mean
    X /= train_std

    return ((train_features, train_labels),
            (valid_features, valid_labels),
            train_mean,
            train_std)


def simple_bracket_min(f, pt0, pt1):
    v0 = f(pt0)
    v1 = f(pt1)
    if v0 > v1:
        while v0 > v1:
            raise NotImplementedError()


