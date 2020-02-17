#################
# Code for Purged K-Fold Cross Validation based on Marcos Lopez de Prado(2018)
#################
import numpy as np
import pandas as pd
from sklearn.model_selection import *
import sklearn.model_selection
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import rv_continuous
from sklearn.utils import resample


def downsample(y, majority_case, minority_case):
    filt1 = y==majority_case
    majority = y[filt1]
    filt2 = y==minority_case
    minority = y[filt2]
    downsampled_majority = resample(majority, replace=False, n_samples=len(minority))
    return np.concatenate([majority, minority])




class PurgedKFold(KFold):
    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)

        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    # noinspection PyPep8Naming
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None):

        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(index=[self.samples_info_sets[start_ix]], data=[self.samples_info_sets[end_ix-1]])
            train_times = ml_get_train_times(self.samples_info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


def clfHyperFit(feat, tgt, t1, pipe_clf, param_grid, n_splits, rndSearchIter=0, n_jobs=-1, pct_embargo=0, **fit_params):
    if set(tgt.values) == {0,1}:
        scoring='f1'
    else:
        scoring='neg_log_loss'
    inner_cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=pct_embargo)
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, \
                        n_jobs=n_jobs,iid=False)
    else:
        gs=RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, \
                              cv=inner_cv, n_jobs=n_jobs, iid=False, n_iter=rndSearchIter)
    gs=gs.fit(feat, tgt, **fit_params).best_estimator_
    return gs



class logUniform_gen(rv_continuous):
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)

def logUniform(a=1, b=np.exp(1)):
    return logUniform_gen(a=a, b=b, name='logUniform')



def cvScore(clf,X,y,scoring='neg_log_loss',t1=None,n_splits=None, cvGen=None, pctEmbargo=None):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')

    if cvGen is None:
        cvGen=PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pctEmbargo)
    score = []
    for train, test in cvGen.split(X=X):
        fit = clf.fit(X=X.iloc[train,:], y=y.iloc[train])
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob,labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test], pred)
        score.append(score_)
    return np.array(score)



