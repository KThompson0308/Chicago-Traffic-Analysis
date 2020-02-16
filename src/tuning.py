#################
# Code for Purged K-Fold Cross Validation based on Marcos Lopez de Prado(2018)
#################
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
import sklearn.model_selection
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import rv_continuous


class PurgedKFold(KFold):
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pandas series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo
    
    def split(self, X, y=None, groups=None):
        if (X.index==self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0]*self.pctEmbargo)
        test_starts = [(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i,j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            train_indices = np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            import pdb; pdb.set_trace()
            yield train_indices, test_indices


def clfHyperFit(feat, tgt, t1, pipe_clf, param_grid, n_splits, rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    if set(tgt.values) == {0,1}:
        scoring='f1'
    else:
        scoring='neg_log_loss'
    inner_cv = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pctEmbargo)
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


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    categoricals = ['CRASH_DATE_EST_I', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION',
                'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE',
                'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ROADWAY_SURFACE_COND',
                'ROAD_DEFECT', 'REPORT_TYPE', 'CRASH_TYPE', 'INTERSECTION_RELATED_I',
                'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE',
                'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION', 'STREET_NAME', 'PHOTOS_TAKEN_I',
                'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I',
                'MOST_SEVERE_INJURY', 'BEAT_OF_OCCURRENCE']
    dtypes = dict.fromkeys(categoricals, 'category')
    crashes = pd.read_csv("../data/TrafficCrashesChicago.csv", parse_dates = ['CRASH_DATE',
                                                                                    'DATE_POLICE_NOTIFIED'],dtype=dtypes)
    weather = pd.read_csv("../data/ChicagoWeather.csv", usecols=['dt_iso',
                                                                       'weather_main',
                                                                       'weather_description'],
                     dtype={'weather_main': 'category', 'weather_description': 'category'})
    # Merge and Filter Dataset to relevant and processible time horizon
    crashes = crashes[crashes.CRASH_DATE > '2018-01-01 00:00:00']
    crashes['hourly'] = crashes['CRASH_DATE'].dt.round('H')
    weather['dt_iso'] = pd.to_datetime(weather['dt_iso'], format="%Y-%m-%d %H:%M:%S +0000 UTC")
    crashes_merged = pd.merge(crashes, weather, how='left', left_on='hourly', right_on = 'dt_iso')
    # Replace unknown values with explicit missing values
    crashes_merged = crashes_merged.replace('UNKNOWN', np.nan)
    # Create Target Variable
    def are_there_injuries(total_injuries):
        if (total_injuries != np.nan):
            return True if total_injuries > 0 else False
        else:
            return np.nan

    crashes_merged['injuries'] = crashes_merged['INJURIES_TOTAL'].apply(lambda x: are_there_injuries(x))
    inactive_variables = ['RD_NO', 'CRASH_DATE_EST_I', 'LANE_CNT', 'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'PHOTOS_TAKEN_I',
                      'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'hourly',
                      'dt_iso', 'DATE_POLICE_NOTIFIED', 'LONGITUDE', 'LATITUDE', 'LOCATION']
    # Some variables are unusable because 95% + are missing, so we will drop them.
    final_set = crashes_merged.drop(inactive_variables, axis=1)
    final_set = final_set.dropna()
    final_set = final_set.set_index('CRASH_DATE')
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    injuries = final_set[['injuries']]
    nums = final_set.select_dtypes(['float64', 'int64'])
    onehotencoded = pd.get_dummies(final_set.select_dtypes('category'))
    final_set = pd.concat([injuries, nums, onehotencoded], axis=1)
    y = final_set['injuries']
    X = final_set.drop(['injuries'], axis=1)

    import matplotlib.pyplot as plt
    from multiprocessing import cpu_count

    # Randomized Grid Search with Purged K-Fold Cross Validation
    cores = cpu_count() - 1
    idx = X.index.to_series()

    grid = logUniform(0.01,5).rvs(size=1000)
    
    clfHyperFit(X, y, t1=idx, pipe_clf=LogisticRegression(), param_grid={'C':grid}, n_splits=3, rndSearchIter=1000, n_jobs = cores, pctEmbargo=0.2) 
