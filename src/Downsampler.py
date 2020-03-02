from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample


class Downsampler(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Constructor
        """
        super(Downsampler, self).__init__()
    
    def fit(self, X, y, **kwargs):
        """

        method used to fit the step and learn by examples
        :param X: features - Dataframe
        :param y: target - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer -Transformer
        """
        return self

    def transform(self, X, y, **kwargs):
        """
        method that is used to transform according to what happened in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        if len(y.unique()) != 2:
            raise ValueError("y must be a pandas series with two possible values")
        if y.mode().size() != 1:
            


        

    def fit_transform(self, X, y=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        self = self.fit(X, y)
        return self.transform(X, y)