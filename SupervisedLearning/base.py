from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from joblib import Parallel, delayed
from abc import ABCMeta, abstractclassmethod
def Flat(l):
	return l.reshape(l.shape[0], -1)
class SupervisedLearningPipeline(Pipeline, metaclass=ABCMeta):
	r"""
	Metaclass of pipelines.

    From sklearn.pipeline.Pipeline
    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None`

    Parameters
    ----------
    From sklearn.pipeline.Pipeline
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    verbose : boolean, optional
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    plot_method : None or function
        If it is not None, there would be a method to plot the results from a single embedding,
        which is supposed to be defined
    flat_method : Flattening methods which is being use
    kfold : Folding method from sklearn.model_selection

    Attributes
    ----------
    plot : None
    	Plot the graph of one-time testing if it is not None.
    train_test : double
    	pass
    cross_val_score : tuple, double and list of double
    	pass
    """
	def __init__(self, 
				steps, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=4),
				flat_method = Flat
				):
		super(SupervisedLearningPipeline, self).__init__(steps, memory, verbose=verbose)
		self.flat_method = flat_method
		if plot_method is not None:
			self.plot_method = plot_method
		if kfold is not None:
			self.kfold = kfold

	def __getitem__(self, ind):
		"""Returns a sub-pipeline or a single esimtator in the pipeline
        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
		if isinstance(ind, slice):
			if ind.step not in (1, None):
				raise ValueError('Pipeline slicing only supports a step of 1')
			if ind.start in (0, None):
				return self.__class__(self.steps[ind])
			else:
				return Pipeline(self.steps[ind])
		try:
			name, est = self.steps[ind]
		except TypeError:
			# Not an int, try get step by name
			return self.named_steps[ind]
		return est
	def plot(self, X, y=None):
		return self.plot_method(self[-1], self[:-1].transform(X), y)
	@property
	def train_test(self, dataset_train, dataset_test, labels_train, labels_test):
		pass
	@property
	def cross_val_score(self, dataset, labels, groups=None):
		pass
