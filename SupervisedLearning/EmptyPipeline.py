import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from .base import SupervisedLearningPipeline
from joblib import Parallel, delayed
def Flat(l):
	return l.reshape(l.shape[0], -1)
class EmptyPipeline(SupervisedLearningPipeline):
	r"""
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
    embedding_method : None, sklearn.base.BaseEstimator or its subclasses
        If it is None, the steps will equal to learning_method. If it is a string matches
        regex r"^[Ee][Mm][Pp][Tt][Yy]\w*$" (E.x., 'EmPtyEMBeddiNg'), steps will add
        flattening as first transform. If it is BaseEstimator or its subclasses,
        steps will add both embedding_method and flattening method.
    flat_method : Flattening methods which is being use
    kfold : Folding method from sklearn.model_selection

    Attributes
    ----------
     named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    get_steps_name : list of string
        Read-only attribute to access step parameter lists.
    train_test : double
        The accuracy of training on one dataset(training dataset) and test on the other dataset(tesing dataset).
        It also supports group folding for testing. The voting method is also used in this method, which means that the
        detected label is decided through voting by the sessions.
    cross_val_score : tuple, double and list of double
        The average accuracy of kfolding and the list of the accuracies of kfolding. It also supports voting on sessions
        and GroupKFold.
    """
	def __init__(self, 
				learning_method, 
				memory=None, 
				verbose=False, 
				plot_method=None, 
				kfold = KFold(n_splits=4),
				flat_method = Flat
				):
		super(EmptyPipeline, self).__init__(steps=learning_method, memory=memory, verbose=verbose, plot_method=plot_method, kfold=kfold, flat_method=flat_method)
		if not isinstance(self.steps[0][1], FunctionTransformer):
			self.steps = [('Flat', FunctionTransformer(self.flat_method, validate=False))] + self.steps
		if plot_method is not None:
			self.plot_method = plot_method
		if kfold is not None:
			self.kfold = kfold
	def get_steps_name(self):
		"""Get list of step name strings. Read-only attribute to access step parameter lists.
        Parameters
        ----------
        Returns
        -------
        params : list of strings refering to the names of the steps
        """
		stp = ['Flat']
		stp.extend([x[1].__class__.__name__  for x in self.steps[1:]])
		return stp
	def train_test(self, dataset_train, dataset_test, labels_train, labels_test, group_train=None, group_test=None, vote=False):
		"""Get the accuracy of training on one dataset(training dataset) and test on the other dataset(tesing dataset).
        Also supporting group folding and the voting method.
        Parameters
        ----------
        dataset_train : iterable
            The training dataset for one-time evaluation. The type could be list of adjacency matrices, a 3D numpy array
            or others.
        dataset_test : iterable
            The testing dataset for one-time evaluation. The type could be list of adjacency matrices, a 3D numpy array
            or others.
        labels_train : iterable
            The training labels for one-time evaluation. The type should be list of labels(e.x., integer).
        labels_test : iterable
            The testing labels for one-time evaluation. The type should be list of labels(e.x., integer).
        group_train : None
            pass
        group_test : None or iterable
            Required group labels for testing in the voting method. If None, the voting cannot be done.
        vote ： bool
            Wheather the method of voting on sessions is performed. If true and group_test is not None, voting method
            will be performed.
        Returns
        -------
        params : float
            The accuracy of one-time evaluation.
        """
		self[-1].fit(dataset_train, labels_train)
		if vote and group_test is not None:
			res = self[-1].predict(dataset_test)
			results = {}
			for i in np.unique(group_test).tolist():
				results[i] = []
			for i in range(len(res)):
				results[group_test[i]].append(res[i])
			result = []
			for r in results.values():
				keys = np.unique(r).tolist()
				res_dict = {k:0 for k in keys}
				for i in r:
					res_dict[i] += 1
				max_k = keys[0]
				for k in keys:
					if res_dict[k] > res_dict[max_k]:
						max_k = k
				result.extend([max_k]*len(r))
			acurrate = 0
			for i in range(len(labels_test)):
				if result[i] == labels_test[i]:
					acurrate += 1
			return acurrate/len(result)
		else:
			return self[-1].score(dataset_test, labels_test)
	def cross_val_score(self, dataset, labels, groups=None, vote=False):
		"""Get the cross valuation scores using self.kfold. Returns the average and the score list.
        Parameters
        ----------
        dataset : iterable
            The dataset for cross evaluation. The type could be list of adjacency matrices, a 3D numpy array
            or others.
        labels : iterable
            The labels for cross evaluation. The type should be list of labels(e.x., integer).
        group : None or iterable
            Required group labels for group k-folding. If it is not None, group k-folding will be performed if type(self.kfold)
        group_test : None or iterable
            Required group labels for testing in the voting method. If None, the voting cannot be done.
        vote ： bool
            Wheather the method of voting on sessions is performed. If true and group_test is not None, voting method
            will be performed.
        Returns
        -------
        params : float
            The accuracy of one-time evaluation.
        """
		test_results = []
		dataset, labels, groups = np.array(dataset), np.array(labels), np.array(groups)
		dataset = self[:-1].fit_transform(dataset)
		if groups is None:
			test_results = Parallel(n_jobs=1)(delayed(self.train_test)(dataset[train_index],dataset[test_index], labels[train_index], labels[test_index], vote=vote) for train_index, test_index in self.kfold.split(dataset, labels))
		else:
			test_results = Parallel(n_jobs=1)(delayed(self.train_test)(dataset[train_index],dataset[test_index], labels[train_index], labels[test_index], vote=vote, group_test=groups[test_index]) for train_index, test_index in self.kfold.split(dataset, labels, groups))
		avg_score = sum(test_results)/len(test_results)
		return avg_score, test_results
