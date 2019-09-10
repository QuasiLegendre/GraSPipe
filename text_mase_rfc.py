from SupervisedLearning import MASEPipeline
from sklearn.decomposition import PCA
from Utils import ReadThemAllOld
from sklearn.model_selection import GroupKFold, ParameterGrid
from csv import DictReader
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from collections import ChainMap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from pandas import read_csv
matplotlib.use('Qt5Agg')
def heatmap(data, row_labels, col_labels, x_label='x', y_label='y', dataset_name=None, ax=None, steps=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_label_position('top')
    if dataset_name is not None and steps is not None:
        ax.set_title(f'Heat map of dataset {dataset_name} with {"->".join(steps)}')
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
#Testing	
if __name__ == '__main__':
	import numpy as np
	path = './Graphs/JHU/'
	mat_groups = ReadThemAllOld(path)
	dataset = []
	groups = []
	keys = sorted(mat_groups.keys())
	for key in keys:
		for mat in mat_groups[key]:
			dataset.append(mat)
			groups.append(key)
	labels = read_csv('HNU1R.csv').sort_values(by=['SUBID', 'SESSION']).get('SEX').values.tolist()
	"""
	labels = []
	label_name = 'SEX'
	with open('HNU1.csv') as HNU1:
		reader = DictReader(HNU1)
		for row in reader:
			labels.append(int(row[label_name]))
	"""

	#labels = [labels[x] for x in range(0, len(labels), len(keys))] * len(keys)
	def plotSVC(clf, X, y):
		h = 10
		x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
							 np.arange(y_min, y_max, h))
		import matplotlib
		matplotlib.use('QT5Agg')
		import matplotlib.pyplot as plt
		# Plot the decision boundary. For that, we will assign a color to each
		# point in the mesh [x_min, x_max]x[y_min, y_max].
		plt.subplots(figsize=(10, 10))
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		# Put the result into a color plot
		Z = np.array(Z)
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
		# Plot also the training points
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
		plt.xlabel('Dimension 1')
		plt.ylabel('Dimension 2')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.show()
	grid = {'rfc__criterion':['gini', 'entropy'], 'rfc__n_estimators':list(range(10, 110, 50)),
			'rfc__max_depth':list(range(10, 110, 50)), 'rfc__n_jobs':[1], 'rfc__min_samples_split':[2, 10, 20], 'rfc__max_features':['auto', 'log2', None],'MASE__n_components':list(range(10, 30, 5))}
	grid = {'rfc__criterion':['gini'], 'rfc__n_estimators':list(range(10, 120, 10)),
			'rfc__max_depth':[23],
			'rfc__n_jobs':[1], 'rfc__min_samples_split':[23], 'rfc__max_features':[None],
			'MASE__n_components':list(range(2, 46, 2))}#, 'pca__n_components':list(range(43, 64, 1))}
	"""
	def knn_run(MASEP, params):
		results = {}
		MASEP.set_params(**params)
		cvs, _ = MASEP.cross_val_score(dataset, labels, groups)
		print(params)
		print(cvs)
		results[cvs] = params
		return results


	out = Parallel(n_jobs=-2)(
		delayed(knn_run)(MASEPipeline([('pca', PCA()), ('rfc', RandomForestClassifier())], kfold=GroupKFold(n_splits=4)),
						 params) for params in ParameterGrid(grid))
	out = dict(ChainMap(*out))
	print(max(out.keys()))
	print(out[max(out.keys())])
	"""

	def knn_run(MASEP, params):
		results = {}
		MASEP.set_params(**params)
		cvs, _ = MASEP.cross_val_score(dataset, labels, groups, vote=True)
		print(params)
		print(cvs)
		results[cvs] = params
		return results, (params, cvs)
	out = Parallel(n_jobs=-2)(delayed(knn_run)(MASEPipeline([('rfc', RandomForestClassifier())], kfold=GroupKFold(n_splits=10)), params) for params in ParameterGrid(grid))
	dicts = [k[1] for k in out]
	out = [k[0] for k in out]
	out = dict(ChainMap(*out))
	max_num = max(out.keys())
	print(max_num)
	print(out[max_num])
	K = out[max_num]
	print(K)
	Cm = K['rfc__n_estimators']
	MASENm = K['MASE__n_components']
	C = grid['rfc__n_estimators']
	MASEN = grid['MASE__n_components']
	out_simp = {(k[0]['rfc__n_estimators'], k[0]['MASE__n_components']):k[1] for k in dicts}
	print(out_simp.keys())
	Z = np.zeros([len(C), len(MASEN)])
	for x in range(len(C)):
		for y in range(len(MASEN)):
			Z[x][y] = out_simp[(C[x], MASEN[y])]
	print('-----------------')
	print(np.sum(Z)/(len(C)*len(MASEN)))
	print('-----------------')
	fig1, ax4 = plt.subplots()

	im1, cbar1 = heatmap(Z, C, MASEN, y_label='rfc__n_estimators', x_label='MASE__n_components', ax=ax4, dataset_name='JHU', steps=MASEPipeline([('rfc', RandomForestClassifier())]).get_steps_name(),
						 cmap="YlGn", cbarlabel="accuracy")
	texts1 = annotate_heatmap(im1, valfmt="{x:.4f}")


	#texts1 = annotate_heatmap(im1, valfmt="{x:.6f}")
	fig1.tight_layout()
	plt.show()

