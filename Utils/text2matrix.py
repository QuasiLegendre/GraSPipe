import numpy as np


def Text2Matrix(file_name, undirected=True):
    """Get a single adjacency matrix from ssv file.
    Parameters
    ----------
    file_name : string
        The path of a ssv file.
    undirected : bool
        Whether the graph is directed or undirected.
    Returns
    -------
    params : numpy.array
        The graph adjacency matrix of the ssv file.
    """
    text_file = open(file_name, 'r')
    #graph_list = list(map(lambda x: tuple(map(lambda y: int(float(y)), x.split('\n')[0].split(' '))), text_file.readlines()))
    graph_list = list(map(lambda x: tuple(*map(lambda y: [int(y[0]), int(y[1]), float(y[2])], [x.split('\n')[0].split(' ')])), text_file.readlines()))
    text_file.close()
    if undirected:
        xs = []
        ys = []
        for (x, y, _) in graph_list:
            xs.append(x)
            ys.append(y)
        size_of_matrix = max(max(xs), max(ys))
        mat = np.zeros([size_of_matrix, size_of_matrix])
        for (x, y, weight) in graph_list:
            mat[x-1][y-1] = weight
            mat[y-1][x-1] = weight
    else:
        xs = []
        ys = []
        for (x, y, _) in graph_list:
            xs.append(x)
            ys.append(y)
        size_of_matrix = max(max(xs), max(ys))
        mat = np.zeros([size_of_matrix, size_of_matrix])
        for (x, y, weight) in graph_list:
            mat[x-1][y-1] = weight
    return mat
