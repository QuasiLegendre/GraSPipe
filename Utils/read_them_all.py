from os import listdir
from os.path import isfile, join
from .text2matrix import Text2Matrix
from networkx.readwrite.gpickle import read_gpickle
from networkx.convert_matrix import to_numpy_matrix
from numpy import asarray
def ReadThemAllOld(file_path):
    """
    Read all of the ssv files in file_path, and transforms them into adjacency matrices.
    Parameters
    ----------
    file_path : string
       The directory path of the ssv files.
    Returns
    ----------
        params : Adjacency matrices dictionary labeled by individuals' SUBID
    """
    matrices_dict = {}
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #print(files)
    for file in files:
        person = int(file.split('_')[0])
        matrices_dict[person] = {}
    for file in files:
        person, num = file.split('.')[0].split('_')
        person = int(person)
        num = int(num)
        matrices_dict[person][num] = Text2Matrix(file_path+file)
    for person in matrices_dict.keys():
        gl = []
        keys = sorted(matrices_dict[person].keys())
        for key in keys:
            gl.append(matrices_dict[person][key])
        matrices_dict[person] = gl
    return matrices_dict
def ReadThemAll(file_path):
    """
    Read all of the ssv files in file_path, and transforms them into adjacency matrices.
    Parameters
    ----------
    file_path : string
       The directory path of the ssv files.
    Returns
    ----------
        params : Adjacency matrices dictionary labeled by session numbers.
    """
    matrices_dict = {}
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #print(files)
    for file in files:
        num = int(file.split('.')[0].split('_')[1])
        matrices_dict[num] = {}
    for file in files:
        person, num = file.split('.')[0].split('_')
        person = int(person)
        num = int(num)
        matrices_dict[num][person] = Text2Matrix(file_path+file)
    for num in matrices_dict.keys():
        gl = []
        keys = sorted(matrices_dict[num].keys())
        for key in keys:
            gl.append(matrices_dict[num][key])
        matrices_dict[num] = gl
    return matrices_dict
def ReadGpickle(file_path):
    r"""
    Read all of the gpickle files in file_path, and transforms them into adjacency matrices.
    Parameters
    ----------
    file_path : string
       The directory path of the gpickle files.
    Returns
    ----------
        params : Adjacency matrices dictionary labeled by individuals' SUBID.
    Notes
    ----------
        The version of networkx should be 1.9.0 when participate this function in the graph dataset in https://neurodata.io/mri-cloud .
    """
    matrices_dict = {}
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    #print(files)
    files = sorted(files)
    for file in files:
        num = int(file.split('.')[0].split('_')[1])
        matrices_dict[num] = {}
    for file in files:
        person, num = file.split('.')[0].split('_')
        person = int(person)
        num = int(num)
        matrices_dict[num][person] = asarray(to_numpy_matrix(read_gpickle(file_path+file)))
    for num in matrices_dict.keys():
        gl = []
        keys = sorted(matrices_dict[num].keys())
        for key in keys:
            gl.append(matrices_dict[num][key])
        matrices_dict[num] = gl
    return matrices_dict
if __name__ == '__main__':
    path = '../Graphs/JHU/'
    d = ReadThemAll(path)
    print(d[25427][0][0])