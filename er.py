#####################################################################################
# calculation of the EigenRank profile given the coordinates of a protein structure #
#####################################################################################
import numpy as np
from numba import jit
from scipy import spatial
from scipy import stats

# @jit(nopython=True)
def lrloop(dm):
    n = dm.shape[0]
    leaderranks = np.zeros((11,n), dtype=np.single)
    for a in range(5, 16):
        adjmat = np.greater_equal(a, dm)
        np.fill_diagonal(adjmat, 0)
        eg2 = np.ones((n+1,1), dtype=np.single)
        eg2[n] = 0  # ground node
        eg1 = np.vstack((adjmat, np.ones((1,n), dtype=np.single)))
        eg1 = np.hstack((eg1, eg2))
        eg1 = (eg1.T/np.sum(eg1, axis=0))
        # M = np.zeros_like(eg2)
        error, error_threshold = 10000, 2e-05
        while error > error_threshold:
        # while not np.all(np.less_equal(np.abs(eg2-M), 1e-08 + 0.00005 * np.abs(M))):
            M = eg2
            eg2 = eg1.dot(eg2)
            error = np.sum(np.divide(np.abs(eg2-M),M))/(n+1)
        ground = eg2[n]/n
        leaderranks[a-5,] = np.delete(eg2,-1)+ground
    return leaderranks.T

def eigenrank(atom_coordinates, numba=1):
    ''' calculating distance matrix -> adjacency matrices for different distance cutoffs -> PCA, scaling -> EigenRank '''
    if atom_coordinates.shape[1] != 3:
        atom_coordinates = atom_coordinates.T

    def pca_correcting(leaderranks):
        ''' PCA, checking maximal value for 8 angstrom cutoff and correcting principal components sign accordingly '''
        lrcenter = leaderranks - np.mean(leaderranks, axis=0)
        cov_matrix = np.dot(lrcenter.T,lrcenter)/lrcenter.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        ind = np.argsort(-eigenvalues)# [::-1] -> minus instead
        princo = eigenvectors[:,ind][:,0] # principal component
        princo = np.dot(princo.T, lrcenter.T)
        # fixing random sign of PCA
        i = np.argmax(lrcenter[:,4])
        if not np.array_equal(np.greater(lrcenter[i,4], 0), np.greater(princo[i], 0)):
            princo = -princo
        return stats.zscore(princo, axis=0)

    distance_matrix = spatial.distance_matrix(atom_coordinates, atom_coordinates, p=2)
    leaderranks = lrloop(distance_matrix)
    return pca_correcting(leaderranks)

def add_eigenrank(structure, numba=1):
    er_dict = {}
    for chain_id, coords in structure.coordinates.items():
        er_dict[chain_id] = eigenrank(coords)
    structure.er = er_dict
    return structure
