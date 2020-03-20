'''
EigenRank based protein structure comparison

cite:
Heinke, F., Hempel, L., and Labudde, D. (2019). A Novel Approach
for Fast Protein Structure Comparison and Heuristic Structure Database Searching
Based on Residue EigenRank Scores.
'''

import numpy as np
from numba import jit,njit
import pandas as pd
import os,sys,json,itertools
import tqdm
import pickle
from pathlib import Path
from time import time
from scipy import spatial
from scipy import stats
import scipy.special as sc
import multiprocessing as mp
import matplotlib.pyplot as plt
import cProfile
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
from tools import *
# print(fasta.read('/home/julius/Sync/cops/queries/fasta/c1f43A_.fasta'))
### GENERAL FUNCTIONS ###

def dmabs(a,b): # distance between elements of 2 sequences
    # a,b = a[:,0],b[:,0]
    m,n = len(a),len(b)
    result = np.empty((m,n),dtype=float)
    if m < n:
        for i in range(m):
            result[i,:] = np.abs(a[i]-b)
    else:
        for j in range(n):
            result[:,j] = np.abs(a-b[j])
    return result

# @jit(nopython=True) # no working scipy implementation possible
def dmnd(a,b): # distance between elements of 2 sequences
    m,n = len(a),len(b)
    result = np.zeros((m,n), dtype=np.single)
    if m < n:
        for i in range(m):
            result[i,:] = np.abs(sc.ndtr(a[i])-sc.ndtr(b))
    else:
        for j in range(n):
            result[:,j] = np.abs(sc.ndtr(a)-sc.ndtr(b[j]))
    return result



### PROTEIN CLASS ###
@jit(nopython=True)
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


class Protein:
    def __init__(self,file):
        self.file = file
        self.id = str(file)[-10:-4]
        self.readcoords(file)
        self.ER()

    def readcoords(self,file):
        atom_id = 0
        coords = []
        b = 0
        for line in open(file, 'r'):
            if line.startswith('ATOM'):
                if atom_id < int(line[6:11]):
                    if line[13:15] == 'CA':
                        b += 1
                        atom_id = int(line[6:11])
                        x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
                        coords.append([x,y,z])
            elif line.startswith('REMARK  99 ASTRAL SCOPe-sccs:'):
                self.sccs = line[30:].rstrip()
        self.coordinates = np.asarray(coords)

    def distance_matrix(self): # distance matrix of point coordinate sets
        self.n = self.coordinates.shape[0]
        self.dist = spatial.distance_matrix(self.coordinates, self.coordinates, p=2)

    def ER(self): # calculate EigenRank profile, including PCA and zscore
        self.distance_matrix()
        self.leaderranks = lrloop(self.dist)
        ### PCA ###
        lrcenter = self.leaderranks - np.mean(self.leaderranks, axis=0)
        cov_matrix = np.dot(lrcenter.T,lrcenter)/lrcenter.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        ind = np.argsort(-eigenvalues)# [::-1] -> minus instead
        # self.ev = eigenvalues[ind[0]]
        princo = eigenvectors[:,ind][:,0]
        princo = np.dot(princo.T, lrcenter.T)
        # fixing random sign of PCA
        i = np.argmax(lrcenter[:,4])
        if not np.array_equal(np.greater(lrcenter[i,4], 0), np.greater(princo[i], 0)):
            princo = -princo
        self.er = stats.zscore(princo, axis=0)
        # clearing unused variables/attributes:
        self.adjmat = None
        self.leaderranks = None
        # self.ev = None
        self.dist = None





### ALIGNMENT CLASS ###
### numba part:
@jit(nopython=True)
def nlocalalign(ab,a,b,gap,factor,limit):
    m,n = len(a),len(b)
    f = np.zeros((m+1,n+1))
    t = np.zeros((m+1,n+1))
    for i in range(1,m+1):
        prev = 0
        for j in range(1,n+1):
            down = f[i-1,j] - gap
            right = prev - gap
            diag = f[i-1,j-1] - factor*ab[i-1,j-1] + limit
            c = max([down,right,diag,0])
            prev,f[i,j] = c,c
            if diag == c:
                t[i,j] = 2
            elif right == c:
                t[i,j] = -1
            elif down == c:
                t[i,j] = 1
            else:
                t[i,j] = 0
    ## acquired f and t matrices, followed by traceback
    i,j = np.where(f==f.max())
    i,j = i[0],j[0]
    score = f[i,j]
    is_gap = []
    i_list = []
    j_list = []
    c = 0
    while t[i,j] != 0:
        is_gap.append(1)
        i_list.append(0)
        j_list.append(0)
        i_list[c] = i-1
        j_list[c] = j-1

        editing = t[i,j]
        if editing != 2:
            if editing == -1:
                j -= 1
            else:
                i -= 1
            is_gap[c] = 1
        else:
            is_gap[c] = 0
            i -= 1
            j -= 1
        c += 1
    return i_list,j_list,is_gap,score


class Alignment:
    '''pairwise alignment of two protein structures.
    query and target as Protein objects.'''
    def __init__(self, query_protein, target_protein, factor, limit):
        self.query = query_protein
        self.target = target_protein
        self.query_id = self.query.id
        self.target_id = self.target.id
        self.limit = limit
        self.factor = factor
        self.csa()

    def localalign(self,a,b,gapcost=1):
        ab = dmabs(a,b) # normal distribution (dmnd) or plain difference (dmabs)
        self.i_list, self.j_list, self.gap, self.score = nlocalalign(ab,a,b,gapcost,self.factor,self.limit)
        self.traceback_len = len(self.gap)

    def Kabsch(self): # get the local alignment, calculate optimal rotation matrix for structures to fit into each other
        self.localalign(self.query.er,self.target.er)

        def rmsd_(a,b): # a and b are vectors
            diff = np.array(a) - np.array(b)
            n = len(a)
            return np.sqrt((diff * diff).sum() / n)
            # return np.sqrt(((a-b)**2).sum() / n)
        a = self.query.coordinates
        b = self.target.coordinates
        i_list = [i for i,g in zip(self.i_list,self.gap) if g == 0]
        j_list = [j for j,g in zip(self.j_list,self.gap) if g == 0]
        self.len_wo_gaps = len(i_list)
        if self.len_wo_gaps < 1:
            self.rmsd = 0
            return np.zeros((2,3)),np.full((2,3),1000)
        else:
            a = a[i_list,:]
            b = b[j_list,:]
            # print(np.mean(a, axis=0), a.shape)
            self.query_centroid = np.mean(a, axis=0)
            self.target_centroid = np.mean(b, axis=0)
            a -= self.query_centroid
            b -= self.target_centroid
            h = a.T@b
            u,s,v = np.linalg.svd(h.T)
            d = np.linalg.det(v.T@u.T)
            r = v.T@np.diag([1,1,d])@u.T
            a = a@r
            self.rmsd = rmsd_(a,b)
            #self.rotation_matrix = r
            #self.structure_alignment = (a,b)
            return a,b

    def csa(self): # compute structure alignment and a set of scores
        a,b = self.Kabsch()
        dists = np.linalg.norm(a-b, axis=1)
        f1 = np.count_nonzero(np.where(dists < 1))
        f2 = np.count_nonzero(np.where(dists < 2))
        f4 = np.count_nonzero(np.where(dists < 4))
        f8 = np.count_nonzero(np.where(dists < 8))
        self.gdt_ts = 25 * sum([f1,f2,f4,f8])/self.len_wo_gaps if self.len_wo_gaps > 0 else 0
        self.gdt_sim = self.score * np.sqrt(self.len_wo_gaps*self.gdt_ts)


# t = Protein('/home/julius/Sync/perf_test/alignment/t_pdb_validated/c2ba0I1.pdb')
# q = Protein('/home/julius/Sync/perf_test/alignment/q_pdb_validated/c2nn6C_.pdb')
# r = Protein('/home/julius/Sync/perf_test/alignment/t_pdb_validated/c3d29W_.pdb')
# print(q.er)
# a = Alignment(q,t,1,0.7)
# # print(a.__dict__)
# print(len(a.i_list[::-1]))
# print(a.j_list[::-1])



### COPS ###
def doalign(q,t):
    align = Alignment(q,t,1,0.7)
    ad = align.__dict__
    include = ['query_id','target_id','score','traceback_len','len_wo_gaps','rmsd','gdt_ts','gdt_sim']
    # with open('cops_results2/cops_'+"{:.2f}".format(f)+'_'+"{:.2f}".format(l)+'.txt', 'a') as f:
    with open('val_cops_0.7.txt', 'a') as f:
        f.write(json.dumps({k: ad[k] for k in include})+'\n')
    return None

def cops():
    # query_dir = Path('/home/julius/Sync/perf_test/alignment/q_pdb_validated')
    # target_dir = Path('/home/julius/Sync/perf_test/alignment/t_pdb_validated')
    # query_files = query_dir.rglob('*.pdb')
    # target_files = target_dir.rglob('*.pdb')
    # t3 = time()
    # with mp.Pool(processes=8) as pool:
    #     query_proteins = pool.map(Protein, query_files)
    #     target_proteins = pool.map(Protein, target_files)
    # with open('proteins.pkl', 'wb') as f:
    #     pickle.dump(query_proteins, f)
    #     pickle.dump(target_proteins, f)
    # print('protein initialisation finished in: ', time()-t3)
    with open('proteins.pkl', 'rb') as f:
        query_proteins = pickle.load(f)
        target_proteins = pickle.load(f)

    # factors = np.linspace(2,24,12) # exploring parameter space
    # limits = np.linspace(0.5,5,10)
    # print(factors, limits)
    prod = itertools.product(query_proteins, target_proteins)

    # prod = itertools.product(query_proteins, target_proteins)
    with mp.Pool(processes=8) as pool:
        pool.starmap(doalign, prod)

start = time()
cops()
print(time()-start)
with open('timenumba.txt', 'a') as f:
    f.write(str(time()-start))
# os.system('shutdown')



def read_Lars():
    with open('proteins.pkl', 'rb') as f:
        query_proteins = pickle.load(f)
        target_proteins = pickle.load(f)

    query_dir = Path('/home/julius/Lars/new/ranks_queries_validated')
    target_dir = Path('/home/julius/Lars/new/ranks_database_validated')
    query_files = query_dir.glob('*.ranks')
    target_files = target_dir.glob('*.ranks')
    larsq = {str(f)[-16:-10] : stats.zscore(np.genfromtxt(f, skip_header=1, usecols=2)) for f in query_files}
    larst = {str(f)[-16:-10] : stats.zscore(np.genfromtxt(f, skip_header=1, usecols=2)) for f in target_files}
    # switch my ERs for Lars' in all proteins except the ones with different lengths:
    new_qprot = []
    new_tprot = []
    for p in query_proteins:
        if p.er.shape[0] == larsq[p.id].shape[0]:
            p.er = larsq[p.id]
            new_qprot.append(p)
    for p in target_proteins:
        if p.er.shape[0] == larst[p.id].shape[0]:
            p.er = larst[p.id]
            new_tprot.append(p)
    print(len(new_tprot), len(new_qprot))
    # prod = itertools.product(new_qprot, new_tprot)
    # with mp.Pool(processes=8) as pool:
    #     pool.starmap(larsalign, prod)

    # print(max([sum(abs(p.er-larsq[p.id]))/p.er.shape[0] for p in query_proteins if p.er.shape[0] == larsq[p.id].shape[0]]))
    print(([p.id for p in query_proteins if p.er.shape[0] != larsq[p.id].shape[0]]))
    print(([p.id for p in target_proteins if p.er.shape[0] != larst[p.id].shape[0]]))
    # print(stats.zscore(larsq['3bdmY_']), [p.er for p in query_proteins if p.id == '3bdmY_']) 
    print([p.coordinates.shape for p in query_proteins if p.id == '3bdmY_'])
    print(len( stats.zscore(larst['3d29W_']) ), len( [p.er for p in target_proteins if p.id == '3d29W_'][0] ))


def larsalign(q,t):
        align = Alignment(q,t,1,0.7)
        ad = align.__dict__
        include = ['query_id','target_id','score','traceback_len','len_wo_gaps','rmsd','gdt_ts','gdt_sim']
        with open('cops_lars_er_r_comparison.data', 'a') as f:
            f.write(json.dumps({k: ad[k] for k in include})+'\n')

# read_Lars()




### comparison between R and Python EigenRanks:
# p = Protein('/home/julius/Sync/perf_test/alignment/q_pdb_validated/c1am2A_.pdb')
# olr = np.genfromtxt('/home/julius/Sync/perf_test/R/lr.txt')
# oer = stats.zscore(np.genfromtxt('/home/julius/Sync/perf_test/R/er.txt'), axis=0)
# # print(p.er)
# # print(oer)
# print(np.allclose(olr, p.leaderranks.T, rtol=1e-03, atol=1e-04))
# print(np.allclose(p.er.T, oer, rtol=1e-02, atol=1e-03))

### GENERAL TESTING
# q = Protein('/home/julius/Sync/perf_test/alignment/q_pdb_validated/c2qdtA_.pdb')
# t = Protein('/home/julius/Sync/perf_test/alignment/t_pdb_validated/c2zo4A_.pdb')
# r = Protein('/home/julius/Sync/perf_test/alignment/q_pdb_validated/c1t36E7.pdb')
# dmnd(q.er, t.er)
# print(q.er, q.n)
# a = Alignment(q, t, 2, 1)
# b = Alignment(r, t)
# print(a.traceback_len, a.score, a.rmsd, a.gdt_ts)
# print(np.allclose(np.genfromtxt('perf_test/R/1.txt'), q.er.T, rtol=1e-02, atol=1e-03))
# plt.plot(np.genfromtxt('perf_test/R/2.txt'))
# plt.plot(stats.zscore(r.leaderranks[:,7]))
# plt.plot(r.er)
# plt.show()
# print(a.gdt_ts, a.rmsd, a.score)
# print(a.ab[:10,:10])


