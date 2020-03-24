import numpy as np
from numba import jit
from .tools.auxiliary import *

### ALIGNMENT CLASS ###
### numba part:
# @jit(nopython=True)
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
    def __init__(self, query, target, gap=1, factor=1, limit=0.7):
        self.query = query
        self.target = target
        self.query_id = self.query.id
        self.target_id = self.target.id
        self.limit = limit
        self.factor = factor
        self.gap = gap
        self.align_and_rotate()

    def align_and_rotate(self): # get the local alignment, calculate optimal rotation matrix for structures to fit into each other
        # choosing the first chain's ER profile and coordinates
        q_chain = first_key(self.query.er)
        q_er = self.query.er[q_chain]
        a = self.query.coordinates[q_chain]
        t_chain = first_key(self.target.er)
        t_er = self.target.er[t_chain]
        b = self.target.coordinates[t_chain]

        ab = dm_euclidian(q_er, t_er) # normal distribution (dmnd) or difference (dmabs)

        # actual alignment, numba speedup available:
        self.i_list, self.j_list, self.is_gap, self.score = nlocalalign(ab,q_er,t_er,self.gap,self.factor,self.limit)

        self.traceback_len = len(self.is_gap)

        i_list = [i for i,g in zip(self.i_list,self.is_gap) if g == 0]
        j_list = [j for j,g in zip(self.j_list,self.is_gap) if g == 0]
        self.len_wo_gaps = len(i_list)
        if self.len_wo_gaps < 1:
            self.rmsd = 0
            return np.zeros((2,3)),np.full((2,3),1000)
        else:
            a = a[i_list,:]
            b = b[j_list,:]
            self.query_centroid = np.mean(a, axis=0)
            self.target_centroid = np.mean(b, axis=0)
            a -= self.query_centroid
            b -= self.target_centroid
            h = a.T@b
            u,s,v = np.linalg.svd(h.T)
            d = np.linalg.det(v.T@u.T)
            r = v.T@np.diag([1,1,d])@u.T
            a = a@r
            self.rmsd = rmsd(a,b)
            self.rotation_matrix = r
            self.query_aligned = a
            self.target_aligned = b

        # GDT_TS:
        dists = np.linalg.norm(a-b, axis=1)
        f1 = np.count_nonzero(np.where(dists < 1))
        f2 = np.count_nonzero(np.where(dists < 2))
        f4 = np.count_nonzero(np.where(dists < 4))
        f8 = np.count_nonzero(np.where(dists < 8))
        self.gdt_ts = 25 * sum([f1,f2,f4,f8])/self.len_wo_gaps if self.len_wo_gaps > 0 else 0

