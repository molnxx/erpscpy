import sys
import scipy.special as sc
import numpy as np

def dm_euclidian(a,b):
    ''' euclidian distance matrix of 2 sequences '''
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

def dm_normd(a,b):
    ''' distance matrix for 2 normally distributed sequences '''
    m,n = len(a),len(b)
    result = np.zeros((m,n), dtype=np.single)
    if m < n:
        for i in range(m):
            result[i,:] = np.abs(sc.ndtr(a[i])-sc.ndtr(b))
    else:
        for j in range(n):
            result[:,j] = np.abs(sc.ndtr(a)-sc.ndtr(b[j]))
    return result


def chunks(s, n):
    ''' yield successive n-sized chunks from s '''
    for i in range(0, len(s), n):
        yield s[i:i + n]


def progress_bar(title, value, end, bar_width=50):
    ''' simplest progress bar '''
    percent = float(value) / end
    arrow = '-' * int(round(percent * bar_width)-1) + '>'
    spaces = ' ' * (bar_width - len(arrow))
    sys.stdout.write('\r{}: [{}] {}%'.format(title, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if percent==1.0:
        print()
