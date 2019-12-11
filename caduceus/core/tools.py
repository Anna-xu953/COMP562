import os
import scipy.misc
import numpy as np

import random
import numpy.random
import os.path
import math
import time
from sys import platform
import gc
from scipy.spatial import distance
import numba
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import tensorflow as tf
import itertools


def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L

def rescale_L2(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L = L*L*2.0/(lmax*lmax) 
    L -= I
    return L

def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]

def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        # lamb, U = scipy.sparse.linalg.eigs(L)
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L)
        # lamb, U = scipy.sparse.linalg.eigsh(L)
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U

def RemoveB0Element(bval, bvec):
    noB0Vol = np.count_nonzero(bval)
    newbval = np.zeros(noB0Vol)
    newbvec = np.zeros([3, noB0Vol])

    c = 0
    for q in range(len(bval)):
        if bval[q] == 0:
            continue
        newbval[c] = bval[q]
        newbvec[0, c] = bvec[0, q]
        newbvec[1, c] = bvec[1, q]
        newbvec[2, c] = bvec[2, q]
        c += 1
    return newbval, newbvec

def ExtractSingleShell(bval, bvec, tbval = 1):
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    newbval = np.zeros(noQ)
    newbvec = np.zeros([3, noQ])

    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            newbval[c] = bval[q]
            newbvec[0, c] = bvec[0, q]
            newbvec[1, c] = bvec[1, q]
            newbvec[2, c] = bvec[2, q]
            c += 1

    return newbval, newbvec

def Separate_scan_bvecs_ind_single_shell(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))


    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if bval[q] == tbval:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0



    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noQ]
    print ('Whole gradient ind: ' + `ind`)

    ind_a = ind[0::3] # for axial scans
    ind_c = ind[1::3] # for coronal scans
    ind_s = ind[2::3] # for sagittal scans
    print ('Axial scan ind: ' + `ind_a`)
    print ('Coronal scan ind: ' + `ind_c`)
    print ('Sagittal scan ind: ' + `ind_s`)

    nind = np.zeros(noQ, dtype = np.int16)
    for q in range(noQ):
        cnt = 0
        for p in range(noQ):
            if (ind[q] > ind[p]):
                cnt += 1
        nind[q] = cnt
    nind_a = nind[0::3]
    nind_c = nind[1::3]
    nind_s = nind[2::3]
    print ('Axial scan ind in [0,89]: ' + `nind_a`)
    print ('Coronal scan ind in [0,89]: ' + `nind_c`)
    print ('Sagittal scan ind in [0,89]: ' + `nind_s`)

    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    group_ind_c = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_c):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_c[i])
        group_ind_c.append(group_ind_tmp)  
    
    print ('Coronal scan group ind: ' + `group_ind_c`)
    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    noQ3 = int(noQ/3)
    nbvec_a = np.zeros([3, noQ3])
    nbvec_c = np.zeros([3, noQ3])
    nbvec_s = np.zeros([3, noQ3])

    cnt_a = 0
    cnt_c = 0
    cnt_s = 0
    for q in range(len(bval)):
        if (q in ind_a):
            nbvec_a[0, cnt_a] = bvec[0, q]
            nbvec_a[1, cnt_a] = bvec[1, q]
            nbvec_a[2, cnt_a] = bvec[2, q]
            cnt_a += 1
        elif (q in ind_c):
            nbvec_c[0, cnt_c] = bvec[0, q]
            nbvec_c[1, cnt_c] = bvec[1, q]
            nbvec_c[2, cnt_c] = bvec[2, q]
            cnt_c += 1
        elif (q in ind_s):
            nbvec_s[0, cnt_s] = bvec[0, q]
            nbvec_s[1, cnt_s] = bvec[1, q]
            nbvec_s[2, cnt_s] = bvec[2, q]
            cnt_s += 1
    return group_ind_a, group_ind_c, group_ind_s, nbvec_a, nbvec_c, nbvec_s, nind_a, nind_c, nind_s

def Separate_scan_bvecs_ind_single_shell_concecutive(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))


    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if bval[q] == tbval:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0



    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noQ]
    print ('Whole gradient ind: ' + `ind`)

    noQ3 = int(noQ/3)
    ind_a = ind[0:noQ3] # for axial scans
    ind_c = ind[noQ3:noQ3*2] # for coronal scans
    ind_s = ind[noQ3*2:noQ] # for sagittal scans
    print ('Axial scan ind: ' + `ind_a`)
    print ('Coronal scan ind: ' + `ind_c`)
    print ('Sagittal scan ind: ' + `ind_s`)

    nind = np.zeros(noQ, dtype = np.int16)
    for q in range(noQ):
        cnt = 0
        for p in range(noQ):
            if (ind[q] > ind[p]):
                cnt += 1
        nind[q] = cnt
    nind_a = nind[0:noQ3]
    nind_c = nind[noQ3:noQ3*2]
    nind_s = nind[noQ3*2:noQ]
    print ('Axial scan ind in [0,89]: ' + `nind_a`)
    print ('Coronal scan ind in [0,89]: ' + `nind_c`)
    print ('Sagittal scan ind in [0,89]: ' + `nind_s`)

    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    group_ind_c = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_c):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_c[i])
        group_ind_c.append(group_ind_tmp)  
    
    print ('Coronal scan group ind: ' + `group_ind_c`)
    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    # noQ3 = int(noQ/3)
    nbvec_a = np.zeros([3, noQ3])
    nbvec_c = np.zeros([3, noQ3])
    nbvec_s = np.zeros([3, noQ3])

    cnt_a = 0
    cnt_c = 0
    cnt_s = 0
    for q in range(len(bval)):
        if (q in ind_a):
            nbvec_a[0, cnt_a] = bvec[0, q]
            nbvec_a[1, cnt_a] = bvec[1, q]
            nbvec_a[2, cnt_a] = bvec[2, q]
            cnt_a += 1
        elif (q in ind_c):
            nbvec_c[0, cnt_c] = bvec[0, q]
            nbvec_c[1, cnt_c] = bvec[1, q]
            nbvec_c[2, cnt_c] = bvec[2, q]
            cnt_c += 1
        elif (q in ind_s):
            nbvec_s[0, cnt_s] = bvec[0, q]
            nbvec_s[1, cnt_s] = bvec[1, q]
            nbvec_s[2, cnt_s] = bvec[2, q]
            cnt_s += 1
    return group_ind_a, group_ind_c, group_ind_s, nbvec_a, nbvec_c, nbvec_s, nind_a, nind_c, nind_s

def Separate_scan_bvecs_ind_single_shell_noRearr_axi_sag(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))

    ind = np.zeros(noQ, dtype=np.int16)
    qCount = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            ind[qCount] = q
            qCount += 1

    print ('Whole gradient ind: ' + `ind`)
    whole_ind = ind

    noQ3 = int(noQ/2)
    ind_a = ind[0:noQ3] # for axial scans
    ind_s = ind[noQ3:noQ3*2] # for sagittal scans

    print ('Axial scan ind: ' + `ind_a`)
    print ('Sagittal scan ind: ' + `ind_s`)

    # need to re-arrange ind_a, ind_s
    qdist = np.zeros(len(bval))
    ref_idx = ind_a[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_a:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_a_new = qdist.ravel().argsort()
    ind_a_new = ind_a_new[0:len(ind_a)]

    qdist = np.zeros(len(bval))
    ref_idx = ind_s[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_s:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_s_new = qdist.ravel().argsort()
    ind_s_new = ind_s_new[0:len(ind_s)]

    print ('Axial scan ind_new: ' + `ind_a_new`)
    print ('Sagittal scan ind_new: ' + `ind_s_new`)


    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a_new[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s_new[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    return group_ind_a, group_ind_s

def Separate_scan_bvecs_ind_single_shell_noRearr_fix(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))

    ind = np.zeros(noQ, dtype=np.int16)
    qCount = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            ind[qCount] = q
            qCount += 1

    print ('Whole gradient ind: ' + `ind`)
    whole_ind = ind

    noQ3 = int(noQ/3)
    ind_a = ind[0:noQ3] # for axial scans
    ind_c = ind[noQ3:noQ3*2] # for coronal scans
    ind_s = ind[noQ3*2:noQ] # for sagittal scans
    # ind_a = ind[0::3] # for axial scans
    # ind_c = ind[1::3] # for coronal scans
    # ind_s = ind[2::3] # for sagittal scans
    print ('Axial scan ind: ' + `ind_a`)
    print ('Coronal scan ind: ' + `ind_c`)
    print ('Sagittal scan ind: ' + `ind_s`)

    # need to re-arrange ind_a, ind_c, ind_s
    qdist = np.zeros(len(bval))
    ref_idx = ind_a[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_a:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_a_new = qdist.ravel().argsort()
    ind_a_new = ind_a_new[0:len(ind_a)]

    qdist = np.zeros(len(bval))
    ref_idx = ind_c[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_c:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_c_new = qdist.ravel().argsort()
    ind_c_new = ind_c_new[0:len(ind_c)]

    qdist = np.zeros(len(bval))
    ref_idx = ind_s[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_s:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_s_new = qdist.ravel().argsort()
    ind_s_new = ind_s_new[0:len(ind_s)]

    print ('Axial scan ind_new: ' + `ind_a_new`)
    print ('Coronal scan ind_new: ' + `ind_c_new`)
    print ('Sagittal scan ind_new: ' + `ind_s_new`)


    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a_new[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    group_ind_c = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_c_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_c_new[i])
        group_ind_c.append(group_ind_tmp)  
    
    print ('Coronal scan group ind: ' + `group_ind_c`)
    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s_new[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    return group_ind_a, group_ind_c, group_ind_s, whole_ind

def Separate_scan_bvecs_ind_single_shell_noRearr_fix2(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))

    ind = np.zeros(noQ, dtype=np.int16)
    qCount = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            ind[qCount] = q
            qCount += 1

    print ('Whole gradient ind: ' + `ind`)
    whole_ind = ind

    noQ3 = int(noQ/3)
    ind_a = ind[0:noQ3] # for axial scans
    ind_c = ind[noQ3:noQ3*2] # for coronal scans
    ind_s = ind[noQ3*2:noQ] # for sagittal scans
    # ind_a = ind[0::3] # for axial scans
    # ind_c = ind[1::3] # for coronal scans
    # ind_s = ind[2::3] # for sagittal scans
    print ('Axial scan ind: ' + `ind_a`)
    print ('Coronal scan ind: ' + `ind_c`)
    print ('Sagittal scan ind: ' + `ind_s`)

    # need to re-arrange ind_a, ind_c, ind_s
    qdist = np.zeros(len(bval))
    ref_idx = ind_a[0]
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_a:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_a_new = qdist.ravel().argsort()
    ind_a_new = ind_a_new[0:len(ind_a)]

    qdist = np.zeros(len(bval))
    # ref_idx = ind_c[0]
    ## find the index which is the closest to ind_a[0]
    qdist_tmp = 1.0
    qCount = 0
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_c:
            qdist[q] = 1.0 - np.power((bvec[0,ind_a[0]]*bvec[0,q] + bvec[1,ind_a[0]]*bvec[1,q] + bvec[2,ind_a[0]]*bvec[2,q]),2)
            if (qdist[q] < qdist_tmp):
                ref_idx = ind_c[qCount]
                qdist_tmp = qdist[q]
            qCount += 1
    print ('ref_indx: ' + `ref_idx`)
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_c:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_c_new = qdist.ravel().argsort()
    ind_c_new = ind_c_new[0:len(ind_c)]

    qdist = np.zeros(len(bval))
    # ref_idx = ind_s[0]
    ## find the index which is the closest to ind_a[0]
    qdist_tmp = 1.0
    qCount = 0
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_s:
            qdist[q] = 1.0 - np.power((bvec[0,ind_a[0]]*bvec[0,q] + bvec[1,ind_a[0]]*bvec[1,q] + bvec[2,ind_a[0]]*bvec[2,q]),2)
            if (qdist[q] < qdist_tmp):
                ref_idx = ind_s[qCount]
                qdist_tmp = qdist[q]
            qCount += 1
    print ('ref_indx: ' + `ref_idx`)
    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if q in ind_s:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0

    ind_s_new = qdist.ravel().argsort()
    ind_s_new = ind_s_new[0:len(ind_s)]

    print ('Axial scan ind_new: ' + `ind_a_new`)
    print ('Coronal scan ind_new: ' + `ind_c_new`)
    print ('Sagittal scan ind_new: ' + `ind_s_new`)


    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a_new[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    group_ind_c = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_c_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_c_new[i])
        group_ind_c.append(group_ind_tmp)  
    
    print ('Coronal scan group ind: ' + `group_ind_c`)
    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s_new):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s_new[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    return group_ind_a, group_ind_c, group_ind_s, whole_ind

def Separate_scan2_bvecs_ind_single_shell(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print ('Number of directions: ' + `noQ`)

    qdist = np.zeros(len(bval))


    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if bval[q] == tbval:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0



    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noQ]
    print ('Whole gradient ind: ' + `ind`)
    
    noQ3 = int(noQ/3)
    ind_a = ind[0:noQ3] # for axial scans
    ind_c = ind[noQ3:noQ3*2] # for coronal scans
    ind_s = ind[noQ3*2:noQ] # for sagittal scans
    print ('Axial scan ind: ' + `ind_a`)
    print ('Coronal scan ind: ' + `ind_c`)
    print ('Sagittal scan ind: ' + `ind_s`)

    group_ind_a = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_a):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_a[i])
        group_ind_a.append(group_ind_tmp)  
    
    print ('Axial scan group ind: ' + `group_ind_a`)

    group_ind_c = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_c):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_c[i])
        group_ind_c.append(group_ind_tmp)  
    
    print ('Coronal scan group ind: ' + `group_ind_c`)
    
    group_ind_s = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind_s):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind_s[i])
        group_ind_s.append(group_ind_tmp)  
    
    print ('Sagittal scan group ind: ' + `group_ind_s`)

    return group_ind_a, group_ind_c, group_ind_s

def Separate_bvecs_ind_single_shell(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print (noQ)

    qdist = np.zeros(len(bval))


    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if bval[q] == tbval:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0



    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noQ]
    print ind

    group_ind = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind[i])
        group_ind.append(group_ind_tmp)  
    
    print group_ind
    
    return group_ind

def Separate_bvecs_ind_single_shell_ext(_params, bval, bvec, ds_factor, tbval = 1, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    #noB0Vol = np.count_nonzero(bval)
    c = 0
    for q in range(len(bval)):
        if bval[q] == tbval:
            c += 1
    noQ = c
    print (noQ)

    qdist = np.zeros(len(bval))


    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        if bval[q] == tbval:
            qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)
        else:
            qdist[q] = 2.0



    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noQ]
    print ind

    group_ind = [] # subset of [0,1,...,288]

    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind[i])
        group_ind.append(group_ind_tmp)  
    
    print group_ind

    ind_red = np.zeros(len(ind),dtype=np.int16)

    for i,j in enumerate(ind):
        cnt = 0
        for k in range(0,len(ind)):
            if (ind[k] < ind[i]):
                cnt += 1
        ind_red[i] = round(cnt)
    print ind_red

    group_ind_red = [] # subset of [0,1,...,89]
    for k in range(0, ds_factor):
        group_ind_tmp_red = []
        for i,j in enumerate(ind_red):
            if (i % ds_factor == k):
                group_ind_tmp_red.append(ind_red[i])
        group_ind_red.append(group_ind_tmp_red)  
    print group_ind_red

    return group_ind, group_ind_red

def Separate_bvecs_ind(_params, bval, bvec, ds_factor, ref_idx = 1):
    #noSrc = _params["training_parameters"]["InDirNo"]["value"]
    noB0Vol = np.count_nonzero(bval)
    #noTar = noB0Vol - noSrc
    qdist = np.zeros(len(bval))

    for q in range(len(bval)):
        # if bval[q] == 0:
        #     continue
        qdist[q] = 1.0 - np.power((bvec[0,ref_idx]*bvec[0,q] + bvec[1,ref_idx]*bvec[1,q] + bvec[2,ref_idx]*bvec[2,q]),2)


    #ind = np.argsort(qdist)
    ind = qdist.ravel().argsort()
    #print ind
    ind = ind[0:noB0Vol]
    print ind


    group_ind = []
    for k in range(0, ds_factor):
        group_ind_tmp = []
        for i,j in enumerate(ind):
            if (i % ds_factor == k):
                group_ind_tmp.append(ind[i])
        group_ind.append(group_ind_tmp)        
    
    print group_ind

    return group_ind

def RemoveB0Element_addB0(bval, bvec):
    noB0Vol = np.count_nonzero(bval) + 1
    newbval = np.zeros(noB0Vol)
    newbvec = np.zeros([3, noB0Vol])

    c = 0
    for q in range(len(bval)):
        if bval[q] == 0 and q != 0:
            continue
        newbval[c] = bval[q]
        newbvec[0, c] = bvec[0, q]
        newbvec[1, c] = bvec[1, q]
        newbvec[2, c] = bvec[2, q]
        c += 1
    return newbval, newbvec

def RemoveB0Element_addB0FA(bval, bvec):
    noB0Vol = np.count_nonzero(bval) + 2
    newbval = np.zeros(noB0Vol)
    newbvec = np.zeros([3, noB0Vol])

    c = 0
    for q in range(len(bval)):
        if bval[q] == 0 and q != 0:
            continue
        newbval[c] = bval[q]
        newbvec[0, c] = bvec[0, q]
        newbvec[1, c] = bvec[1, q]
        newbvec[2, c] = bvec[2, q]
        c += 1
        
    return newbval, newbvec

def KeepOnlyOneNonB0Element(bval, bvec):
    noB0Vol = np.count_nonzero(bval)
    newbval = np.zeros(90)
    newbvec = np.zeros([3, 90])

    c = 0
    for q in range(len(bval)):
        if bval[q] != 1:
            continue
        newbval[c] = bval[q]
        newbvec[0, c] = bvec[0, q]
        newbvec[1, c] = bvec[1, q]
        newbvec[2, c] = bvec[2, q]
        c += 1
    return newbval, newbvec

def NormalizeVector(v):
    d = np.sqrt(np.vdot(v, v))
    return v/d

# @numba.jit(nopython=False, nogil=True)
# def ComputeDistance(_params, dims, idx, tarV, bval, bvec, sigma, weightThreshold):
#     dimL = dims[0] * dims[1] * dims[2] * dims[3]
#     dist = np.zeros(dimL)
#     pos = np.array(idx[0:3])

#     maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
#     maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
#     tarbvec = bvec[:, idx[3]]
#     tarbvec = NormalizeVector(tarbvec)
#     tarbvec[np.isnan(tarbvec)] = 0
#     tarbvec[np.isinf(tarbvec)] = 0

#     tarbval = bval[idx[3]]/1000.0 #normalized bvalues
#     dims_I = dims[1]*dims[2]*dims[3]
#     dims_J = dims[2]*dims[3]
#     dims_K = dims[3]
#     qDist = np.zeros(dims[3])

#     for q in range(dims[3]):
#         curbvec = bvec[:, q]
#         curbvec = NormalizeVector(curbvec)
#         curbvec[np.isnan(curbvec)] = 0
#         curbvec[np.isinf(curbvec)] = 0
#         #angDist = np.exp(np.power(1.0 - np.vdot(tarbvec, curbvec), 2)/-sigma[1])
#         angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
#         #angDist = np.exp(np.power(1.0 - np.abs(np.vdot(tarbvec, curbvec)), 2)/-sigma[1])
#         curbval = bval[q]/1000.0 #normalized bvalues
#         bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2]) 
#         # if curbval != tarbval:
#         #     bDist = 0.0
#         qDist[q] = angDist*bDist

#     # q0 = qDist[0]
#     # qDist[qDist < weightThreshold*500.0] = 0.0
#     # qDist[0] = q0

#     v = tarV
#     while v < dimL:
#         i = v / dims_I
#         j = (v - i*dims_I) / dims_J
#         k = (v - i*dims_I - j*dims_J) / dims_K
#         pos2 = np.array([i, j, k])
#         dif = pos2 - pos
#         d = np.sqrt(np.vdot(dif, dif))
#         spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
#         # if (spatialDist < weightThreshold):
#         #     spatialDist = 0.0
#         idx = i*dims_I + j*dims_J + k*dims_K
#         dist[idx:idx+dims[3]] = spatialDist*qDist
#         v += dims[3]

#     return dist

#@numba.jit(nopython=False, nogil=True)
def ComputeDistance_offset(_params, dims, idx, tarV, bval, bvec, sigma, tar_offset_slice, group_ind):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])

    # maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
    maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0

    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    for q in range(dims[3]):
        curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:
            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist
            # determine which group g belongs to             
            for ds in range(0, ds_factor):
                if (q in group_ind[ds]):
                    offset_slice = ds
                    break;

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        pos2 = np.array([i, j, k])
        dif = pos2 - pos 
        d = np.sqrt(np.vdot(dif, dif) + (tar_offset_slice - offset_slice)*(tar_offset_slice - offset_slice))
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

@numba.jit(nopython=False, nogil=True)
def ComputeDistance(_params, dims, idx, tarV, bval, bvec, sigma):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])

    if (_params["training_parameters"]["input_channels"]["value"] != 1) and (_params["training_parameters"]["input_channels"]["value"] != _params["training_parameters"]["output_channels"]["value"]):
        maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]*_params["training_parameters"]["ds_factor"]["value"]-1)*(dims[2]*_params["training_parameters"]["ds_factor"]["value"]-1))
    else:
        maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0

    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    for q in range(dims[3]):
        curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:
            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        pos2 = np.array([i, j, k])
        # if _params["training_parameters"]["input_channels"]["value"] != 1 and (_params["training_parameters"]["input_channels"]["value"] != _params["training_parameters"]["output_channels"]["value"]):
        #     # pos2 = np.array([i, j, k*_params["training_parameters"]["ds_factor"]["value"]])
        #     pos2 = np.array([i, j, k*2])
        dif = pos2 - pos
        if _params["training_parameters"]["input_channels"]["value"] != 1 and (_params["training_parameters"]["input_channels"]["value"] != _params["training_parameters"]["output_channels"]["value"]):
            d = np.sqrt(dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2]*_params["training_parameters"]["ds_factor"]["value"])
        else:
            d = np.sqrt(np.vdot(dif, dif))        
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

@numba.jit(nopython=False, nogil=True)
def ComputeDistance_tri(_params, dims, idx, tarV, bval, bvec, sigma, nind_a, nind_c, nind_s):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])
    tarq = idx[3]
    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    # maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
    maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0

    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    for q in range(dims[3]):
        curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:
            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)
        
        ii = i
        jj = j
        kk = k
        if (len(nind_a) == 0): # noRearrangement
            if (q >= 30):
                if (q >=60): #sagittal scan
                    [ii,jj,kk] = [kk,ii,jj]
                else: # coronal scan
                    [ii,jj,kk] = [ii,kk,jj]
        else:
            if (q in nind_c): #coronal scan
                [ii,jj,kk] = [ii,kk,jj]
            elif (q in nind_s): # sagittal scan
                [ii,jj,kk] = [kk,ii,jj]
        pos2 = np.array([ii, jj, kk])
        dif = pos2 - pos
        d = np.sqrt(np.vdot(dif, dif))
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        # wfactor = 1.0
        # tarscan = (tarq/30)
        # curscan = (q/30)
        # if (tarscan != curscan): # different scan direction
        #     # print (tarscan)
        #     # print (curscan)
        #     wfactor = 0.5 
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

@numba.jit(nopython=False, nogil=True)
def ComputeDistance_tri3D_2slc(_params, dims, idx, tarV, bval, bvec, sigma, soffset):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])
    # determine slice offset
    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    
    # maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
    maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0

    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    for q in range(dims[3]):
        curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:
            # if (idx[3]%30 == q%30): # more weight on the same scan direction
            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            # else:
            #     angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/(-sigma[1]*0.5))
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)

        ii = i
        jj = j
        kk = k
        kk = k + soffset[q]
        # if (q < 30):
        #     kk = k + soffset[q]
        # elif (q < 60):
        #     jj = j + soffset[q]
        # else:
        #     ii = i + soffset[q]

        # transform from index to position
        if (q >= 30):
            if (q >=60): #sagittal scan
                [ii,jj,kk] = [kk,ii,jj]#[k,i,j]
            else: # coronal scan
                [ii,jj,kk] = [ii,kk,jj]

        pos2 = np.array([ii, jj, kk])
        dif = pos2 - pos
        d = np.sqrt(np.vdot(dif, dif))
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

@numba.jit(nopython=False, nogil=True)
def ComputeDistance_tri3D(_params, dims, idx, tarV, bval, bvec, sigma, soffset):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])
    # determine slice offset
    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    
    # maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
    maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0

    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    for q in range(dims[3]):
        curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:
            # if (idx[3]%30 == q%30): # more weight on the same scan direction
            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            # else:
            #     angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/(-sigma[1]*0.5))
            if (idx[3]%30 != q%30): # less weight on the same scan direction
                angDist *= 0.5
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)

        ii = i
        jj = j
        kk = k
        # kk = k + soffset[q]
        # if (q < 30):
        #     kk = k + soffset[q]
        # elif (q < 60):
        #     jj = j + soffset[q]
        # else:
        #     ii = i + soffset[q]
        # transform from index to position
        if (q >= 30):
            if (q >=60): #sagittal scan
                [ii,jj,kk] = [kk,ii,jj]#[k,i,j]
            else: # coronal scan
                [ii,jj,kk] = [ii,kk,jj]

        pos2 = np.array([ii, jj, kk])
        dif = pos2 - pos
        d = np.sqrt(np.vdot(dif, dif))
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

@numba.jit(nopython=False, nogil=True)
def ComputeDistance_tri3D_reori(_params, dims, idx, tarV, bval, bvec, sigma, soffset):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]
    dist = np.zeros(dimL)
    pos = np.array(idx[0:3])
    # determine slice offset
    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    
    # maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[2]-1)*(dims[2]-1))
    maxDist = np.sqrt((dims[0]-1)*(dims[0]-1) + (dims[1]-1)*(dims[1]-1) + (dims[1]-1)*(dims[1]-1))
    maxDist = maxDist/(np.power(_params["training_parameters"]["voxelsize"]["value"],3))
    # tarbvec = bvec[:, idx[3]]
    tarbvec = np.zeros(3, dtype=np.float32)
    if (idx[3] >=30):
        if (idx[3] >= 60): #sagittal scan
            tarbvec[0] = bvec[1, idx[3]]
            tarbvec[1] = bvec[2, idx[3]]
            tarbvec[2] = bvec[0, idx[3]]
        else:
            tarbvec[0] = bvec[0, idx[3]]
            tarbvec[1] = bvec[2, idx[3]]
            tarbvec[2] = bvec[1, idx[3]]
    else:
        tarbvec = bvec[:, idx[3]]
    tarbvec = NormalizeVector(tarbvec)
    tarbvec[np.isnan(tarbvec)] = 0
    tarbvec[np.isinf(tarbvec)] = 0


    tarbval = bval[idx[3]]/1000.0 #normalized bvalues
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    qDist = np.zeros(dims[3])

    for q in range(dims[3]):
        curbvec = np.zeros(3, dtype=np.float32)
        if (q >=30):
            if (q >= 60): #sagittal scan
                curbvec[0] = bvec[1, q]
                curbvec[1] = bvec[2, q]
                curbvec[2] = bvec[0, q]
            else:
                curbvec[0] = bvec[0, q]
                curbvec[1] = bvec[2, q]
                curbvec[2] = bvec[1, q]
        else:
            curbvec = bvec[:, q]
        # curbvec = bvec[:, q]
        curbvec = NormalizeVector(curbvec)
        curbvec[np.isnan(curbvec)] = 0
        curbvec[np.isinf(curbvec)] = 0
        if sigma[1] == 0.0:
            if q == idx[3]: # same gradient direction
                qDist[q] = 1.0
            else:
                qDist[q] = 0.0
        else:

            angDist = np.exp((1.0 - np.power(np.vdot(tarbvec, curbvec), 2))/-sigma[1])
            curbval = bval[q]/1000.0 #normalized bvalues
            bDist = np.exp(np.power(np.sqrt(curbval)-np.sqrt(tarbval), 2)/-sigma[2])
            qDist[q] = angDist*bDist

    v = tarV
    while v < dimL:
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)

        ii = 0
        jj = 0
        kk = 0
        if (q < 30):
            kk = k + soffset[q]
        elif (q < 60):
            jj = j + soffset[q]
        else:
            ii = i + soffset[q]
        # 
        # if (q >= 30):
        #     if (q >=60): #sagittal scan
        #         [i,j,k] = [k,j,i]#[k,i,j]
        #     else: # coronal scan
        #         [i,j,k] = [i,k,kk

        pos2 = np.array([ii, jj, kk])
        dif = pos2 - pos
        d = np.sqrt(np.vdot(dif, dif))
        spatialDist = np.exp(np.power(d/maxDist, 2)/-sigma[0])
        idx = i*dims_I + j*dims_J + k*dims_K
        dist[idx:idx+dims[3]] = spatialDist*qDist
        v += dims[3]

    return dist

def TestUnPooling(testData):
    i = 0
    n, m = testData.shape
    tNew = np.zeros([n, m*2])
    while i < m:
        val = testData[0, i]
        tNew[0, i*2] = val
        tNew[0, i*2 + 1] = val
        i += 1
    return tNew

def TestMaxPooling(testData):
    i = 0
    n, m = testData.shape
    tNew = np.zeros([n, m/2])
    while i < m - 1:
        val = testData[0, i]
        if (val < testData[0, i+1]):
            val = testData[0, i+1]
        tNew[0, i/2] = val
        i += 2
    return tNew

def LaplacianFromAdjacency(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
    
    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

#@numba.jit(nopython=False, nogil=True)
def Laplacian(_params,dims, bval, bvec, weightThreshold = 0.01, normalized = True, group_ind = [], sigma = []):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]

    A = np.zeros([dimL, dimL])
    D = np.zeros([dimL, dimL])

    v = 0
    if len(sigma) == 0:
        sigma = np.array([_params["training_parameters"]["sdist"]["value"], _params["training_parameters"]["adist"]["value"], _params["training_parameters"]["mdist"]["value"]])

    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    for v in range(dimL):
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        # if (_params["training_parameters"]["input_channels"]["value"] != 1) and (_params["training_parameters"]["input_channels"]["value"] != _params["training_parameters"]["output_channels"]["value"]):
        #     ds_factor = _params["training_parameters"]["ds_factor"]["value"]
        #     # k = k*ds_factor # for slice undersampling
        #     k = k*2 # for slice undersampling
        q = (v - i*dims_I - j*dims_J - k*dims_K)

        idx = np.array([i, j, k, q])
        if len(group_ind) != 0:
            # determine which group g belongs to 
            ds_factor = _params["training_parameters"]["ds_factor"]["value"]
            #print (group_ind_red)
            for ds in range(0, ds_factor):
                if (q in group_ind[ds]):
                    offset_slice = ds
                    break;
            #print (offset_slice)    
            weight = ComputeDistance_offset(_params, dims, idx, v, bval, bvec, sigma, offset_slice, group_ind)
        else:
            weight = ComputeDistance(_params, dims, idx, v, bval, bvec, sigma)
        # weight[weight<weightThreshold] = 0.0
        A[v, v:] = weight[v:]
        A[v:, v] = weight[v:]
        A[v, v] = 0.0
        v += 1

    for v in range(dimL):
        if not normalized:
            D[v, v] = np.sum(A[:, v])
        else:
            D[v, v] = 1.0/np.sqrt(np.sum(A[:, v]))

    D = scipy.sparse.csr_matrix(D, dtype=np.float32)
    A = scipy.sparse.csr_matrix(A, dtype=np.float32)
    if not normalized:
        L = D - A
    else:
        # I = np.identity(dimL)
        I = scipy.sparse.identity(D.size, dtype=np.float32)
        # L = I - np.matmul(D, np.matmul(A,D))
        L = I - D * A * D

    return L, A, D

#@numba.jit(nopython=False, nogil=True)
def Laplacian_ref(_params,dims, bval, bvec, weightThreshold = 0.01, normalized = True, group_ind = [], sigma = []):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]

    A = np.zeros([dimL, dimL])
    D = np.zeros([dimL, dimL])

    v = 0
    if len(sigma) == 0:
        sigma = np.array([_params["training_parameters"]["sdisth"]["value"], _params["training_parameters"]["adisth"]["value"], _params["training_parameters"]["mdisth"]["value"]])

    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    for v in range(dimL):
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)

        idx = np.array([i, j, k, q])
        if len(group_ind) != 0:
            # determine which group g belongs to 
            ds_factor = _params["training_parameters"]["ds_factor"]["value"]
            #print (group_ind_red)
            for ds in range(0, ds_factor):
                if (q in group_ind[ds]):
                    offset_slice = ds
                    break;
            #print (offset_slice)    
            weight = ComputeDistance_offset(_params, dims, idx, v, bval, bvec, sigma, offset_slice, group_ind)
        else:
            weight = ComputeDistance(_params, dims, idx, v, bval, bvec, sigma)
        # weight[weight<weightThreshold] = 0.0
        A[v, v:] = weight[v:]
        A[v:, v] = weight[v:]
        A[v, v] = 0.0
        v += 1

    for v in range(dimL):
        if not normalized:
            D[v, v] = np.sum(A[:, v])
        else:
            D[v, v] = 1.0/np.sqrt(np.sum(A[:, v]))

    D = scipy.sparse.csr_matrix(D, dtype=np.float32)
    A = scipy.sparse.csr_matrix(A, dtype=np.float32)
    if not normalized:
        L = D - A
    else:
        # I = np.identity(dimL)
        I = scipy.sparse.identity(D.size, dtype=np.float32)
        # L = I - np.matmul(D, np.matmul(A,D))
        L = I - D * A * D

    return L, A, D

#@numba.jit(nopython=False, nogil=True)
def Laplacian_tri(_params,dims, bval, bvec, weightThreshold = 0.01, normalized = True, nind_a = [], nind_c = [], nind_s = []):
    dimL = dims[0] * dims[1] * dims[2] * dims[3]

    A = np.zeros([dimL, dimL])
    D = np.zeros([dimL, dimL])
    ds_factor = _params["training_parameters"]["ds_factor"]["value"]
    v = 0
    sigma = np.array([_params["training_parameters"]["sdist"]["value"], _params["training_parameters"]["adist"]["value"], _params["training_parameters"]["mdist"]["value"]])
    dims_I = dims[1]*dims[2]*dims[3]
    dims_J = dims[2]*dims[3]
    dims_K = dims[3]
    for v in range(dimL):
        i = v / dims_I
        j = (v - i*dims_I) / dims_J
        k = (v - i*dims_I - j*dims_J) / dims_K
        q = (v - i*dims_I - j*dims_J - k*dims_K)
        ii = i
        jj = j
        kk = k
        if (len(nind_a) == 0): # noRearrangement
            if (q >= 30):
                if (q >=60): #sagittal scan
                    [ii,jj,kk] = [kk,ii,jj]
                    # print ('sag')
                else: # coronal scan
                    [ii,jj,kk] = [ii,kk,jj]
                    # print ('cor')
        else:
            if (q in nind_c): # coronal scan
                [ii,jj,kk] = [ii,kk,jj]
            elif (q in nind_s): #sagittal scan
                [ii,jj,kk] = [kk,ii,jj]
        idx = np.array([ii, jj, kk, q])

        weight = ComputeDistance_tri(_params, dims, idx, v, bval, bvec, sigma, nind_a, nind_c, nind_s)
        # weight[weight<weightThreshold] = 0.0
        A[v, v:] = weight[v:]
        A[v:, v] = weight[v:]
        A[v, v] = 0.0
        v += 1

    for v in range(dimL):
        if not normalized:
            D[v, v] = np.sum(A[:, v])
        else:
            D[v, v] = 1.0/np.sqrt(np.sum(A[:, v]))

    D = scipy.sparse.csr_matrix(D, dtype=np.float32)
    A = scipy.sparse.csr_matrix(A, dtype=np.float32)
    if not normalized:
        L = D - A
    else:
        # I = np.identity(dimL)
        I = scipy.sparse.identity(D.size, dtype=np.float32)
        # L = I - np.matmul(D, np.matmul(A,D))
        L = I - D * A * D

    return L, A, D


