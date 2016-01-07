import numpy as np
import scipy.sparse as sparse

def save(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    return
    
def load(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])