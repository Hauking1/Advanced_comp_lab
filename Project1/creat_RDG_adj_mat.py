import numpy as np
import numba as nb

@nb.njit()
def any_zero_diff(rows:np.ndarray,cols:np.ndarray)->bool:
    return np.sum(rows-cols==0)!=0

@nb.njit()
def generate_rrg_np(n: int, d: int) -> tuple[np.ndarray]:
    """
    Erzeugt einen zufälligen d-regulären Graphen auf n Knoten
    per Stub-Pairing, bis ein einfacher Graph entsteht.
    Gibt die Adjazenzmatrix als scipy.sparse.csr_matrix zurück.
    """
    assert 0 <= d < n, "Grad d muss 0 <= d < n sein"
    assert (n * d) % 2 == 0, "n*d muss gerade sein, sonst kein regular graph möglich"

    # Erzeuge Liste von 'Stubs'
    stubs = np.repeat(np.arange(0,n,dtype=np.int64),d)

    while True:
        np.random.shuffle(stubs)
        edges = np.column_stack((stubs,np.roll(stubs,1)))[::2]
        # Self-Loops filtern
        if any_zero_diff(edges[:,0],edges[:,1]):
            continue
        # Doppelte Kanten filtern (ungerichteter Graph)
        seen = set()
        ok = True
        for u, v in edges:
            if (v, u) in seen or (u, v) in seen:
                ok = False
                break
            seen.add((u, v))
        if not ok:
            continue

        # Wenn hier, ist der Graph einfach
        break

    rows = edges[:,0]
    cols = edges[:,1]
    row = np.zeros(2*len(rows),dtype=np.int64)
    col = np.zeros(2*len(cols),dtype=np.int64)

    for index in range(len(rows)):
        row[2*index] = rows[index]
        row[2*index+1] = cols[index]
        col[2*index] = cols[index]
        col[2*index+1] = rows[index]

    data = np.ones(len(row), dtype=np.int64)
    return row,col,data

@nb.njit()
def set_diag(mat_size:int,rows:np.ndarray,cols:np.ndarray,data:np.ndarray,diag_elements:np.ndarray)->tuple[np.ndarray]:
    """does not set diagonal elements only works when no diagonal elements are previously there"""
    len_mat = len(rows)
    new_rows = np.zeros(len_mat+mat_size,dtype=np.int64)
    new_cols = np.zeros(len_mat+mat_size,dtype=np.int64)
    new_data = np.zeros(len_mat+mat_size)
    for index in range(len_mat):
        new_rows[index]=rows[index]
        new_cols[index]=cols[index]
        new_data[index]=data[index]
    for index in range(mat_size):
        new_index = len_mat+index
        new_rows[new_index] = index
        new_cols[new_index] = index
        new_data[new_index] = diag_elements[index]
    return new_rows,new_cols,new_data

if __name__=="__main__":
    import time
    from scipy.sparse import csc_matrix
    net_size = 4
    c = 3
    start = time.time()
    adj_mat_1 = generate_rrg_np(net_size,c)
    print(f"first itter: {time.time()-start:.2f}")
    start = time.time()
    adj_mat = generate_rrg_np(net_size,c)
    print(f"second itter: {time.time()-start:.2f}")
    #print(adj_mat)
    start = time.time()
    set_diag(net_size,adj_mat_1[0],adj_mat_1[1],adj_mat_1[2],np.random.random(net_size))
    print(f"set diag: {time.time()-start:.2f}")
    start = time.time()
    row,col,data = set_diag(net_size,adj_mat[0],adj_mat[1],adj_mat[2],np.random.random(net_size))
    print(f"set diag: {time.time()-start:.2f}")
    mat = csc_matrix((data,(row,col)))
    print(mat)
    print(mat.toarray())