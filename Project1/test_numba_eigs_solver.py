import numpy as np
from numba import njit, prange

@njit()
def coo_to_csr(rows, cols, data, n):
    """
    Converts COO (rows, cols, data) to CSR format.
    Returns (csr_data, csr_indices, csr_indptr)
    """
    nnz = data.shape[0]

    # Step 1: Count entries per row → indptr
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(nnz):
        indptr[rows[i] + 1] += 1
    for i in range(1, n + 1):
        indptr[i] += indptr[i - 1]

    # Step 2: Fill indices and data arrays
    csr_data    = np.empty(nnz, dtype=data.dtype)
    csr_indices = np.empty(nnz, dtype=cols.dtype)
    counter = indptr.copy()

    for i in range(nnz):
        row = rows[i]
        dest = counter[row]
        csr_data[dest]    = data[i]
        csr_indices[dest] = cols[i]
        counter[row] += 1

    return csr_data, csr_indices, indptr

@njit(parallel=True,nogil=True)
def csr_matvec(data, indices, indptr, x, y):
    n = indptr.shape[0] - 1
    for i in prange(n):
        y[i] = 0.0
    for i in prange(n):
        acc = 0.0
        for ptr in range(indptr[i], indptr[i+1]):
            acc += data[ptr] * x[indices[ptr]]
        y[i] = acc

@njit(parallel=True,nogil=True)
def cg_solve_csr(data, indices, indptr, shift, b, x, tol, maxiter):
    n = b.shape[0]
    Ap = np.empty(n)
    # compute initial residual r = b - (A - shift I) x
    csr_matvec(data, indices, indptr, x, Ap)
    for i in prange(n):
        Ap[i] -= shift * x[i]
    r = b - Ap
    p = r.copy()
    rsold = np.dot(r, r)
    if rsold < tol * tol:
        return x
    for _ in range(maxiter):
        csr_matvec(data, indices, indptr, p, Ap)
        for i in prange(n):
            Ap[i] -= shift * p[i]
        denom = np.dot(p, Ap)
        if abs(denom) < 1e-16:
            break
        alpha = rsold / denom
        for i in prange(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rsnew = np.dot(r, r)
        if rsnew < tol * tol:
            break
        beta = rsnew / (rsold + 1e-16)
        for i in prange(n):
            p[i] = r[i] + beta * p[i]
        rsold = rsnew
    return x

@njit(parallel=True,nogil=True)
def gram_schmidt(X, eps=1e-16):
    n, k = X.shape
    Q = np.empty((n, k))
    for j in range(k):
        v = X[:, j].copy()
        for i in range(j):
            rij = 0.0
            for t in range(n):
                rij += Q[t, i] * v[t]
            for t in range(n):
                v[t] -= rij * Q[t, i]
        norm_sq = np.dot(v, v)
        if norm_sq < eps:
            # replace with random vector if nearly zero
            v = np.random.randn(n)
            norm_sq = np.dot(v, v)
        norm = np.sqrt(norm_sq)
        for t in range(n):
            Q[t, j] = v[t] / norm
    return Q

@njit(parallel=True,nogil=True)
def block_inverse_iteration_csr(data, indices, indptr, shift, k, tol, maxiter, cg_tol, cg_maxit):
    n = indptr.shape[0] - 1
    X = np.random.randn(n, k)
    X = gram_schmidt(X)
    Y = np.empty_like(X)
    T = np.empty((k, k))
    x0 = np.zeros(n)

    for iteration in range(maxiter):
        # solve block of systems
        for j in prange(k):
            # reset initial guess
            for i in prange(n):
                x0[i] = 0.0
            yj = cg_solve_csr(data, indices, indptr, shift, X[:, j], x0, cg_tol, cg_maxit)
            for i in prange(n):
                Y[i, j] = yj[i]

        # orthonormalize basis
        Q = gram_schmidt(Y)

        # build small matrix T = Q^T A Q
        for i in prange(k):
            for j in range(k):
                acc = 0.0
                for p in range(n):
                    tmp = 0.0
                    for ptr in range(indptr[p], indptr[p+1]):
                        tmp += data[ptr] * Q[indices[ptr], j]
                    acc += Q[p, i] * tmp
                T[i, j] = acc

        # solve small eigenproblem
        w, S = np.linalg.eigh(T)

        # select k closest eigenvalues to shift
        idxs = np.arange(k)
        for a in range(k):
            best = a
            for b in range(a+1, k):
                if abs(w[b] - shift) < abs(w[best] - shift):
                    best = b
            # swap
            temp = w[a]; w[a] = w[best]; w[best] = temp
            temp_i = idxs[a]; idxs[a] = idxs[best]; idxs[best] = temp_i

        # reconstruct Ritz vectors
        Vritz = np.empty((n, k))
        for i in prange(n):
            for j in range(k):
                acc = 0.0
                col = idxs[j]
                for p in range(k):
                    acc += Q[i, p] * S[p, col]
                Vritz[i, j] = acc

        # convergence check
        max_res = 0.0
        for j in prange(k):
            for i in prange(n):
                tmp = 0.0
                for ptr in range(indptr[i], indptr[i+1]):
                    tmp += data[ptr] * Vritz[indices[ptr], j]
                tmp -= shift * Vritz[i, j]
                tmp -= w[j] * Vritz[i, j]
                if abs(tmp) > max_res:
                    max_res = abs(tmp)
        if max_res < tol:
            return w[:k], Vritz

        X = Vritz

    return w[:k], X


if __name__=="__main__":
    import creat_RDG_adj_mat as RDG
    from scipy.sparse import csr_matrix
    import time
    N=2**12
    c=3
    lamb = 0.
    num_evecs = 4
    mat = RDG.generate_rrg_np(N,c)
    mat_csr = csr_matrix((mat[2],(mat[0],mat[1])))
    mat = coo_to_csr(mat[0],mat[1],mat[2],N)
    print(mat_csr.toarray())
    time_numb = time.time()
    vals,vecs = block_inverse_iteration_csr(mat[0],mat[1],mat[2],lamb,num_evecs,10e-8,200,10e-8,200)
    print(f"numba took first itter: {time.time()-time_numb}")
    time_numb = time.time()
    vals,vecs = block_inverse_iteration_csr(mat[0],mat[1],mat[2],lamb,num_evecs,10e-8,200,10e-8,200)
    print(f"numba took second itter: {time.time()-time_numb}")
    print(vals)
    print(vecs.shape)
    time_np = time.time()
    nvals,nvecs = np.linalg.eigh(mat_csr.toarray())
    print(f"np took: {time.time()-time_np}")
    print(nvals)
    print(nvecs.shape)
