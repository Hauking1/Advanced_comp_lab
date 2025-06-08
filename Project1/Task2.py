import networkx as netx
from scipy.sparse import csr_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
from Task1 import plot

def plot_histogram(data:list[np.ndarray], x_name:str, y_name:str, plot_name:str, plot_title:str, show:bool=False, bin_count:int=10, save:bool = False, file_path:str=None, labels:list[str]=None, density:bool = False, with_error:bool=False):
    fig, ax = plt.subplots()
    if with_error:
        all_counts = np.zeros((len(data),bin_count))
        for index,dat in enumerate(data):
            counts, bins = np.histogram(dat,bins=bin_count,density=density)
            all_counts[index] = counts
        
        counts, bins = np.histogram(np.concatenate(data),bins=bin_count,density=density)
        ax.stairs(counts, bins,fill=True,baseline=0)
        bins = bins[:-1]+np.diff(bins)/2
        ax.errorbar(bins,counts,yerr=np.std(all_counts,axis=0), fmt='none',
             ecolor='black',capsize=3)
    else:
        ax.hist(x=data,bins=bin_count,density=density)

    ax.set(xlabel=x_name, ylabel=y_name,
       title=plot_title)
    ax.grid()
    if np.any(labels):
        ax.legend(labels)
    
    if save:
        if file_path==None:
            plt.savefig(plot_name)
        else:
            plt.savefig(file_path+plot_name)

    if show:
        plt.show()
    plt.close()

def plot_histogram_line(data:list[np.ndarray], x_name:str, y_name:str, plot_name:str, plot_title:str,func_line:callable, args_line:list, show:bool=False, bin_count:int=None, save:bool = False, file_path:str=None, labels:list[str]=None, density:bool = False):
    fig, ax = plt.subplots()
    if density:
        if bin_count:
            ax.hist(x=data,bins=bin_count,density=True)
        else:
            ax.hist(x=data,density=True)
    else:
        if bin_count:
            ax.hist(x=data,bins=bin_count)
        else:
            ax.hist(x=data)
    
    line_xs = np.linspace(np.round(min(data)),np.round(max(data)),len(data),endpoint=True)

    ax.plot(line_xs,func_line(line_xs,*args_line))
    ax.set(xlabel=x_name, ylabel=y_name,
       title=plot_title)
    ax.grid()
    if np.any(labels):
        ax.legend(labels)
    
    if save:
        if file_path==None:
            plt.savefig(plot_name)
        else:
            plt.savefig(file_path+plot_name)

    if show:
        plt.show()
    plt.close()

def semi_circle(x:float|np.ndarray,)->float|np.ndarray:
    return np.sqrt(4-np.square(x))/(2*np.pi)

def part1(plotsave_path:str,tasks:str):
    def part_1_task_1():
        num_mat = 10
        mat_size = 2**10
        connectivity = 3

        all_eigenvals = np.zeros((num_mat,mat_size))
        start_total = time.time()
        for num_loop in range(1,num_mat+1):
            start_loop = time.time()
            adjacency_matrix:csr_matrix = netx.adjacency_matrix(netx.random_regular_graph(connectivity,mat_size),dtype=np.bool)
            adjacency_matrix.data = np.random.randn(len(adjacency_matrix.data))*np.sqrt(1/connectivity)
            adjacency_matrix = adjacency_matrix.toarray()

            all_eigenvals[num_loop-1] = np.linalg.eigvalsh(adjacency_matrix)
            print(f"loop: {num_loop} took: {time.time()-start_loop:0.2f}. {num_mat-num_loop} loops left")
        print(f"total computation took: {time.time()-start_total:0.2f} for n = {mat_size} for {num_mat} matrices")
        bin_count = 10
        plot_histogram(all_eigenvals,"eigenvalues","#eigenvalues in the shown range",f"Task_2_1_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count,with_error=True)
        bin_count = 20
        plot_histogram(all_eigenvals,"eigenvalues","#eigenvalues in the shown range",f"Task_2_1_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count,with_error=True)
        bin_count = 5
        plot_histogram(all_eigenvals,"eigenvalues","#eigenvalues in the shown range",f"Task_2_1_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count,with_error=True)
        bin_count = 100
        plot_histogram(all_eigenvals,"eigenvalues","#eigenvalues in the shown range",f"Task_2_1_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count,with_error=True)

    def part_1_task_2():
        num_mat = 10
        mat_size = 2**10
        connectivity = mat_size-1

        all_eigenvals = np.zeros((num_mat,mat_size))
        start_total = time.time()
        for num_loop in range(1,num_mat+1):
            start_loop = time.time()
            adjacency_matrix = np.random.randn(mat_size,mat_size)*np.sqrt(1/connectivity)
            np.fill_diagonal(adjacency_matrix,0)

            all_eigenvals[num_loop-1] = np.linalg.eigvalsh(adjacency_matrix)
            print(f"loop: {num_loop} took: {time.time()-start_loop:0.2f}. {num_mat-num_loop} loops left")
        print(f"total computation took: {time.time()-start_total:0.2f} for n = {mat_size} for {num_mat} matrices")
        bin_count = 10
        plot_histogram(np.concatenate(all_eigenvals),"eigenvalues","#eigenvalues in the shown range",f"Task_2_2_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count)
        bin_count = 20
        plot_histogram(np.concatenate(all_eigenvals),"eigenvalues","#eigenvalues in the shown range",f"Task_2_2_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count)
        bin_count = 5
        plot_histogram(np.concatenate(all_eigenvals),"eigenvalues","#eigenvalues in the shown range",f"Task_2_2_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count)
        bin_count = 100
        plot_histogram(np.concatenate(all_eigenvals),"eigenvalues","#eigenvalues in the shown range",f"Task_2_2_{bin_count}",f"Histogram of eigenvalue occurence for {bin_count} bins",save=True,file_path=plotsave_path,density=True,bin_count=bin_count)
    
    def part_1_task_3():
        mat_size = 2**12
        connectivity = mat_size-1

        start_total = time.time()
        adjacency_matrix = np.random.randn(mat_size,mat_size)*np.sqrt(1/connectivity)
        np.fill_diagonal(adjacency_matrix,0)
        print(f"setup took: {time.time()-start_total:0.2f} for n = {mat_size}")

        all_eigenvals = np.linalg.eigvalsh(adjacency_matrix)

        print(f"total computation took: {time.time()-start_total:0.2f} for n = {mat_size}")

        labels = ["analytic", "numeric"]

        num_bins = 5
        plot_histogram_line(all_eigenvals,"eigenvalues","occurence eigenvalues",f"Task_2_3_{num_bins}",f"Eigenvalues occourence for {num_bins} bins",semi_circle,(),bin_count=num_bins,save=True,file_path=plotsave_path,density=True,labels=labels)
        num_bins = 10
        plot_histogram_line(all_eigenvals,"eigenvalues","occurence eigenvalues",f"Task_2_3_{num_bins}",f"Eigenvalues occourence for {num_bins} bins",semi_circle,(),bin_count=num_bins,save=True,file_path=plotsave_path,density=True,labels=labels)
        num_bins = 20
        plot_histogram_line(all_eigenvals,"eigenvalues","occurence eigenvalues",f"Task_2_3_{num_bins}",f"Eigenvalues occourence for {num_bins} bins",semi_circle,(),bin_count=num_bins,save=True,file_path=plotsave_path,density=True,labels=labels)
        num_bins = 100
        plot_histogram_line(all_eigenvals,"eigenvalues","occurence eigenvalues",f"Task_2_3_{num_bins}",f"Eigenvalues occourence for {num_bins} bins",semi_circle,(),bin_count=num_bins,save=True,file_path=plotsave_path,density=True,labels=labels)
    


    run_tasks = tasks.split(",")
    run_tasks = [tsk.strip() for tsk in run_tasks]
    
    if "1" in run_tasks:
        part_1_task_1()
    
    if "2" in run_tasks:
        part_1_task_2()
    
    if "3" in run_tasks:
        part_1_task_3()

def spectral_density(lamnbda:float,N:int,epsilon:float,to_cav_mat:csr_matrix,max_iter:int,threshold:float,c:int,marginal_mat:csr_matrix)->float:
    cavities = cavity_precision(lamnbda,N,epsilon,to_cav_mat,max_iter,threshold,c,marginal_mat)
    return np.sum(np.imag(1j/cavities))/(np.pi*N)

def cavity_precision(lamnbda:float,N:int,epsilon:float,to_cav_mat:csr_matrix,max_iter:int,threshold:float,c:int,marginal_mat:csr_matrix)->np.ndarray:
    one_over_marginal = np.divide(1,marginal_precision(lamnbda,N,epsilon,marginal_mat,max_iter,threshold,c))
    cavity = 1j*lamnbda+epsilon+to_cav_mat.dot(one_over_marginal)
    return cavity

def to_cavity_matrix(N:int,matrix:csr_matrix,c:int)->csr_matrix:
    data = matrix.data
    col_elements = matrix.nonzero()[1].copy()
    row_elements = matrix.nonzero()[0].copy()
    col_count = np.zeros(N)
    row_count = np.zeros(N)
    for index in range(len(col_elements)):
        col_count_index = col_elements[index]
        row_count_index = row_elements[index]

        col_elements[index] = col_count[col_count_index]
        row_elements[index] = row_count[row_count_index]

        col_count[col_count_index]+=1
        row_count[row_count_index]+=1
    new_coloms = matrix.nonzero()[1]*c+col_elements
    new_row = matrix.nonzero()[0]
    return csr_matrix((data,(new_row,new_coloms)))

def marginal_matrix(N:int,matrix:csr_matrix,c:int)->csr_matrix:
    data = matrix.data
    col_elements = matrix.nonzero()[1].copy()
    row_elements = matrix.nonzero()[0].copy()
    col_count = np.zeros(N)
    row_count = np.zeros(N)
    data_dict = {(col_elements[index],row_elements[index]):data[index] for index in range(len(data))}
    for index in range(len(col_elements)):
        col_count_index = col_elements[index]
        row_count_index = row_elements[index]

        col_elements[index] = col_count[col_count_index]
        row_elements[index] = row_count[row_count_index]

        col_count[col_count_index]+=1
        row_count[row_count_index]+=1
    new_coloms = np.delete(np.repeat(np.reshape(matrix.nonzero()[1]*c+col_elements,(N,c)),c,axis=0).flatten(),[c*x+x%c for x in range(c*N)])
    new_row = np.repeat(matrix.nonzero()[0]*c+row_elements,c-1)
    new_data = [data_dict[pair] for pair in zip(new_coloms//c,new_row//c)]
    return csr_matrix((new_data,(new_row,new_coloms)))


def marginal_precision(lamnbda:float,N:int,epsilon:float,marginal_mat:csr_matrix,max_iter:int,threshold:float,c:int)->np.ndarray:
    marginal_bevore = np.random.random(c*N)+1j*np.random.random(c*N)
    max_rel_difference = 0
    start = time.time()
    for index in range(max_iter):
        marginal_next = 1j*lamnbda+epsilon+marginal_mat.dot(np.divide(1,marginal_bevore))
        max_rel_difference = np.max(np.abs((marginal_next-marginal_bevore)/np.linalg.norm(marginal_bevore)))
        if max_rel_difference<threshold:
            print(f"time for loop: {time.time()-start:.2f} for lambda: {lamnbda:+.3f} finished after: {index:05d} itterations, diff: {max_rel_difference:.1e}, epsilon: {epsilon:.0e}", end="\r")
            return marginal_next
        marginal_bevore = marginal_next
    print(f"Failed after {time.time()-start:.2f} seconds for lambda: {lamnbda:+.3f}, difference: {max_rel_difference:.1e}, epsilon: {epsilon:.0e}", end="\r")
    return marginal_bevore

def part2(save_plots_to:str,tasks:str):
    mat_size = 2**11    # equivalent to N
    connectivity = 3
    epsilon = 1e-2
    max_iter = 6000
    threshold = 1e-4
    lamnbdas = np.linspace(-3,3,2000)

    start_set_up = time.time()
    adjacency_matrix:csr_matrix = netx.adjacency_matrix(netx.random_regular_graph(connectivity,mat_size),dtype=np.bool)
    adjacency_matrix.data = np.random.randn(len(adjacency_matrix.data))*np.sqrt(1/connectivity)
    square_adj_mat = adjacency_matrix.power(2)
    to_cavity_mat = to_cavity_matrix(mat_size,square_adj_mat,connectivity)
    marginal_mat = marginal_matrix(mat_size,square_adj_mat,connectivity)
    set_up_time = time.time()-start_set_up
    
    start_eigenvals = time.time()
    eigenvals = np.linalg.eigvalsh(adjacency_matrix.toarray())
    eigvals_time = time.time()-start_eigenvals

    start_cavity = time.time()
    density = [spectral_density(lamnbda,mat_size,epsilon,to_cavity_mat,max_iter,threshold,connectivity,marginal_mat) for lamnbda in lamnbdas]
    density = np.abs(density)/np.trapezoid(density,lamnbdas)      #np.linalg.norm(density)
    cavity_time = time.time()-start_cavity

    print("")
    print(f"setup is done after: {set_up_time:.2f} seconds")
    print(f"time for direct diogonalization: {eigvals_time:.2f}")
    print(f"time with cavity: {cavity_time:.2f}")
    print("")

    plt.hist(eigenvals,density=True,bins=50,label = "direct Diagonalisation")
    plt.plot(lamnbdas,density, label = r"Cavity method with $\epsilon=$"+f"{epsilon:.2f}",alpha = 0.7)
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.title(r"$\rho(\lambda)$ for $\epsilon=$"+f"{epsilon:.2f}")
    plt.savefig(save_plots_to+"Task_2_5_2")
    plt.close()

    epsilon = 1e-1

    start_cavity = time.time()
    density = [spectral_density(lamnbda,mat_size,epsilon,to_cavity_mat,max_iter,threshold,connectivity,marginal_mat) for lamnbda in lamnbdas]
    density = np.abs(density)/np.trapezoid(density,lamnbdas)      #np.linalg.norm(density)
    cavity_time = time.time()-start_cavity

    print("")
    print(f"setup is done after: {set_up_time:.2f} seconds")
    print(f"time for direct diogonalization: {eigvals_time:.2f}")
    print(f"time with cavity: {cavity_time:.2f}")
    print("")

    plt.hist(eigenvals,density=True,bins=50,label = "direct Diagonalisation")
    plt.plot(lamnbdas,density, label = r"Cavity method with $\epsilon=$"+f"{epsilon:.1f}",alpha = 0.7)
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.title(r"$\rho(\lambda)$ for $\epsilon=$"+f"{epsilon:.1f}")
    plt.savefig(save_plots_to+"Task_2_5_1")
    plt.close()

    epsilon = 1e-3

    start_cavity = time.time()
    density = [spectral_density(lamnbda,mat_size,epsilon,to_cavity_mat,max_iter,threshold,connectivity,marginal_mat) for lamnbda in lamnbdas]
    density = np.abs(density)/np.trapezoid(density,lamnbdas)      #np.linalg.norm(density)
    cavity_time = time.time()-start_cavity

    print("")
    print(f"setup is done after: {set_up_time:.2f} seconds")
    print(f"time for direct diogonalization: {eigvals_time:.2f}")
    print(f"time with cavity: {cavity_time:.2f}")
    print("")

    plt.hist(eigenvals,density=True,bins=50,label = "direct Diagonalisation")
    plt.plot(lamnbdas,density, label = r"Cavity method with $\epsilon=$"+f"{epsilon:.3f}",alpha = 0.7)
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.title(r"$\rho(\lambda)$ for $\epsilon=$"+f"{epsilon:.3f}")
    plt.savefig(save_plots_to+"Task_2_5_3")
    plt.close()


if __name__=="__main__":
    directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"
    directory_path_plots = directory_path+"plots\\"
    print("hi")
    part1(directory_path_plots,"1,2,3")
    part2(directory_path_plots,"4,5,6")
   
    """ #the following if only for testing the functions
    mat_size = 4    # equivalent to N
    connectivity = 3
    epsilon = 1e-2
    max_iter = 6000
    threshold = 1e-4
    lamnbdas = np.linspace(-2.7,2.7,2000)

    start_set_up = time.time()
    adjacency_matrix:csr_matrix = netx.adjacency_matrix(netx.random_regular_graph(connectivity,mat_size),dtype=np.bool)
    adjacency_matrix.data = np.random.randn(len(adjacency_matrix.data))*np.sqrt(1/connectivity)
    set_up_time = time.time()-start_set_up

    marginal_matrix(mat_size,adjacency_matrix,connectivity)
    to_cavity_matrix(mat_size,adjacency_matrix,connectivity)"""
    
    