import numpy as np
import matplotlib.pyplot as plt
import networkx as netx
import time
from scipy.sparse import csr_matrix
from Task2 import to_cavity_matrix,marginal_matrix
import numba as nb


def spectral_density(lamnbda:float,N:int,epsilon:float,to_cavity_mat:csr_matrix,max_iter:int,threshold:float,energies:np.ndarray,marginal_mat:csr_matrix,c:int,rep_e:np.ndarray)->float:
    cavities = cavity_precision(lamnbda,N,epsilon,to_cavity_mat,max_iter,threshold,energies,marginal_mat,c,rep_e)
    return np.sum(np.imag(1j/cavities))/(np.pi*N)

def cavity_precision(lamnbda:float,N:int,epsilon:float,to_cavity_mat:csr_matrix,max_iter:int,threshold:float,energies:np.ndarray,marginal_mat:csr_matrix,c:int,rep_e:np.ndarray)->np.ndarray:
    one_over_marginal = np.divide(1,marginal_precision(lamnbda,N,epsilon,marginal_mat,max_iter,threshold,rep_e,c))
    cavity = 1j*(lamnbda-1j*epsilon-energies)+to_cavity_mat.dot(one_over_marginal)
    return cavity

def marginal_precision(lamnbda:float,N:int,epsilon:float,marginal_mat:csr_matrix,max_iter:int,threshold:float,rep_e:np.ndarray,c:int)->np.ndarray:
    marginal_bevore = np.random.random(c*N)+1j*np.random.random(c*N)
    max_rel_difference = 0
    start = time.time()
    for index in range(max_iter):
        marginal_next = 1j*(lamnbda-1j*epsilon-rep_e)+marginal_mat.dot(np.divide(1,marginal_bevore))
        max_rel_difference = np.max(np.abs((marginal_next-marginal_bevore)/np.linalg.norm(marginal_bevore)))
        if max_rel_difference<threshold:
            print(f"time for loop: {time.time()-start:.2f} for lambda: {lamnbda:+.3f} finished after: {index:05d} itterations, diff: {max_rel_difference:.1e}, epsilon: {epsilon:.0e}", end="\r")
            return marginal_next
        marginal_bevore = marginal_next
    print(f"Failed after {time.time()-start:.2f} seconds for lambda: {lamnbda:+.3f}, difference: {max_rel_difference:.1e}, epsilon: {epsilon:.0e}")
    return marginal_bevore

@nb.njit()
def pd_h(lambda_e:float,energy:float,sum_f_omega:float,epsilon:float):
    return 1j*(lambda_e-1j*epsilon-energy)+sum_f_omega

@nb.njit()
def pd_sweep(population:np.ndarray,c:int,disorder:float,lambda_e:float,epsilon:float):
    indieces = np.random.randint(0,len(population),c)
    rand_elements = population[indieces[:-1]]
    energy = (np.random.rand()-0.5)*disorder
    population[indieces[-1]] = pd_h(lambda_e,energy,np.sum(np.divide(1,rand_elements)),epsilon=epsilon)

@nb.njit()
def pd_multiple_sweeps(N_sweeps:int,population:np.ndarray,c:int,disorder:float,lambda_e:float,epsilon:float):
    for _ in range(N_sweeps):
        pd_sweep(population,c,disorder,lambda_e,epsilon)

@nb.njit()
def pd_equalibrium_single_lambda(precision:float,max_itter:int,populationsize:int,c:int,disorder:float,lambd:np.ndarray,epsilon:float)->np.ndarray:
    init_population = np.random.rand(populationsize)+1j*np.random.rand(populationsize)
    new_population = init_population.copy()
    for itter in range(max_itter):
        pd_multiple_sweeps(populationsize,new_population,c,disorder,lambd,epsilon)
        difference = np.abs(np.var(new_population)-np.var(init_population))/np.abs(np.var(init_population))
        converged = difference<precision
        if converged:
            #print(f"\r converged itterations: {itter:03d}*pop. size, diff: {difference:.1e}, disorder: {disorder}", end="\r")
            print("converged after: "+str(itter))
            return new_population
        #np.copyto(init_population,new_population)
        init_population = new_population.copy()
    #print(f"not converged disorder: {disorder} diff: {difference:.1e}")
    print(difference)
    return new_population

def pd_equilibrium_multiple_lambda(precision:float,max_itter:int,populationsize:int,c:int,disorder:float,lambdas:np.ndarray,epsilon:float)->np.ndarray:
    all_equilibrium = np.zeros((len(lambdas),populationsize),dtype=complex)
    for index,lambd in enumerate(lambdas):
        init_population = np.random.rand(populationsize)+1j*np.random.rand(populationsize)
        new_population = init_population.copy()
        for itter in range(max_itter):
            pd_multiple_sweeps(populationsize,new_population,c,disorder,lambd,epsilon)
            difference = np.abs(np.average(new_population)-np.average(init_population))/np.abs(np.average(init_population))
            converged = difference<precision
            if converged:
                all_equilibrium[index] = new_population
                print(f"\r converged itterations: {itter*populationsize:06d}, lambda: {lambd:+.2f}, diff: {difference:.1e}", end="\r")
                break
            np.copyto(init_population,new_population)
        else:
            print(f"not converged for lmbda: {lambd:.2f}, diff: {difference}")
            all_equilibrium[index] = init_population
    return all_equilibrium

def pd_measurement(N_mes:int,eq_population:np.ndarray,c:int,lambda_e:float,epsilon:float,disorder:float)->np.ndarray:
    indizes = np.random.randint(0,len(eq_population),c)
    energy = (np.random.rand()-0.5)*disorder
    resulting_mes = np.zeros(N_mes,dtype=complex)
    for index in range(N_mes):
        resulting_mes[index] = pd_h(lambda_e,energy,np.sum(1/eq_population[indizes]),epsilon)
        pd_sweep(eq_population,c,disorder,lambda_e,epsilon)
    return resulting_mes

def pd_spectral_densitiy(cavities:np.ndarray):
    return np.sum(np.imag(1j/cavities))/(np.pi*len(cavities))

#@nb.njit()
def pd_typical_cavity_variance(cavities:np.ndarray)->float:
    return np.exp(np.average(np.log(np.imag(np.divide(1j,cavities)))))

def Task_3_2(save_plots_to:str):
    N = 2**10
    connectivity = 3
    epsilon = 1e-3
    max_iter = 6000
    threshold = 1e-4
    lamnbdas = np.linspace(-3,3,500)
    disorder = 0.3

    start_set_up = time.time()
    adjacency_matrix:csr_matrix = netx.adjacency_matrix(netx.random_regular_graph(connectivity,N),dtype=np.bool)    #(j,k)
    energies = (np.random.rand(N)-0.5)*disorder
    hamiltonian:csr_matrix = -1*adjacency_matrix.copy()
    hamiltonian.setdiag(energies)
    to_cavity_mat = to_cavity_matrix(N,adjacency_matrix,connectivity)
    marginal_mat = marginal_matrix(N,adjacency_matrix,connectivity)
    repeated_energies = np.repeat(energies,connectivity)
    set_up_time = time.time()-start_set_up
    
    start_eigenvals = time.time()
    eigenvals = np.linalg.eigvalsh(hamiltonian.toarray())
    eigvals_time = time.time()-start_eigenvals

    start_cavity = time.time()
    density = np.zeros(len(lamnbdas))
    for index,lamnbda in enumerate(lamnbdas):
        density[index] = spectral_density(lamnbda,N,epsilon,to_cavity_mat,max_iter,threshold,energies,marginal_mat,connectivity,repeated_energies)
    density = np.abs(density)/np.trapezoid(density,lamnbdas)
    cavity_time = time.time()-start_cavity

    print("")
    print(f"setup is done after: {set_up_time:.2f} seconds")
    print(f"time for direct diogonalization: {eigvals_time:.2f}")
    print(f"time with cavity: {cavity_time:.2f}")
    print("")

    plt.hist(eigenvals,density=True,bins=50,label = "direct Diagonalisation")
    plt.plot(lamnbdas,density, label = r"Cavity method: $\epsilon=$"+f"{epsilon:.0e}",alpha = 0.7)
    plt.legend()
    plt.grid()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.title(r"$\rho(\lambda)$ for $\epsilon=$"+f"{epsilon:.0e}")
    plt.savefig(save_plots_to+"Task_3_2")
    #plt.show()
    plt.close()

def Task_3_3(save_plots_to:str):
    precision = 1e-3
    max_itter = 500
    populationsize = 10**3
    c = 3
    disorder = 0.3
    lambdas = np.linspace(-3,3,200)
    epsilon = 1e-3
    n_measurements = 2**10

    start_pd = time.time()
    equilibriums_marginal = pd_equilibrium_multiple_lambda(precision,max_itter,populationsize,c,disorder,lambdas,epsilon)
    resulting_mesurements = [pd_measurement(n_measurements,equilibriums_marginal[index],c,lambdas[index],epsilon,disorder) for index in range(len(lambdas))]
    specrtal_densities = np.zeros(len(lambdas))
    for index in range(len(lambdas)):
        specrtal_densities[index] = pd_spectral_densitiy(resulting_mesurements[index])
    end_pd = time.time()-start_pd
    
    """for referenc a spectrum from direct diagonalisation"""
    start_eigenvals = time.time()
    adjacency_matrix:csr_matrix = netx.adjacency_matrix(netx.random_regular_graph(c,n_measurements),dtype=np.bool)    #(j,k)
    energies = (np.random.rand(n_measurements)-0.5)*disorder
    hamiltonian:csr_matrix = -1*adjacency_matrix.copy()
    hamiltonian.setdiag(energies)
    eigenvals = np.linalg.eigvalsh(hamiltonian.toarray())
    eigvals_time = time.time()-start_eigenvals

    print("\n")
    print(f"time for populationdynamics: {end_pd:.2f} time for direct: {eigvals_time:.2f}")

    plt.hist(eigenvals,density=True,bins=50,label = "direct Diagonalisation")
    plt.plot(lambdas,specrtal_densities/np.trapezoid(specrtal_densities,lambdas),label = f"pd: {n_measurements} meas, pop.size: {populationsize}")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.title(r"$\rho(\lambda)$ for $\epsilon=$"+f"{epsilon:.0e}"+r" and $N_p=$"+str(populationsize))
    plt.savefig(save_plots_to+"Task_3_3")
    #plt.show()
    plt.close()

def Task_3_4(save_plots_to:str):
    precision = 5*1e-5
    max_itter = 100000
    populationsizes = [10**3,5*10**3,10**4]
    c = 3
    disorders = np.arange(10,21)
    lambd = 0
    epsilon = 1e-300
    start_pd = time.time()
    typ_gs = np.zeros((len(populationsizes),len(disorders)))
    #res_measurements = np.zeros((len(disorders),n_measurements),dtype=complex)
    for index_pop,populationsize in enumerate(populationsizes):
        for index_dis,disorder in enumerate(disorders):
            start = time.time()
            equilibriums_marginal = pd_equalibrium_single_lambda(precision,max_itter,populationsize,c,disorder,lambd,epsilon)
            typ_gs[index_pop][index_dis] = pd_typical_cavity_variance(equilibriums_marginal)
            print(f"disorder: {disorder} at pop_size: {populationsize} done after: {time.time()-start:.2f} seconds")
        #typ_gs[index_pop] = np.array([pd_typical_cavity_variance(res_measurements[index]) for index in range(len(disorders))])
        print(f"populationsize: {populationsize} done :)")
    end_pd = time.time()-start_pd
    
    print("\n")
    print(f"time for populationdynamics: {end_pd:.2f}")

    for index in range(len(populationsizes)):
        plt.plot(disorders,typ_gs[index],label = f"pop.size: {populationsizes[index]}")
    plt.grid()
    plt.legend()
    plt.xlabel("Disorder: W")
    plt.ylabel(r"$g^{typ}$")
    plt.yscale("log")
    plt.title(r"$g^{typ}$ for $\epsilon=$"+f"{epsilon:.0e} and for different populationsizes")
    plt.savefig(save_plots_to+"Task_3_4")
    plt.show()
    plt.close()

if __name__=="__main__":
    print("Hello :)")
    directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"+"plots\\"
    #Task_3_2(directory_path)
    #Task_3_3(directory_path)
    Task_3_4(directory_path)


