import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

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
    for _ in range(max_itter):
        pd_multiple_sweeps(populationsize,new_population,c,disorder,lambd,epsilon)
        difference = np.abs(np.var(new_population)-np.var(init_population))/np.abs(np.var(init_population))
        converged = difference<precision
        if converged:
            #print(f"\r converged itterations: {itter:03d}*pop. size, diff: {difference:.1e}, disorder: {disorder}", end="\r")
            #print("converged after: "+str(itter))
            return new_population
        #np.copyto(init_population,new_population)
        init_population = new_population.copy()
    #print(f"not converged disorder: {disorder} diff: {difference:.1e}")
    print("not converged with difference: "+str(difference))
    return new_population

@nb.njit()
def const_pd_equalibrium_single_lambda(num_itter:int,populationsize:int,c:int,disorder:float,lambd:np.ndarray,epsilon:float)->np.ndarray:
    init_population = np.random.rand(populationsize)+1j*np.random.rand(populationsize)
    for _ in range(num_itter):
        pd_multiple_sweeps(populationsize,init_population,c,disorder,lambd,epsilon)
    return init_population

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

@nb.njit()
def pd_measurement(N_mes:int,eq_population:np.ndarray,c:int,lambda_e:float,epsilon:float,disorder:float)->np.ndarray:
    indizes = np.random.randint(0,len(eq_population),c)
    energy = (np.random.rand()-0.5)*disorder
    resulting_mes = np.zeros(N_mes,dtype=np.complex128)
    for index in range(N_mes):
        resulting_mes[index] = pd_h(lambda_e,energy,np.sum(1/eq_population[indizes]),epsilon)
        pd_sweep(eq_population,c,disorder,lambda_e,epsilon)
    return resulting_mes

def pd_spectral_densitiy(cavities:np.ndarray):
    return np.sum(np.imag(1j/cavities))/(np.pi*len(cavities))

@nb.njit()
def pd_typical_cavity_variance(cavities:np.ndarray)->float:
    return np.exp(np.average(np.log(np.imag(np.divide(1j,cavities)))))

@nb.njit(parallel=True,nogil=True)
def parrallel_const_fast_3_4(num_itter:int,populationsize:int,c:int,disorders:np.ndarray,lambd:float,epsilon:float):
    res = np.zeros(len(disorders))
    length_disorder = len(disorders)
    for index in nb.prange(length_disorder):
        res[index] = pd_typical_cavity_variance(const_pd_equalibrium_single_lambda(num_itter,populationsize,c,disorders[index],lambd,epsilon))
    return res

@nb.njit(parallel=True,nogil=True)
def parrallel_const_fast_3_5(n_measurements:int,num_itter:int,populationsize:int,c:int,disorders:np.ndarray,lambd:float,epsilon:float):
    res = np.zeros(len(disorders))
    length_disorder = len(disorders)
    for index in nb.prange(length_disorder):
        res[index] = pd_typical_cavity_variance(pd_measurement(n_measurements,const_pd_equalibrium_single_lambda(num_itter,populationsize,c,disorders[index],lambd,epsilon),c,lambd,epsilon,disorders[index]))
    return res


def Task_3_4_parallel(save_plots_to:str,save_data_to:str):
    """Task 3.4 with a constant number of sweeps as sugested."""
    itters = 50000
    populationsizes = [10**3,5*10**3,10**4]
    c = 3
    disorders = np.arange(10,21,0.1)
    lambd = 0.
    epsilon = 1e-300

    start_pd = time.time()
    typ_gs = np.zeros((len(populationsizes),len(disorders)))
    for index_pop,populationsize in enumerate(populationsizes):
        start_tot = time.time()
        typ_gs[index_pop]=parrallel_const_fast_3_4(itters,populationsize,c,disorders,lambd,epsilon)
        print(f"populationsize: {populationsize} done :) after:{time.time()-start_tot:.2f}")
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
    plt.savefig(save_plots_to+f"Task_3_4_{itters}")
    plt.show()
    plt.close()

    with open(save_data_to+f"Task_3_4_{itters}.txt","w") as file:
        header = ["Disorder\t"]+[str(pop)+"\t" for pop in populationsizes]
        header[0] = header[-1][:-2]+"\n"
        file.writelines(header)
        lines = []
        for index in range(len(disorders)):
            string_line = f"{disorders[index]}\t"
            for pop_index in range(len(populationsizes)):
                string_line+=str(typ_gs[pop_index][index])+"\t"
            lines.append(string_line[:-2]+"\n")
        file.writelines(lines)


def Task_3_5_parallel(save_plots_to:str,save_data_to:str):
    """Task 3.4 with a constant number of sweeps as sugested."""
    itters = 50000
    populationsize = 10**4
    c = 3
    disorders = np.arange(10,21,0.1)
    lambd = 0.
    epsilon = 1e-6
    n_mes = 10**3
    time_now = time.gmtime()
    print(f"starting at: {time_now.tm_hour+2}:{time_now.tm_min} o'Clock")
    start_pd = time.time()
    typ_Gs=parrallel_const_fast_3_5(n_mes,itters,populationsize,c,disorders,lambd,epsilon)
    end_pd = time.time()-start_pd

    time_now = time.gmtime()
    print(f"ending at: {time_now.tm_hour+2}:{time_now.tm_min} o'Clock")
    
    print("\n")
    print(f"time for populationdynamics: {end_pd:.2f}")

    plt.plot(disorders,typ_Gs,label = f"pop.size: {populationsize}")
    plt.grid()
    plt.legend()
    plt.xlabel("Disorder: W")
    plt.ylabel(r"$G^{typ}$")
    plt.yscale("log")
    plt.title(r"$G^{typ}$ for $\epsilon=$"+f"{epsilon:.0e} and for"+r"$N_{pop}=$"+f"{populationsize} with {itters} sweeps")
    plt.savefig(save_plots_to+f"Task_3_5_{itters}")
    plt.show()
    plt.close()

    with open(save_data_to+f"Task_3_5_{itters}.txt","w") as file:
        header = ["Disorder\t"+str(populationsize)+"\n"]
        header[0] = header[-1][:-2]+"\n"
        file.writelines(header)
        lines = []
        for index in range(len(disorders)):
            string_line = f"{disorders[index]}\t"
            string_line+=str(typ_Gs[index])+"\t"
            lines.append(string_line[:-2]+"\n")
        file.writelines(lines)


if __name__=="__main__":
    print("Hello :)")
    directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"+"plots\\"
    data_directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"+"data\\"

    #Task_3_4_parallel(directory_path,data_directory_path)
    Task_3_5_parallel(directory_path,data_directory_path)
