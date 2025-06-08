import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable

def omega_k(beta:float,pk1:float,pk2:float)->float:
    return 1/(2*beta)*np.log(pk1/pk2)

def next_omega_k(omega_prev:float,h:float, beta:float)->float:
    return h+1/(2*beta)*np.log(np.cosh(beta*(1+omega_prev))/np.cosh(beta*(1-omega_prev)))

def all_omega(h:float, beta:float, pk1:float, pk2:float, num_nodes:int)->np.ndarray:
    omegas = np.zeros(num_nodes)
    omegas[0] = omega_k(beta=beta,pk1=pk1,pk2=pk2)
    for index, value in enumerate(omegas[:-1]):
        omegas[index+1] = next_omega_k(h=h,beta=beta,omega_prev=value)
    return omegas

def find_fixed_point(function:Callable, args:list, inital_value:float, threashhold:float, max_tries:int=10000)->list:
    guess_now = inital_value
    diff = np.inf
    num_itter = 0
    while diff>threashhold:
        if num_itter==max_tries:
            return [guess_now, diff, num_itter,"max tries exceeded"]
    
        gues_next = function(guess_now,*args)
        diff = gues_next-guess_now
        guess_now = gues_next
        num_itter+=1

    return [guess_now, diff, num_itter, "succes"]

def cavity_dist_from_omega(sigma:int,beta:float,omega:float)->float:
    return 0.5*(1+sigma*np.tanh(beta*omega))

def dist_from_omega(sigma:int,beta:float,omega:float,h:float):
    return np.exp(beta*h*sigma)*np.square(np.exp(beta*sigma)*cavity_dist_from_omega(1,beta,omega)+
                                          np.exp(-1*beta*sigma)*cavity_dist_from_omega(-1,beta,omega))

def mean_mag_from_fixed_omega(omega:float|np.ndarray,beta:float|np.ndarray,h:float)->float|np.ndarray:
    dist_1 = dist_from_omega(1,beta=beta,omega=omega,h=h)
    dist_minus_1 = dist_from_omega(-1,beta=beta,omega=omega,h=h)
    return dist_1/(dist_1+dist_minus_1)-dist_minus_1/(dist_1+dist_minus_1)

def analytical_mag(beta:float|np.ndarray,h:float)->float|np.ndarray:
    return (np.exp(beta)*np.sinh(beta*h))/(np.sqrt(np.exp(2*beta)*np.square(np.sinh(beta*h))+np.exp(-2*beta)))


def plot(y_data:list[np.ndarray], x_name:str, y_name:str, plot_name:str, plot_title:str, show:bool=False, line:bool =False, save:bool = False, file_path:str=None, labels:list[str]=None, x_data:np.ndarray=None)->None:
    fig, ax = plt.subplots()
    if np.any(x_data):
        if line:
            ax.plot(x_data,y_data)
        else:
            ax.plot(x_data,y_data,linewidth = 0, marker = "x")
    else:
        if line:
            ax.plot(y_data)
        else:
            ax.plot(y_data,linewidth = 0, marker = "x")
    
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



if __name__ == "__main__":
    directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"
    directory_path_plots = directory_path+"plots\\"
    h = 1/3
    beta = 1/2
    pk_plus = 1/2
    pk_minus = 1/2
    num_nodes = 50
    omegas = all_omega(h=h,beta=beta,pk1=pk_plus,pk2=pk_minus,num_nodes=num_nodes)
    plot(omegas,"cavity position",r"$\omega^i$","task_1_2",r"cavity parameter for $\beta=\frac{1}{2}$ and $h=\frac{1}{3}$ and uniform init. distr.",x_data=[x for x in range(1,num_nodes+1)],save=True, file_path=directory_path_plots)

    pk_plus = 3/4
    pk_minus = 1/4
    num_nodes = 50
    omegas = all_omega(h=h,beta=beta,pk1=pk_plus,pk2=pk_minus,num_nodes=num_nodes)
    plot(omegas,"cavity position",r"$\omega^i$","task_1_3",r"cavity parameter for $\beta=\frac{1}{2}$ and $h=\frac{1}{3}$ and $P^2(\sigma_1=1)=\frac{3}{4}$",x_data=[x for x in range(1,num_nodes+1)],save=True, file_path=directory_path_plots)

    h=1
    betas = np.linspace(1e-7,2.5,100)
    omegas = np.zeros(len(betas))
    iters = np.zeros(len(betas))

    for index,beta in enumerate(betas):
        res_fixed = find_fixed_point(next_omega_k,[h,beta],omega_k(beta=beta,pk1=pk_plus,pk2=pk_minus),1e-12)
        omegas[index] = res_fixed[0]
    
    plot(list(zip(mean_mag_from_fixed_omega(omegas,betas,h),analytical_mag(betas,h))),r"$\beta$",r"<$\sigma$>","task_1_4",r"Numerical <$\sigma$> for different $\beta$ at h=1",save=True,file_path=directory_path_plots,labels=[r"Numeric <$\sigma$>",r"Analytic <$\sigma$>"],x_data=betas)

