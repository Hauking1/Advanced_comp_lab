import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    plot_directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"+"plots\\"
    data_directory_path = "\\".join(__file__.split("\\")[:-1])+"\\"+"data\\"
    with open(data_directory_path+"Task_3_4_50000.txt") as file:
        data = file.readlines()
    splitted = [line.strip("\n").split("\t") for line in data]
    header = splitted[0]
    one_div_popsizes = [1/int(x) for x in header[1:]]
    splitted_floats = np.array([[float(x) for x in part] for part in splitted[1:]])
    disorders = splitted_floats[:,0]
    log_g_typs = np.log(splitted_floats[:,1:])
    disorder_fastes_change = disorders[np.argmin(np.diff(log_g_typs,axis=0),axis=0)]
    poly_coeffs = np.polyfit(one_div_popsizes,disorder_fastes_change,2)
    poly = np.poly1d(poly_coeffs)
    plt.scatter(one_div_popsizes,disorder_fastes_change,label="Data")
    plt.plot(np.linspace(0,one_div_popsizes[0],100),poly(np.linspace(0,one_div_popsizes[0],100)),label="polynomial fit deg:2")
    plt.scatter(0,poly(0),label=r"fit pop. size=$\infty$, $W_c=$"+f"{poly(0):.2f}")
    plt.legend()
    plt.grid()
    plt.ylabel("Disorder")
    plt.xlabel(r"$\frac{1}{populationsize}$")
    plt.title(r"Disorder koresponding to $max(|\Delta log(g^{typ})|)$ plotted over $\frac{1}{pop.size}$")
    plt.savefig(plot_directory_path+"Taske3_4_W_c.png")
    plt.show()
    plt.close()

