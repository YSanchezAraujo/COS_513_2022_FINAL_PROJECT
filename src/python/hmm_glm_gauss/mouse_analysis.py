import numpy as numpy
import os
import matplotlib.pyplot as plt
import scipy.stats as ss

data_path = "/jukebox/witten/yoel/data"
mouse = 29
day = 1

def load_data(path, mouse, day):
    X_load_str = "X_fip_" + str(mouse) + "_day_" + str(day) + ".npy"
    y_load_str = "Y_fip_" + str(mouse) + "_day_" + str(day) + ".npy"
    X = np.load(os.path.join(path, X_load_str))
    Y = np.load(os.path.join(path, y_load_str))
    return {"X":X, "Y":Y}

dat = load_data(data_path, mouse, day)    

lik_dist = [ss.norm, ss.norm]

est = fit_hmm_glm_em(dat["Y"][:, 0], lik_dist, dat["X"])


