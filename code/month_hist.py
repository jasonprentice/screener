import numpy as np
import matplotlib.pyplot as plt
import os

dirs = os.listdir("../data/edgar")

month_hist = np.zeros(12)
for CIK in dirs:
    if CIK[0] != '.':
        yr_dirs = [x for x in os.listdir("../data/edgar/" + CIK) if os.path.isdir(os.path.join("../data/edgar/"+CIK, x))]
        for yr in yr_dirs:
            if yr[0] != '.':
                mo_dirs = os.listdir("../data/edgar/" + CIK + "/" + yr)
                for mo in mo_dirs:
                     if mo[0] != '.':
                        month_hist[int(mo)-1] = month_hist[int(mo)-1] + 1

print month_hist
plt.bar(range(1,13), month_hist)
