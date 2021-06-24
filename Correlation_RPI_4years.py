import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

rpi_data = pd.read_csv('RPI_2016-2018_Titles.csv')

rpi = pd.DataFrame(rpi_data)
rpi.head()
rpi_corr = rpi.corr(method='pearson')

sb.heatmap(rpi_corr,
           xticklabels=rpi_corr.columns,
           yticklabels=rpi_corr.columns,
           cmap='RdBu_r',
           annot=True,
           linewidth=0.5)
plt.show()