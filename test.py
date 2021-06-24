import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame({"label" : np.random.randint(2, size=100).astype(str),
                    "data" : np.random.rayleigh(size=100)})

sns.set(style="darkgrid")
sns.distplot(df[df['label']=='0']['data'],color='green',label='Benign URLs')
sns.distplot(df[df['label']=='1']['data'],color='red',label='Phishing URLs')
plt.title('Url Length Distribution')
plt.legend(loc='upper right')
plt.xlabel('Length of URL')

plt.show()