import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('RPI_Conference_Tournament_2017-2019.csv',
                  usecols=['Year', 'RPI', 'Rank', 'SOS', 'National Conference Participant', 'Conference Tournament Participant'])
df = pd.DataFrame(data)
columns = ['Year', 'RPI', 'Rank', 'SOS', 'National Conference Participant', 'Conference Tournament Participant']
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
team = df['Year']
rpi = df['RPI']
rank = df['Rank']
sos = df['SOS']
national = df['National Conference Participant']
conference = df['Conference Tournament Participant']

#rpishist = plt.hist([rpi], label='RPI', bins= 'auto')
soshist = plt.hist([sos], label='SOS', color='orange', bins='auto')
plt.xlabel('SOS')
plt.title('SOS Distribution for NCAA D1 Baseball Teams from 2017-2019')
plt.legend()
plt.show()