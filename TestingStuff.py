import numpy as np
import pandas as pd

def cvscore(fclist):
    sd = np.std(fclist)
    mean = np.mean(fclist)
    cv = sd/mean
    return cv

df = pd.DataFrame({'AAA' : ['w','x','y','z'], 'BBB' : [10,20,30,40],
                   'CCC' : [100,50,-30,-50]})

df['Score'] = df.iloc[:, 1:].apply(cvscore, axis=1)
print(df)