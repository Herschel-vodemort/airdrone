import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(6,4))
df = pd.DataFrame(data=None, columns=['CityId', 'Day', 'Hour', 'x', 'y'])
df.loc[0] = [1,1,1,1,1]
df.loc[1] = [1,1,1,1,1]
print df.index
print df