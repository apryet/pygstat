execfile('./test.py')

import numpy as np
from pygstat import *


a = np.array(np.arange(3))
a = np.vstack((a,a))


gstat = GstatModel()	

a = np.array(np.arange(3))
a = np.vstack((a,a))

gstat.write_table(a,'./test.dat')
