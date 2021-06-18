import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


sys.path.append('/srv/common/Programmes/python/pygstat/')
sys.path.append('/Users/apryet/Programmes/python/pygstat/')
import pygstat 


# perform unconditional simulation

x = np.arange(0,1001,10)
y = np.arange(0,1001,10)

xx, yy = np.meshgrid(x, y)

loc = np.array( [ xx.ravel(), yy.ravel() ] )
loc = np.array( zip( xx.ravel(), yy.ravel() ) )

vario = {'nugget':0.2, 'model':'Sph', 'c':0.8, 'range':250, 'angle':90.0, 'anis':0.5}

gstat = pygstat.GstatModel(loc = loc, sk_mean = 1, vario = vario, nsim=1)	

gstat.run()

pred = gstat.read_pred()

pred_array = np.reshape(pred[:,3],(101,101))

gstat.plot_pred(N=100j)

# perform kriging

x = np.linspace(1, 100, 25)
y = np.linspace(1, 100, 25)
xx, yy = np.meshgrid(x, y)

loc = np.array( [ xx.ravel(), yy.ravel() ] )
loc = np.array( zip( xx.ravel(), yy.ravel() ) )

px = [10,20,60,80,20,40,40]
py = [5,60,40,20,50,20,90]
pval = [18, 5, 10, 4, 2, 8, 1]

data = np.transpose(np.array([px,py,pval]))

vario = {'nugget':0.1, 'model':'Sph', 'c':1.0, 'range':30.0, 'angle':90.0, 'anis':0.5}

gstat = pygstat.GstatModel(loc = loc, data = data, vario = vario)	

gstat.run()

pred = gstat.read_pred()

gstat.plot_pred(N=100j)


# ============= TRASH

import pygstat

reload(pygstat)


def sine_topo(x,y):
    x0 = 50
    y0 = 50
    L  = 100
    return( np.cos( 2*np.pi/L * np.sqrt( (x-x0)**2 + (y-y0)**2) ) )





vecfunc = np.vectorize(sine_topo)

x = np.linspace(1, 100, 10)
y = np.linspace(1, 100, 10)
xx, yy = np.meshgrid(x, y)

topo=vecfunc(xx,yy)

plt.imshow(topo)
plt.ion()
plt.show()




loc = np.array( [ xx.ravel(), yy.ravel() ] )
loc = np.transpose(loc)

vario = { 'nugget':0, 'range':10, 'model':'Sph', 'c':2}

gstat = pygstat.GstatModel(loc = loc, vario = vario, nsim=10)	

gstat.run()

gstat.plot_pred(N=100j)



# perform unconditional simulation

x = np.linspace(1, 100, 10)
y = np.linspace(1, 100, 10)
xx, yy = np.meshgrid(x, y)

loc = np.array( [ xx.ravel(), yy.ravel() ] )

vario = {'nugget':0.1, 'model':'Sph', 'c':1.0, 'range':100}

gstat = pygstat.GstatModel(loc = loc, vario = vario, nsim=10)	


gstat.run()

pred = gstat.read_pred()



# -------------------

pset = gstat.gen_ptset(10)
plt.scatter(pset[:,0],pset[:,1])

gstat.run()

gstat.plot_pred(N=100j)


pred = gstat.read_pred()


xs0 = pred[:,0]
ys0 = pred[:,1]
zs0 = pred[:,2]

N=100j

extent = ( np.min(pred[:,0]), np.max(pred[:,0]), np.min(pred[:,1]), np.max(pred[:,1]) )

xs,ys = np.mgrid[extent[0]:extent[1]:N, extent[2]:extent[3]:N]

resampled = griddata(xs0, ys0, zs0, xs, ys)

plt.imshow(resampled.T, extent=extent)

reload(pygstat)


gstat = pygstat.GstatModel(loc = a, vario = vario, nsim=10)	

pset = gstat.gen_ptset(10)
plt.scatter(pset[:,0],pset[:,1])

plt.show()


f = open('case.pred')






