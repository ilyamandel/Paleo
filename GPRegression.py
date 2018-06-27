import numpy as np
import pandas as pd
import GPy
from matplotlib import pyplot as plt
import pylab as pb

## Read in the modern data

Tmod = pd.read_csv("modern_temp.csv")
YY = pd.read_csv("mod_ilr.csv")
Tmod = np.asmatrix(Tmod)
YY = np.asmatrix(YY)
Tmod = np.asarray(pd.DataFrame(Tmod).apply(pd.to_numeric))
YY = np.asarray(pd.DataFrame(YY).apply(pd.to_numeric))
Tmod = Tmod[:,1]
Y1 = YY[:,1]
Y2 = YY[:,2]
Y3 = YY[:,3]

Tmod = Tmod[:,None]
Y1 = Y1[:,None]
Y2 = Y2[:,None]
Y3 = Y3[:,None]

## Fit the 'backward model', i.e. temperature = f(GDGTs) + noise, with 
## linear mean function

KK = GPy.kern.ExpQuad(input_dim = 3,ARD = True) + GPy.kern.Linear(3,ARD = True) + GPy.kern.Bias(3)

mb = GPy.models.GPRegression(YY[:,1:4],Tmod,KK)

mb.optimize()

gppred = mb.predict(YY[:,1:4])

## plot predictions against latitude

latitudes = pd.read_csv("Modern.csv").ix[:,'latitude']
latitudes = np.asarray(pd.DataFrame(latitudes).apply(pd.to_numeric))

plt.scatter(latitudes,gppred[0])

## predict rockall data

Yrock = pd.read_csv("rock_ilr.csv")
Yrock = np.asarray(pd.DataFrame(Yrock).apply(pd.to_numeric))
predrockall = mb.predict(Yrock[:,1:4])

sortedrock = np.vstack((predrockall[0][:,0],
                               predrockall[1][:,0])).T

sortedrock = sortedrock[np.argsort(sortedrock[:, 0])]
                        
## plot sorted predictions with standard deviation of posterior predictive

plt.errorbar(np.linspace(1,77,77),
             sortedrock[:,0],
             np.sqrt(sortedrock[:,1]),
             fmt = 'bo')


## predict cretaceous data

Ycret = pd.read_csv("cret_ilr.csv")
Ycret = np.asarray(pd.DataFrame(Ycret).apply(pd.to_numeric))
predcret = mb.predict(Ycret[:,1:4])

sortedcret = np.vstack((predcret[0][:,0],
                               predcret[1][:,0])).T

sortedcret = sortedcret[np.argsort(sortedcret[:, 0])]
                        
plt.errorbar(np.linspace(1,948,948),
             sortedcret[:,0],
             np.sqrt(sortedcret[:,1]),
             fmt = 'bo')

## predict eocene data

Yeoc = pd.read_csv("eoc_ilr.csv")
Yeoc = np.asarray(pd.DataFrame(Yeoc).apply(pd.to_numeric))
predeoc = mb.predict(Yeoc[:,1:4])

sortedeoc = np.vstack((predeoc[0][:,0],
                               predeoc[1][:,0])).T

sortedeoc = sortedeoc[np.argsort(sortedeoc[:, 0])]
                        
plt.errorbar(np.linspace(1,692,692),
             sortedeoc[:,0],
             np.sqrt(sortedeoc[:,1]),
             fmt = 'bo')

## predict eocene data with latitudes
Yeoclat = pd.read_csv("eoclat_ilr.csv")
Yeoclat = np.asarray(pd.DataFrame(Yeoclat).apply(pd.to_numeric))
predeoclat = mb.predict(Yeoclat[:,2:5])

sortedeoclat = np.vstack((predeoclat[0][:,0],
                               predeoclat[1][:,0])).T

sortedeoclat = sortedeoclat[np.argsort(sortedeoclat[:, 0])]
                        
plt.errorbar(np.linspace(1,842,842),
             sortedeoclat[:,0],
             np.sqrt(sortedeoclat[:,1]),
             fmt = 'bo')

plt.plot(Yeoclat[:,1],predeoclat[0])

## cross-validation

validation_rows = pd.read_csv("validation.txt",sep = '\t',header = None)
validation_rows = np.asmatrix(validation_rows)
validation_rows = validation_rows[:,0:85]
validation_rows = validation_rows - 1

mod_ilr = YY[:,1:4]
rmsegp = np.empty([10])

for i in range(10):
    xtrain = np.delete(mod_ilr,validation_rows[i,:],axis = 0)
    ytrain = np.delete(Tmod,validation_rows[i,:],axis = 0)
    xtest = mod_ilr[validation_rows[i,:].astype(int),:][0,:,:]
    ytest = Tmod[validation_rows[i,:].astype(int),:][0,:,:]
    
    KK = GPy.kern.ExpQuad(3,ARD = True) + GPy.kern.Linear(3,ARD =True) + GPy.kern.Bias(3)
    mb = GPy.models.GPRegression(xtrain,ytrain,KK)
    mb.optimize()
    gppred = mb.predict(xtest)[0]
    rmsegp[i] = (np.sqrt(np.mean(np.square(gppred - ytest))))