# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

sys.path.append( os.path.abspath( '../../opti-trade' ) )

# ***********************************************************************
# Utilities
# ***********************************************************************

def make_signal_ode( Gamma, bcTime, endTime, nTimes ):
    
    nDims = Gamma.shape[0]

    def fun( t, y ):
        return -np.tensordot( Gamma,
                              np.tensordot( y, y, axes = 0 ),
                              ( (1,2), (0,1) ) )

    def jac( t, y ):
        return -2.0 * np.tensordot( Gamma, y, axes = ( (2), (0) ) )

    timeSpan = ( bcTime, endTime )        
    timeEval = np.linspace( bcTime, endTime, nTimes )

    res = solve_ivp( fun      = fun, 
                     y0       = bcVec, 
                     t_span   = timeSpan,
                     t_eval   = timeEval,
                     method   = 'LSODA', 
                     rtol     = 1.0e-6            )
    
    print( 'Success:', res.success )

    if bcTime < endTime:
        y = res.y 
    else:
        y = np.flip( res.y, 1 )
        
    return y

# ***********************************************************************
# Generate nonlinear curves
# ***********************************************************************

nDims    = 3
nInTimes = 100000
nOosTimes= int( 0.2 * nInTimes )
nTimes   = nInTimes + nOosTimes
bcVec    = np.array( [ 1.0, 1.0, 1.0 ] )
Gamma    = np.zeros( shape = ( nDims, nDims, nDims ), dtype = 'd' )

Gamma[0][0][1] = -0.5
Gamma[0][0][2] = 0.4
Gamma[1][1][0] = 0.2
Gamma[1][1][2] = 0.2
Gamma[2][2][0] = -0.2
Gamma[2][2][1] = 0.2

Gamma[0][1][0] = Gamma[0][0][1]
Gamma[0][2][0] = Gamma[0][0][2]
Gamma[1][0][1] = Gamma[1][1][0]
Gamma[1][2][1] = Gamma[1][1][2]
Gamma[2][0][2] = Gamma[2][2][0]
Gamma[2][1][2] = Gamma[2][2][1]
    
yIn  = make_signal_ode( Gamma   = Gamma,
                        bcTime  = 1.0,
                        endTime = 0.0,
                        nTimes  = nInTimes  )

yOos = make_signal_ode( Gamma   = Gamma,
                        bcTime  = 1.0,
                        endTime = 1.2,
                        nTimes  = nOosTimes )

y    = np.zeros( shape = ( nDims, nTimes ), dtype = 'd' )

y[0] = np.concatenate( [ yIn[0], yOos[0] ] )
y[1] = np.concatenate( [ yIn[1], yOos[1] ] )
y[2] = np.concatenate( [ yIn[2], yOos[2] ] )

plt.plot( y[0] )
plt.plot( y[1] )
plt.plot( y[2] )
plt.legend( [ 'y1', 'y2', 'y3' ] )
plt.show()

# ***********************************************************************
# Generate signals
# ***********************************************************************

sigma  = 0.02
sigmas = np.random.uniform( 0.0, sigma, nTimes )

for i in range( nTimes ):
    sigmas[i] = max( sigmas[i], 0 )

# t = np.linspace( 0, 1.2, nTimes )
# f1 = 1.2
# f2 = 1.5
# f3 = 1.2
# y[0] += sigmas * np.sin( 2.0 * np.pi * f1 * (t-t[nInTimes-1]) )
# y[1] += sigmas * np.sin( 2.0 * np.pi * f2 * (t-t[nInTimes-1]) )
# y[2] += sigmas * np.sin( 2.0 * np.pi * f3 * (t-t[nInTimes-1]) )

y1_signal = np.random.normal( y[0], sigmas )
y2_signal = np.random.normal( y[1], sigmas )
y3_signal = np.random.normal( y[2], sigmas )

y1_signal[nInTimes-1] = y[0][nInTimes-1]
y2_signal[nInTimes-1] = y[1][nInTimes-1]
y3_signal[nInTimes-1] = y[2][nInTimes-1]

plt.plot( y1_signal )
plt.plot( y2_signal )
plt.plot( y3_signal )
plt.plot( y[0] )
plt.plot( y[1] )
plt.plot( y[2] )
plt.show()

initDate = pd.to_datetime( '2004-01-06 09:00:00' )

dates = []
for i in range( nTimes ):
    dates.append( initDate + datetime.timedelta( minutes = i ) )

df = pd.DataFrame( { 'Date' : dates,
                     'y1'  : y1_signal,
                     'y2'  : y2_signal,
                     'y3'  : y3_signal } )

df.to_pickle( 'random_signals.pkl' )

print( 'minTrnDate =', dates[0] )
print( 'maxTrnDate =', dates[nInTimes-1] )
print( 'maxOosDate =', dates[nTimes-1] )

