# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

sys.path.append( os.path.abspath( '../../opti-trade' ) )

# ***********************************************************************
# Some utilities
# ***********************************************************************

def plot_3d( ax,
             vec1,
             vec2,
             vec3,
             name1 = None,
             name2 = None,
             name3 = None,
             lColor  = 'b-',
             pColor  = 'r',
             endFlag = True,
             cBeg1   = 1,
             cBeg2   = 1,
             cBeg3   = 1,
             cEnd1   = 1,
             cEnd2   = 1,
             cEnd3   = 1,
             textBeg = '$t = 0$',
             textEnd = '$t = T$',             
             vecInds = [],
             vecLen  = 150      ):

    ax.plot( vec1, vec2, vec3, lColor )

    if endFlag:
        xBeg = [ vec1[0] ]
        yBeg = [ vec2[0] ]
        zBeg = [ vec3[0] ]
        xEnd = [ vec1[-1] ]
        yEnd = [ vec2[-1] ]
        zEnd = [ vec3[-1] ]

        ax.scatter( xBeg,
                    yBeg,
                    zBeg,
                    c = pColor,
                    depthshade = False,
                    marker = 'o' )

        ax.scatter( xEnd,
                    yEnd,
                    zEnd,
                    c = pColor,
                    depthshade = False,
                    marker = 'o' )

        ax.text( cBeg1 * vec1[0],
                 cBeg2 * vec2[0],
                 cBeg3 * vec3[0],
                 textBeg,
                 fontsize = 16 )

        ax.text( cEnd1 * vec1[-1],
                 cEnd2 * vec2[-1],
                 cEnd3 * vec3[-1],
                 textEnd,
                 fontsize = 16 )

    vel1 = np.gradient( vec1 )
    vel2 = np.gradient( vec2 )
    vel3 = np.gradient( vec3 )

    for ind in vecInds:
    
        ax.quiver( vec1[ind],
                   vec2[ind],
                   vec3[ind],
                   vel1[ind],
                   vel2[ind],
                   vel3[ind],
                   color     = 'r',
                   length    = vecLen,
                   normalize = False )

    ax.xaxis.labelpad = -5.0      
    ax.yaxis.labelpad = -5.0
    ax.zaxis.labelpad = -8.0
        
    ax.grid( True )
    ax.set_xticks( [] )
    ax.set_yticks( [] )
    ax.set_zticks( [] )
    
    if name1 is not None:
        ax.set_xlabel( name1, fontsize = 16 )

    if name2 is not None:
        ax.set_ylabel( name2, fontsize = 16 )

    if name3 is not None:
        ax.set_zlabel( name3, fontsize = 16 )

# ***********************************************************************
# Generate the demo plots
# ***********************************************************************
    
df = pd.read_pickle('visualization_daily_dfFile.pkl')
 
df[ 'u1' ] = df.SPY.rolling( 720,
                             win_type = 'blackman',
                             center = True ).mean()

df[ 'u2' ] = df.TYX.rolling( 720,
                             win_type = 'blackman',
                             center = True ).mean()

df[ 'u3' ] = df.OIH.rolling( 720,
                             win_type = 'blackman',
                             center = True ).mean()

df = df.dropna()

df[ 'u1' ] = df.u1.apply( lambda x : ( x - df.u1.min() ) / ( df.u1.max() - df.u1.min() ) )
df[ 'u2' ] = df.u2.apply( lambda x : ( x - df.u2.min() ) / ( df.u2.max() - df.u2.min() ) )
df[ 'u3' ] = df.u3.apply( lambda x : ( x - df.u3.min() ) / ( df.u3.max() - df.u3.min() ) )

df[ 'x1' ] = np.cumsum( df.SPY )
df[ 'x2' ] = np.cumsum( df.TYX )
df[ 'x3' ] = np.cumsum( df.OIH )

plt_df = df[ df.Date >= '2009-01-01' ]
plt_df = plt_df.sort_values( 'Date' )

vec1 = np.array( plt_df.u1 )
vec2 = np.array( plt_df.u2 )
vec3 = np.array( plt_df.u3 )

fig  = plt.figure()
ax_x = fig.add_subplot( 111, projection='3d' )

plot_3d( ax_x,
         vec1,
         vec2,
         vec3,
         '$x^{1}$',
         '$x^{2}$',
         '$x^{3}$',
         cBeg1   = 1.5,
         cEnd2   = 1.1,
         cEnd3   = 1.2,
         vecInds = [ 700 ],
         vecLen  = 90  )

ax_x.scatter( vec1[700],
              vec2[700],
              vec3[700],
              c = 'r',
              depthshade = True,
              marker = 'o' )

ax_x.text( vec1[700],
           vec2[700],
           1.1 * vec3[700],
           '$p(t)$',
           fontsize = 16 )

plt.show()

# fig  = plt.figure()
# ax_u = fig.add_subplot( 111, projection='3d' )

# plot_3d( ax_u,
#          np.array( plt_df.u1 ),
#          np.array( plt_df.u2 ),
#          np.array( plt_df.u3 ),
#          '$u^{1}(t)$',
#          '$u^{2}(t)$',
#          '$u^{3}(t)$',
#          cBeg1 = 1.1,
#          cBeg3 = 1.1,
#          cEnd1 = 0.9,
#          cEnd2 = 0.8,
#          cEnd3 = 1.3   )

# plt.show()

# ***********************************************************************
# Generate 2d model plots
# ***********************************************************************

ecoMfd = dill.load( open( 'model.dill', 'rb' ) ).ecoMfd

plt.semilogy( ecoMfd.funcVals, 'k-o' )
plt.xlabel( 'Iterations' )
plt.ylabel( 'Gradient Norm' )
plt.show()

actSol    = ecoMfd.actSol
actOosSol = ecoMfd.actOosSol
odeObj    = ecoMfd.getSol( ecoMfd.GammaVec )
oosOdeObj = ecoMfd.getOosSol()
sol       = odeObj.getSol()
oosSol    = oosOdeObj.getSol()
stdVec    = ecoMfd.getConstStdVec()

minVals   = np.zeros( shape = (3), dtype = 'd' )
maxVals   = np.zeros( shape = (3), dtype = 'd' )

for m in range( ecoMfd.nDims ):
    minVals[m] = min( min( actSol[m] ),
                      min( actOosSol[m] ) )
    maxVals[m] = max( max( actSol[m] ),
                      max( actOosSol[m] ) )

normFunc  = lambda v, m: ( v - minVals[m] ) / \
    ( maxVals[m] - minVals[m] )

u1     = normFunc( sol[0], 0 )
u2     = normFunc( sol[1], 1 )
u3     = normFunc( sol[2], 2 )
uOos1  = normFunc( oosSol[0], 0 )
uOos2  = normFunc( oosSol[1], 1 )
uOos3  = normFunc( oosSol[2], 2 )

uAct1     = normFunc( actSol[0], 0 )
uAct2     = normFunc( actSol[1], 1 )
uAct3     = normFunc( actSol[2], 2 )
uActOos1  = normFunc( actOosSol[0], 0 )
uActOos2  = normFunc( actOosSol[1], 1 )
uActOos3  = normFunc( actOosSol[2], 2 )

nAllTimes = ecoMfd.nTimes + ecoMfd.nOosTimes 
timeInc = 1.0 / ecoMfd.nTimes

t = np.linspace( 0,
                 ecoMfd.nTimes * timeInc,
                 ecoMfd.nTimes )

tOos = np.linspace( ecoMfd.nTimes * timeInc,
                    nAllTimes * timeInc,
                    ecoMfd.nOosTimes )

plt.plot( t, uAct1, 'b-' )
plt.plot( tOos, uActOos1, 'b-' )
plt.plot( t, u1, 'r-', linewidth = 3 )
plt.plot( tOos, uOos1, 'y-', linewidth = 3 )
plt.xlabel( '$t$', fontsize = 22 )
plt.ylabel( '$u^{1}(t)$', fontsize = 22 )
plt.xticks( [0, 1], fontsize = 22 )
plt.yticks( [0, 1], fontsize = 22 )
plt.show()

plt.plot( t, uAct2, 'b-' )
plt.plot( tOos, uActOos2, 'b-' )
plt.plot( t, u2, 'r-', linewidth = 3 )
plt.plot( tOos, uOos2, 'y-', linewidth = 3 )
plt.xlabel( '$t$', fontsize = 22 )
plt.ylabel( '$u^{2}(t)$', fontsize = 22 )
plt.xticks( [0, 1], fontsize = 22 )
plt.yticks( [0, 1], fontsize = 22 )
plt.show()

plt.plot( t, uAct3, 'b-' )
plt.plot( tOos, uActOos3, 'b-' )
plt.plot( t, u3, 'r-', linewidth = 3 )
plt.plot( tOos, uOos3, 'y-', linewidth = 3 )
plt.xlabel( '$t$', fontsize = 22 )
plt.ylabel( '$u^{3}(t)$', fontsize = 22 )
plt.xticks( [0, 1], fontsize = 22 )
plt.yticks( [0, 1], fontsize = 22 )
plt.show()

# ***********************************************************************
# Generate 3d model plot
# ***********************************************************************

offset = 0.0

xAct1 = np.cumsum( uAct1 ) 
xAct2 = np.cumsum( uAct2 ) 
xAct3 = np.cumsum( uAct3 ) 
xAct1 = xAct1 - xAct1[-1] + offset
xAct2 = xAct2 - xAct2[-1] + offset
xAct3 = xAct3 - xAct3[-1] + offset

xActOos1 = np.cumsum( uActOos1 ) + offset
xActOos2 = np.cumsum( uActOos2 ) + offset
xActOos3 = np.cumsum( uActOos3 ) + offset

x1 = np.cumsum( u1 )
x2 = np.cumsum( u2 )
x3 = np.cumsum( u3 )
x1 = x1 - x1[-1] + offset
x2 = x2 - x2[-1] + offset
x3 = x3 - x3[-1] + offset

xOos1 = np.cumsum( uOos1 ) + offset
xOos2 = np.cumsum( uOos2 ) + offset
xOos3 = np.cumsum( uOos3 ) + offset

fig  = plt.figure()
ax_x = fig.add_subplot( 111, projection = '3d' )

plot_3d( ax_x,
         xAct1,
         xAct2,
         xAct3,
         '$x^{1}$',
         '$x^{2}$',
         '$x^{3}$',
         endFlag = True,
         textBeg = '$t = 0$',
         textEnd = '',             
         cBeg1   = 0.85,
         cEnd2   = 1.0,
         cEnd3   = 0.5,
         pColor  = 'k',
         lColor  = 'b-' )

ax_x.text( 0.0,
           0.0,
           -7000,
           '$t = 1$',
           fontsize = 16 )

plot_3d( ax_x,
         xActOos1,
         xActOos2,
         xActOos3,
         endFlag = False,
         lColor  = 'b-' )

plot_3d( ax_x,
         x1,
         x2,
         x3,
         endFlag = False,
         lColor  = 'r-' )

plot_3d( ax_x,
         xOos1,
         xOos2,
         xOos3,
         endFlag = False,
         lColor  = 'y-' )

plt.show()
