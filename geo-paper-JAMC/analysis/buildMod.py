# ***********************************************************************
# Import libraries
# ***********************************************************************

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../../opti-trade' ) )

from mod.mfdMod import MfdMod

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES

# ***********************************************************************
# Set some parameters and read data
# ***********************************************************************

dfFile      = 'random_signals.pkl'

minTrnDate  = pd.to_datetime( '2004-01-06 09:00:00' )
maxTrnDate  = pd.to_datetime( '2004-03-15 19:39:00' )
maxOosDate  = pd.to_datetime( '2004-03-29 16:59:00' )

velNames    = [ 'y1', 'y2', 'y3' ] 

selParams = { 'inVelNames' : [ 'SPY' ],
              'maxNumVars' : 3,
              'minImprov'  : 0.10,
              'strategy'   : 'forward' }

modFileName = 'model.dill'

# ***********************************************************************
# Build model
# ***********************************************************************

mfdMod = MfdMod(    dfFile       = dfFile,
                    minTrnDate   = minTrnDate,
                    maxTrnDate   = maxTrnDate,
                    maxOosDate   = maxOosDate,
                    velNames     = velNames,
                    maxOptItrs   = 100,
                    optGTol      = 1.0e-12,
                    optFTol      = 1.0e-12,
                    factor       = 1.0e-4,
                    regCoef      = 1.0e-6,
                    selParams    = None,
                    smoothCount  = None,
                    logFileName  = None,
                    verbose      = 1          )

validFlag = mfdMod.build()

print( 'Success :', validFlag )

mfdMod.save( modFileName )
mfdMod.ecoMfd.pltResults()


