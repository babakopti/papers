import sys
import os
import numpy as np
import pandas as pd

sys.path.append( os.path.abspath( '../../opti-trade' ) )

import utl.utils as utl

from dat.assets import OPTION_ETFS as ETFS
from dat.assets import FUTURES, INDEXES

piDatDir   = '/Users/babak/workarea/data/pitrading_data'
symbols    = [ 'SPY', 'TYX', 'OIH' ]

symbols = ETFS + INDEXES

piDf = utl.mergePiSymbols( symbols = symbols,
                           datDir  = piDatDir,
                           minDate = None,
                           logger  = None     )
piDf.to_pickle( 'visualization_dfFile_all.pkl' )
