import pandas as pd

df = pd.read_pickle('visualization_daily_dfFile.pkl')

smoothCnt = 360 #* 19 * 60

df['SPY_smooth'] = df['SPY'].rolling(smoothCnt,
                                     win_type='blackman',
                                     center=True).mean()
df['TYX_smooth'] = df['TYX'].rolling(smoothCnt,
                                     win_type='blackman',
                                     center=True).mean()
df['OIH_smooth'] = df['OIH'].rolling(smoothCnt,
                                     win_type='blackman',
                                     center=True).mean()

df.to_pickle('visualization_dfFile_smooth.pkl')
