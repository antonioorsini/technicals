__author__  = 'Antonio Orsini'
__doc__     = 'Technical indicators for financial time series.'

# general modules
import pandas as pd
import math   as m
import numpy  as np
import six

# =====================================================================
# Utilities
# =====================================================================
def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e

def take_diff(xs, n):
    if n >= 0:
        e = np.diff(xs, n)
        x = np.empty(n)
        x.fill(np.nan)
        e = np.insert(e, 0, values = x)
    else:
        e = np.diff(xs, -n)
        x = np.empty(-n)
        x.fill(np.nan)
        e = np.insert(e, -1, values = x)
    return e

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.to_numpy().strides + (a.to_numpy().strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# =====================================================================
# Indicators tested and working
# =====================================================================

# Moving Average
def MA(series, window):
    weigths = np.repeat(1.0, window)/window
    sma = np.convolve(series, weigths, 'valid')
    return sma # as a numpy array

# Exponential Moving Average
def EWMA(series, window, discount_factor = 0.94):
    weights = np.exp(np.linspace(1-discount_factor, 0., window))
    weights /= weights.sum()
    ewma =  np.convolve(series, weights, mode='full')[:len(series)]
    ewma[:window] = ewma[window]
    return ewma

# Moving Average Convergence Divergence
def MACD(series, slow=24, fast=12):
    macd = pd.DataFrame({'Series':series})
    macd['EWMA_Slow']   = macd['Series'].ewm(slow).mean()
    macd['EWMA_Fast']   = macd['Series'].ewm(fast).mean()
    macd['MACD']        = macd['EWMA_Fast'] - macd['EWMA_Slow']
    macd['MACD_Signal'] = macd['MACD'].ewm(9).mean()
    macd['MACD_Diff']   = macd['MACD'] -macd['MACD_Signal']
    return macd

# Bollinger Bands
def BBANDS( series, window ):
    ma    = series.rolling( window ).mean()
    msd   = series.ewm( window ).std()
    bup   = ma + msd * 2
    bdown = ma - msd * 2
    return {'up':bup, 'down':bdown }

# Momentum
def MOM(series, n): # series needs to be a pandas series
    mom = np.diff(series, n)
    return mom

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    return SOk

#Stochastic oscillator %D
def STOD(df, n):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    SOd = SOk.ewm(span = n, min_periods = n - 1).mean()
    SOd.name = 'SOd_' + str(n)
    return SOd

#Rate of Change
def ROC(series, window):
    M = take_diff(series, -(window - 1))
    N = shift(series, -(window - 1))
    roc = M / N
    return roc

# Relative Strength Index
def RSI( series, n = 14 ):
    ''' 
    This fuction calculates the RSI technical indicator, if formula 1 is used then,
    the only data needed are a series of closing prices from which up and down movement can be
    recursively calculated. If instead formula 2 is used, a comprehensive dataset of ochl
    data series is used, and highs and lows are taken directly from it.
    '''

    diff = series.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg   = 0 * diff
    down_chg = 0 * diff
    
    up_chg[diff > 0]   = diff[ diff>0 ]        
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    up_chg_avg   = up_chg.ewm(n).mean()
    down_chg_avg = down_chg.ewm(n).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi

# Pivot Points, Supports and Resistances
def PPSR(df, method = 1):
    if method == 1:
        PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    if method == 2:
        PP = pd.Series((df['Open'] + \
        pd.Series(df['High'] + df['Low'] + df['Close']).shift(-1)) / 4)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    return PSR

# Fibonacci Retracements
def FIBO(df, lookback = 90, retracements = None):
    ''' This function calculates the Fibonacci Retracements of a dataset containing
    Open and High prices for the day, or in alternative either a dataframe with only the
    Close prices either in a pandas format or as a numpy array. The inputs are the lookback
    period for the retracements and the retracements itself to be used.'''

    if retracements == None: 
        retracements = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    if type(df) == type(np.array([1,2])):
        dfmax = df.max()
        dfmin = df.min()

    elif 'High' in df.columns and 'Low' in df.columns:
        dfmax = df['High'][-lookback:].max()
        dfmin = df['Low'][-lookback:].min()

    elif 'Close' in df.columns and 'High' not in df.columns and 'Low' not in df.columns:
        dfmax = df['Close'][-lookback:].max()
        dfmin = df['Close'][-lookback:].min()

    fibo = pd.DataFrame()
    for i in retracements:
        fibo_i = np.empty(len(df))
        fibo_i[:] = dfmin + (dfmax - dfmin)*i
        fibo[str(i)] = fibo_i
    return(fibo)

def FIBORolling(df, lookback = 90, retracements = None):
    ''' Calculates Fibonacci at any point in time '''
    
    if retracements == None: 
        retracements = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

    if   ('High' in df.columns) and ('Low' in df.columns):
        df['Max'] = df['High'].rolling(lookback).max()
        df['Min'] = df['Low'].rolling(lookback).min()

    elif ('Close' in df.columns) and ('High' not in df.columns and 'Low' not in df.columns):
        df['Max'] = df['Close'].rolling(lookback).max()
        df['Min'] = df['Close'].rolling(lookback).min()

    fibo = pd.DataFrame( index = df.index)

    for i in retracements:
        fibo[str(i)] = df['Min'] + (df['Max'] - df['Min'])*i

    return(fibo)

# Average True Range
def TR( row ):
    ''' gets the true range for a day, formula according to investopedia '''
    tr1 = row['High'] - row['Low']
    tr2 = abs( row['High'] - row['Close'] )
    tr3 = abs( row['Low'] - row['Close'] )
    return max( tr1, tr2, tr3 )

def ATR(df, n, averaging = 'aritmetic'):
    ''' Returns the average true range '''
    df['TR'] = df.apply( TR, axis = 1 )
    if averaging == 'aritmetic':      
        return df['TR'].rolling( n ).mean()
    elif averaging == 'ewm':
        return df['TR'].ewm( n ).mean()

#Average Directional Movement Index
def ADX(df, n):
    ''' ADX indicates the strength of a trend, not its direction '''
    df['ATR'] = ATR(df, n, averaging = 'ewm') # getting the ATR which is used as a standardization factor

    df['HighChg'] = df['High'].diff(1)
    df['LowChg']  = df['Low'].diff(1)

    # creating the Directional Indicators of Trend Strenghts
    # the rationale is: if the high gap is larger, there is more strenght in buying trend
    def posDI(row):
        if row['HighChg'] > row['LowChg']: return max( row['HighChg'], 0 )
        else: return 0

    def negDI(row):
        if row['LowChg'] > row['HighChg']: return max( row['LowChg'], 0 )
        else: return 0

    df['PosDI'] = df.apply( posDI, axis = 1 ) 
    df['NegDI'] = df.apply( negDI, axis = 1 )

    # smothing and standardizing the DIs
    df['PosDI'] = ( df['PosDI'].ewm(n).mean() / df['ATR'] ) * 100
    df['NegDI'] = ( df['NegDI'].ewm(n).mean() / df['ATR'] ) * 100

    df['DX'] = ( abs(df['PosDI'] - df['NegDI']) / abs(df['PosDI'] + df['NegDI']) )* 100

    # smooth the DX to get the ADX
    df['ADX'] = df['DX'].ewm(n).mean() 

    return df['ADX']

def TRIX( series, n ):
    ''' TRIX oscillator, value above zero are buy and uptrending momentum '''
    ex1 = series.ewm(n).mean()
    ex2 = ex1.ewm(n).mean()
    ex3 = ex2.ewm(n).mean()
    return ex3.pct_change()

#Mass Index
def MassI( df ):
    ''' Mass Index. Look for a bulge (indicator up and then down) around the level of 26/27 for a trend reversal '''
    range = df['High'] - df['Low']
    ex1 = range.ewm(9).mean()
    ex2 = ex1.ewm(9).mean()
    mass_i = (ex1/ex2).rolling(25).sum()
    mass_i.name = 'MassIndex'
    return mass_i

#Chaikin Oscillator
def Chaikin(df):
    ''' Oscillator '''
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close']))/(df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    adl = money_flow_volume.cumsum()
    chaikin = adl.ewm(3).mean() - adl.ewm(10).mean()
    chaikin.name = 'Chaikin'
    return chaikin

def MFI( df ):
    ''' Oscillator '''
    df['Typical_Price']    = (df['High'] + df['Low'] + df['Close'])/3
    df['Raw_Money_Flow']   = df['Typical_Price'] * df['Volume']
    df['Raw_Money_Flow']   = df['Raw_Money_Flow'] * np.sign(df['Close'].diff(1))
    df['Neg_Money_Flow']   = abs( df['Raw_Money_Flow'].apply(lambda x: min(x,0)) ).rolling(14).sum()
    df['Pos_Money_Flow']   = abs( df['Raw_Money_Flow'].apply(lambda x: max(x,0)) ).rolling(14).sum()
    df['Money_Flow_Ratio'] = df['Pos_Money_Flow'] / df['Neg_Money_Flow']
    df['Money_Flow_Index'] = 100 - 100 / ( 1 + df['Money_Flow_Ratio'] ) 
    return df['Money_Flow_Index']


################################################################################
#                            Indicators to fix
################################################################################

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))
    df = df.join(VI)
    return df

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df = df.join(KST)
    return df

#True Strength Index
def TSI(df, r = 25, s = 13):
    M     = pd.Series( df['Close'].diff(1) )
    aM    = abs( M )
    EMA1  = pd.Series( pd.ewma(M    , span = r, min_periods = r - 1) )
    aEMA1 = pd.Series( pd.ewma(aM   , span = r, min_periods = r - 1) )
    EMA2  = pd.Series( pd.ewma(EMA1 , span = s, min_periods = s - 1) )
    aEMA2 = pd.Series( pd.ewma(aEMA1, span = s, min_periods = s - 1) )
    TSI   = pd.Series( EMA2 / aEMA2 , name = 'TSI_' + str(r) + '_' + str(s) )
    df    = df.join( TSI )
    return df

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    df = df.join(AD)
    return df

#On-balance Volume
def OBV(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:
            OBV.append(df.get_value(i + 1, 'Volume'))
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:
            OBV.append(0)
        if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:
            OBV.append(-df.get_value(i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

#Force Index
def FORCE(df, n):
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))
    df = df.join(F)
    return df

#Ease of Movement
def EOM(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))
    df = df.join(Eom_ma)
    return df

#Commodity Channel Index
def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))
    df = df.join(CCI)
    return df

#Coppock Curve
def COPP(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))
    df = df.join(Copp)
    return df

#Keltner Channel
def KELCH(df, n):
    KelChM = pd.Series(pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = pd.Series(pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = pd.Series(pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))
    df = df.join(KelChM)
    df = df.join(KelChU)
    df = df.join(KelChD)
    return df

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    df = df.join(UltO)
    return df

#Donchian Channel
def DONCH(df, n):
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    df = df.join(DonCh)
    return df
