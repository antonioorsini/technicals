__author_  = 'Antonio Orsini'
__doc__    = '''The object in this module will use indicators to extract intelligible descriptive signals about a financial time series '''

import technicals.indicators as ti
import six
import pandas as pd
import numpy  as np

class SignalsExtractor( object ):
    ''' 
    class to extract signals from oputput of indicators
    the output is always a list containg buy and sell signlas
    '''    
    def __init__( self, data ):
        
        self.data            = data

        self.sign_value_dict = {} 
        
        if 'Adj. Close' in data.columns:
            self.col_cl     = 'Adj. Close'
        else: 
            self.col_cl     = 'Close'

        self.col_dt     = 'Date'
        self.col_op     = 'Open'
        self.col_hi     = 'High'
        self.col_vl     = 'Volume'
        self.col_lo     = 'Low'

        self.prices_col = self.col_cl

    def getSeries( self, data ):
        return data[ self.prices_col ]

    def countSignals( self, last_n_days, subset_original_value = True ):
        self.sign_count_dict = {'buy':0,'sell':0}
        for indicator, signals in list(six.iteritems(self.sign_value_dict)):
            signals = signals[-last_n_days:]
            if subset_original_value == True: # replace with shorter signals series
                self.sign_value_dict[indicator] = signals
            if   any(x in signals for x in ['buy','oversold'])   : self.sign_count_dict['buy']  += 1.0
            elif any(x in signals for x in ['sell','overbought']): self.sign_count_dict['sell'] += 1.0
            elif any(x in signals for x in ['uptrending'])       : self.sign_count_dict['buy']  += 0.3
            elif any(x in signals for x in ['downtrending'])     : self.sign_count_dict['sell'] += 0.3        
        self.sign_count_dict['score'] = self.sign_count_dict['buy'] - self.sign_count_dict['sell']
        return self.sign_count_dict
    
    @staticmethod
    def _extractCrossingOscillatorSignalRow( row ):
        if   row['Signal_Chg']:
            if   row['Signal'] == True : return 'buy'
            elif row['Signal'] == False: return 'sell'
        elif (row['Value_Chg'] < 0) and (row['Value'] < 0): return 'downtrending'
        elif (row['Value_Chg'] > 0) and (row['Value'] > 0): return 'uptrending'

    def _extractCrossingOscillatorSignal( self, signal_series, value_series = None ):
        if value_series is None:
            value_series = signal_series.copy( deep = True )
        df = pd.DataFrame({'Signal':signal_series,'Value':value_series})
        df['Signal']     = -np.signbit(df['Signal'])
        df['Signal_Chg'] =  np.insert( 
            arr = np.diff(df['Signal'], 1), obj = [0], values = [0] )
        df['Value_Chg']  =  df['Value'].diff(1)
        return df.apply( self._extractCrossingOscillatorSignalRow, axis = 1 )

    @staticmethod
    def _extract2BoundariesOscillatorSignalRow( x, lower_boundary, higher_boundary ):
        if x <= lower_boundary:
            return 'oversold'
        elif x >= higher_boundary:
            return 'overbought'
        else:
            return None
 
    def _extract2BoundariesOscillatorSignal( self, series, lower_boundary = 20, higher_boundary = 80):
        return series.apply(self._extract2BoundariesOscillatorSignalRow, lower_boundary = lower_boundary, higher_boundary = higher_boundary)

    # ======================================================================================
    # Signals ==============================================================================
    # ======================================================================================
    def signMassI( self, bulge_threshold = 26.5, lapse_n = 15 ):
        mass_i = ti.MassI( self.data )
        mass_i = mass_i.apply( lambda x: 1 if x >= bulge_threshold else 0 )
        mass_i = mass_i.diff(1)
        mass_i_list =  mass_i.tolist()
        signals = []
        for i, x in enumerate(mass_i_list):            
            signal = None
            if x == -1: # -1 means the threshold was crossed from above
                #check for bulge in the timeframe specified under lapse_n
                if any(x == 1 for x in mass_i_list[i-lapse_n:i]): # 1 means that the threshold was crossed from below 
                    signal = 'reversal'
            signals.append(signal)
        signals = pd.Series( signals, index = mass_i.index )
        self.sign_value_dict['MassI'] = signals
        return signals

    def signMACD( self, slow = 24, fast = 12 ):
        macd = ti.MACD( self.getSeries( self.data ), slow = slow, fast = fast)
        signals = self._extractCrossingOscillatorSignal( signal_series = macd['MACD_Diff'], value_series = macd['MACD'] )
        self.sign_value_dict['MACD'] = signals
        return signals

    def signChaikin( self ):
        chaikin = ti.Chaikin( self.data )
        signals = self._extractCrossingOscillatorSignal( signal_series = chaikin )
        self.sign_value_dict['Chaikin'] = signals
        return signals

    def signTRIX( self, n = 14 ):
        trix = pd.DataFrame({'TRIX':ti.TRIX( self.getSeries( self.data ), n = n )})
        signals = self._extractCrossingOscillatorSignal( signal_series = trix['TRIX'] )
        self.sign_value_dict['TRIX'] = signals
        return signals

    def signRSI( self, n=14, sensitivity = 20 ):
        rsi = ti.RSI( self.getSeries(self.data), n )        
        signals = self._extract2BoundariesOscillatorSignal( rsi, 0+sensitivity, 100-sensitivity )
        self.sign_value_dict['RSI'] = signals
        return signals

    def signMFI( self, sensitivity = 20):        
        mfi = ti.MFI( self.data )
        signals = self._extract2BoundariesOscillatorSignal( mfi, 0+sensitivity, 100-sensitivity )
        self.sign_value_dict['MFI'] = signals
        return signals

    def signSTOD( self, n=14, sensitivity = 20):        
        stod = ti.STOD( self.data, n )        
        signals = self._extract2BoundariesOscillatorSignal( stod, 0+sensitivity, 100-sensitivity )
        self.sign_value_dict['STOD'] = signals
        return signals

    def signBBANDS( self, window = 15, proximity_parameter = 6 ):        
        sers = self.getSeries(self.data)
        bbands = ti.BBANDS( sers, window )
        bbu, bbd = bbands['up'], bbands['down']
        proximity_area = ((bbu - bbd)/proximity_parameter) # define the space needed to generate a signal
        signals_buy  = (sers >= (bbu - proximity_area) ).replace([True,False],['overbought',''])
        signals_sell = (sers <= (bbd + proximity_area) ).replace([True,False],['oversold' ,''])
        signals = (signals_buy + signals_sell).replace('',None)
        self.sign_value_dict['BBANDS'] = signals
        return signals        

    def signADX( self, n = 14 ):        
        def evaluateADX( x ):
            if   x >= 25: return 'trending'
            elif x >= 50: return 'strongtrend'
            else: return np.nan
        adx = ti.ADX( self.data, n = n )
        signal = adx.apply( evaluateADX )
        self.sign_value_dict['ADX'] = signal
        return signals

    # ======================================================================================
    # Indicat ==============================================================================
    # ======================================================================================

    def indicMovingAverageMinusPrice( self, period ):
        ma = ti.MA( self.data[self.col_cl], period )
        ma = pd.Series( ma, index = self.data.index[-len(ma):] )
        ma.name = 'movavg'
        maset  = pd.concat( [self.data[self.col_cl], ma], axis = 1 )
        maset = maset[period:]
        return (maset[self.col_cl] - maset['movavg'])/maset['movavg']

    def indicMACD( self, slow, fast ):
        macd = ti.MACD( self.data[self.col_cl], slow = slow, fast = fast )['MACD_Signal']
        macd = pd.Series( macd, index = self.data.index[-len(macd):] )
        return macd

    def indicBBANDS( self, window, band = 'up' ):
        bb      = ti.BBANDS( self.data[self.col_cl], window )[band]
        bbname  = 'bband_' + band
        bb.name = bbname
        bbset   = pd.concat( [self.data[self.col_cl], bb], axis = 1 )
        bbset   = bbset[window:]
        return (bbset[bbname] - bbset[self.col_cl])/bbset[self.col_cl]

    def indicRSI( self, n ):
        rsi = ti.RSI( self.data[self.col_cl], n = n )
        rsi = pd.Series( rsi, index = self.data.index[-len(rsi):] )
        return rsi

    def indicReturns( self, period = 1 ):
        ''' Take n period days returns '''
        return self.data[self.col_cl].copy( deep = True ).pct_change( period )

    def indicFIBO( self, lookback = 90, retracement = 0 ):
        ''' takes fibo retracement and creates a distance to current price '''
        fibo = ti.FIBORolling( self.data, lookback = lookback, retracements = [retracement] ).iloc[:,0]
        fibo_distance = (self.data['Close'] - fibo) / fibo
        return fibo_distance

    def indicATR( self, n = 14, averaging = 'aritmetic' ):
        return ti.ATR( self.data, n = n, averaging = averaging )[n:]

    def indicClose( self ):
        return self.data[self.col_cl]

    # ======================================================================================
    # targets ==============================================================================
    # ======================================================================================

    @staticmethod
    def createCathegoricalWinLoss( x ):
        if x >= 0.03:
            return 2
        elif x < 0.0:
            return 1
        else:
            return 0

    def indicReturnsAndShift( self, period, shiftn ):
        ''' take n period days returns and shift dataset to creat a look into future period '''            
        return self.data[self.col_cl].copy( deep = True ).pct_change( period ).shift( shiftn )

    def signalReturnsAndShiftCathegorical( self, period, shiftn ):
        ''' take n period days returns and shift idx ahead to create a look into future period '''            
        data = self.indicReturnsAndShift( period = period, shiftn = shiftn )
        data = data.apply( self.createCathegoricalWinLoss )
        return data
