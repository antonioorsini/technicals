__autor__   = 'Antonio Orsini'
__doc__     = 'Data visualization for a time series'

# Core
import pandas as pd
import numpy as np
import pylab
import time
import datetime as dt
import urllib.request, urllib.error, urllib.parse

# plot tools
import matplotlib
import matplotlib.pyplot as plt
from   matplotlib                 import dates as mdates
from   matplotlib                 import ticker as mticker
from   mplfinance.original_flavor import candlestick_ohlc
matplotlib.rcParams.update({'font.size': 9})

# Data Gathering Tools
pd.core.common.is_list_like = pd.api.types.is_list_like # momentary workaraound
import pandas_datareader.data as pdweb
import quandl
API_KEY = 'ftRcTLovsgiMN7Zy9ZT'
quandl.ApiConfig.api_key = API_KEY

# original modules
from marketdata.alpha_vantage import avrequest as avr
import technicals.indicators as ti
from   variables import path_single_series

def plotSymbol(symbol, source, description = None, start = None , end = None, time_delta = 180):
    """
    This function download data from different sources and creates a candlestick plot
    along with some main technical indicators for financial securities studies.
    """

    time_format = '%Y-%m-%d'

    # Set initial date
    tod_date = dt.date.today()
    if start is None: 
        start = tod_date - dt.timedelta(days=time_delta)
    if end is None:
        end = tod_date


    #===========================================================================
    #                             Data Gathering
    #===========================================================================

    if source == "google": # Almost Deprecated
        data = pdweb.DataReader(symbol, source, start = start, end = end)

    if source == 'fred':
        data = pdweb.DataReader(symbol, source, start = start, end = end)
        data.index.rename('Date', inplace=True)

    if source == 'local':
        data = pd.read_csv(path_single_series + '/' + symbol + '.csv' )

    if source == 'morningstar':
        data = pdweb.DataReader(symbol, source, start = start, end = end)
        data.index = data.index.droplevel(0)

    if source == "quandl":
        data = quandl.get(symbol, start_date = start, end_date = end)
        if 'Close' in data.columns: 1==1
        else:
            if 'Price' in data.columns:
                data = data.rename(columns={'Price': 'Close',})
                
    if source == 'alpha_vantage':
        data = avr(
            function = 'TIME_SERIES_DAILY_ADJUSTED',
            symbol = symbol,
            outputsize = 'full',
            datatype = None,
            interval = None)

        data = data.rename(columns={'close': 'Close','open': 'Open','high':'High','low':'Low','volume':'Volume'})
        data['Date'] = data.index
        data = data[['Date','Open','High','Low','Close','Volume']]
        #for col in ['adjusted close','dividend amount','split coefficient']:
        #    del data[col]
        
    # make sure the time period selection is homogenous 
    dates = pd.to_datetime(data['Date'].copy()).dt.date
    data = data[ (dates >= start) & (dates <= end) ]
    

    #===========================================================================
    #                          Start Plotting
    #===========================================================================

    data['Date']=[dt.datetime.strptime(i, time_format) for i in data['Date']]
    data['Date']=mdates.date2num(data['Date'])
    #8037 
    MA1 = 10 # Quick Moving Average
    MA2 = 30 # Slow Moving Average

    # SP will be used in order to leave out data, due to slow MA size and define
    # the overall scope of the graph.
    SP = len(data['Date'][MA2-1:])

    # Call the figure
    fig = plt.figure(facecolor='#07000d')
    #fig = plt.figure(facecolor='#FFFFFF')

    # Main Graph
    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, facecolor='#07000d')
    candlestick_ohlc(ax1,data[-SP:].values, width=.6, colorup='#53c156', colordown='#ff1717')

    # Plot the moving averages
    Av2 = ti.MA(data['Close'], MA2)
    Av1 = ti.MA(data['Close'], MA1)
    Label1 = str(MA1)+' SMA'
    Label2 = str(MA2)+' SMA'
    ax1.plot(data['Date'][-SP:] ,Av1[-SP:],'#e1edf9',label=Label1, linewidth=1.5)
    ax1.plot(data['Date'][-SP:] ,Av2[-SP:],'#4ee6fd',label=Label2, linewidth=1.5)

    # Plot Boollinger Bands
    bb_window = 20
    bbands = ti.BBANDS(data['Close'], bb_window)
    bbu, bbd = bbands['up'], bbands['down']
    label_bbu = 'BB up' + str(bb_window)
    label_bbd = 'BB down' + str(bb_window)
    ax1.plot(data['Date'][-SP:] ,bbu[-SP:],'#DF013A',label=label_bbu, linewidth=1, ls='dashed')
    ax1.plot(data['Date'][-SP:] ,bbd[-SP:],'#DF013A',label=label_bbd, linewidth=1, ls='dashed')

    # Plot Fibonacci Retracements
    retracements = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    fibo = ti.FIBO(data, lookback = SP, retracements = retracements)
    ax1.plot(data['Date'][-SP:], fibo.iloc[-SP:,0], '#DF013A',
        label='fibo '+str(retracements[0]), linewidth=1.2, ls='solid')
    ax1.plot(data['Date'][-SP:], fibo.iloc[-SP:,6], '#9AF412',
        label='fibo '+str(retracements[6]), linewidth=1.2, ls='solid')

    # Adjust the main graph
    ax1.grid(True, color='gray')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("w")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y', colors='w')
    ax1.tick_params(axis='x', colors='w')

    # plt.gca gets the current axes, creating one if needed.
    # It is only equivalent in the simplest 1 axes case
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    plt.ylabel('Stock price and Volume')

    # legend
    plot_legend = plt.legend(loc=2, ncol=1, prop={'size':7}, fancybox=True, borderaxespad=0.)
    plot_legend.get_frame().set_alpha(0.4)
    textEd = pylab.gca().get_legend().get_texts()
    pylab.setp(textEd[0:5], color = 'w')

    volumeMin = 0

    # RSI Upper Graph
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
    rsi = ti.RSI(data,14)
    rsiCol = '#c1f9f7'
    posCol = '#386d13'
    negCol = '#8f2020'
    ax0.plot(data['Date'][-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
    ax0.axhline(70, color=negCol)
    ax0.axhline(30, color=posCol)
    ax0.fill_between(data['Date'][-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>=70),
        facecolor=negCol, edgecolor=negCol, alpha=0.5)
    ax0.fill_between(data['Date'][-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<=30),
        facecolor=posCol, edgecolor=posCol, alpha=0.5)
    ax0.set_yticks([30,70])
    ax0.yaxis.label.set_color("w")
    ax0.spines['bottom'].set_color("#5998ff")
    ax0.spines['top'].set_color("#5998ff")
    ax0.spines['left'].set_color("#5998ff")
    ax0.spines['right'].set_color("#5998ff")
    ax0.tick_params(axis='y', colors='w')
    ax0.tick_params(axis='x', colors='w')
    plt.ylabel('RSI')

    # Volume graph underlying the candlestick
    ax1v = ax1.twinx()
    ax1v.fill_between(data['Date'][-SP:],volumeMin, data['Volume'][-SP:], facecolor='#00ffe8', alpha=.4)
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    # Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3*data['Volume'].max())
    ax1v.spines['bottom'].set_color("#5998ff")
    ax1v.spines['top'].set_color("#5998ff")
    ax1v.spines['left'].set_color("#5998ff")
    ax1v.spines['right'].set_color("#5998ff")
    ax1v.tick_params(axis='x', colors='w')
    ax1v.tick_params(axis='y', colors='w')


    # Stochasti Lower Graph
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
    fillcolor = '#00ffe8'

    sto_K, sto_D= ti.STO(data, 4)
    ax2.plot(data['Date'][-SP:], sto_K[-SP:], color='#4ee6fd', lw=2)
    ax2.plot(data['Date'][-SP:], sto_D[-SP:], color='#e1edf9', lw=1)

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='x', colors='w')
    ax2.tick_params(axis='y', colors='w')
    plt.ylabel('Stochastic', color='w')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    # # MACD Lower Graph
    # ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, facecolor='#07000d')
    # fillcolor = '#00ffe8'
    #
    # emaslow, emafast, macd, macd_signal, macd_diff = ti.MACD(data['Close'])
    # ax2.plot(data['Date'][-SP:], macd[-SP:], color='#4ee6fd', lw=2)
    # ax2.plot(data['Date'][-SP:], macd_signal[-SP:], color='#e1edf9', lw=1)
    # ax2.fill_between(data['Date'][-SP:], macd_diff[-SP:], 0,
    #   alpha=0.5, facecolor=fillcolor, edgecolor=fillcolor)
    #
    # plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    # ax2.spines['bottom'].set_color("#5998ff")
    # ax2.spines['top'].set_color("#5998ff")
    # ax2.spines['left'].set_color("#5998ff")
    # ax2.spines['right'].set_color("#5998ff")
    # ax2.tick_params(axis='x', colors='w')
    # ax2.tick_params(axis='y', colors='w')
    # plt.ylabel('MACD', color='w')
    # ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
    # for label in ax2.xaxis.get_ticklabels():
    #   label.set_rotation(45)

    # Suptitle
    if description is None: 
        description = symbol
    plt.suptitle(description + " up to " + str(mdates.num2date(data['Date'].values[-1]))[:10], color='w')

    # Adjustments to placing
    plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)

    # Maximize and plot, this will plot the graph full screen
    mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    plt.show()

