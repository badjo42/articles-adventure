#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Joël Repond

"""

#%% IMPORTS
# --------------------------
#   IMPORTS
# -----------------

# general
# -------
import pandas as pd
import mplfinance as mpf
from matplotlib.ticker import AutoMinorLocator

# Exchanges
# ----------
import ccxt

# Technical Analysis
# -------------------
from talib import SMA


#%%  FUNCTIONS DEFINITION
# --------------------------
#   FUNCTIONS DEFINITION
# -----------------

def _time_scaling(timeframe):
    '''
    Parameters
    ----------
    timeframe : string
        possible value '1m', '3m', '5m','15m', '30m', '1h', '2h', '4h', '6h',
                        '8h', '12h' ,'1d', '3d', '1w'

    Returns
    -------
    int that gives the time scaling in seconds corresponding to the timeframe
    defined. Used to load the prices from exchanges.

    '''

    switch = {
            '1m': 60,
            '3m': 60 * 3,
            '5m': 60 * 5,
            '15m': 60 * 15,
            '30m': 60 * 30,
            '1h': 60 * 60,
            '2h': 60 * 60 * 2,
            '4h': 60 * 60 * 4,
            '6h': 60 * 60 * 6,
            '8h': 60 * 60 * 8,
            '12h': 60 * 60 * 12,
            '1d': 60 * 60 * 24,
            '3d': 60 * 60 * 24 * 3,
            '1w': 60 * 60 * 24 * 7,
            }
    if timeframe in switch.keys():
        return int(switch.get(timeframe))
    else:
        raise ValueError('Timeframe not recognized')

def floor_datetime(to_datetime, timeframe):
    switch = {
            '1m': 'min',
            '3m': 'min',
            '5m': 'min',
            '15m': 'min',
            '30m': '30min',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': 'H',
            '8h': 'H',
            '12h': '12H',
            '1d': 'D',
            '3d': 'D',
            '1w': 'D',
            }
    if timeframe in switch.keys():
        return pd.Timestamp.floor(to_datetime, switch.get(timeframe)) # M, D, H, min, S, ms, us, N
    else:
        raise ValueError('Timeframe not recognized')


def load_ohlcv_binance(pair, from_datetime, timeframe):
    '''
    Parameters
    ----------
    pair : string
        String like XRP/USDT or BTC/ETH.
    from_datetime : pandas Timestamp
        Starting date of the data. Usually can be also a datetime.datetime object
    timeframe : string
        The timeframe that can be '1m', '3m', '5m','15m','30m','1h',
        '2h', '4h', '6h', '8h', '12h' ,'1d', '3d', '1w'

    Returns
    -------
    Panda DataFrame with datetime as index and ohlcv data.

    '''
    # The rate limiter is disabled by default
    # Enable since exchanges API have in general a maximum number of request you can do per minutes
    binance = ccxt.binance({'enableRateLimit': True})
    binance.rateLimit = 200 #(= 2000 by default) = delay in milliseconds between two consecutive requests.
        
    from_timestamp = int(binance.parse8601(timestamp=from_datetime.strftime("%Y-%m-%d %H:%M:%S"))/1000)
    exchange_time = binance.seconds()
    time_scaling = _time_scaling(timeframe)

    data = []
    while exchange_time - from_timestamp > time_scaling:
        print(from_timestamp, exchange_time)
        print(f"Gathering data for {pair}")
        ohlcv = binance.fetch_ohlcv(pair, timeframe, since = from_timestamp*1000, limit=500)
        from_timestamp += len(ohlcv) * time_scaling
        data += ohlcv
        print(from_timestamp, exchange_time, len(ohlcv), time_scaling)

    data = pd.DataFrame(data,columns=['datetime','open','high','low','close','volume'])
    data['datetime'] = pd.to_datetime(data['datetime'],unit='ms')
    data.set_index('datetime',inplace=True)

    return data


def file_conformity(path_file, from_datetime, to_datetime):
    conformity = False
    try:
        df = pd.read_pickle(path_file, compression='gzip')
    except:
        return conformity

    if df.index[0] <= from_datetime and df.index[-1] >= to_datetime:
        conformity = True
    return conformity

    
def get_ohlcv_binance(pair, from_datetime, timeframe, to_datetime):
    '''Download and cache Binance dataseries'''
    
    cache_file = f'{pair}_{timeframe}.pkl'.replace('/','_')
    path = './data'
    
    if file_conformity(path+'/'+cache_file, from_datetime, to_datetime):
        df = pd.read_pickle(path + '/' + cache_file, compression='gzip')
        print(f'{cache_file} file loaded from local')
    else:
        print(f'Downloading {cache_file} from Binance')
        df = load_ohlcv_binance(pair, from_datetime, timeframe)
        df.to_pickle('./data/'+cache_file,compression='gzip')
        print(f'Cached at {cache_file}')

    mask = (df.index >= from_datetime)*(df.index <= to_datetime)
    return df[mask]


# %% CODE
# -------------
# CODE
# --------

pair = 'BTC/USDT'

# ['1m', '3m', '5m','15m','30m','1h', '2h', '4h', '6h', '8h', '12h' ,'1d', '3d', '1w']
timeframe = '1d'

from_datetime = pd.Timestamp('2019-09-26 00:00:00')
#to_datetime = pd.Timestamp('2020-09-26 11:00:00')
to_datetime = floor_datetime(pd.Timestamp.today(), timeframe)


data_ohlcv = get_ohlcv_binance(pair, from_datetime, timeframe, to_datetime)

with pd.option_context('display.max_rows', 10, 'display.max_columns', 10):
    print(data_ohlcv.head())
    print(data_ohlcv.tail())


#%% PLOT
value_SMA9 = SMA(data_ohlcv['close'], timeperiod=9)
value_SMA20 = SMA(data_ohlcv['close'], timeperiod=20)
value_SMA50 = SMA(data_ohlcv['close'], timeperiod=50)

addplots = [
            mpf.make_addplot(value_SMA9, type='line', panel=0),
            mpf.make_addplot(value_SMA20, type='line', panel=0),
            mpf.make_addplot(value_SMA50, type='line', panel=0),
            ]

fig,ax = mpf.plot(data_ohlcv, type='candle', volume=True,show_nontrading=True, addplot=addplots,
         main_panel=0,volume_panel=1, num_panels=2, panel_ratios=(3,1), style='default',
         returnfig=True)

ax[0].legend(('SMA 9', 'SMA 20', 'SMA 50'), frameon=True, loc='best')
# ax[0].set_xlim((from_datetime,to_datetime))

# ax[2].set_xlabel('testx', fontsize=16)
# ax[0].set_ylabel('testy', fontsize=16)



# ax[0].spines['top'].set_linewidth(2)                
# ax[0].spines['right'].set_linewidth(2)
# ax[0].spines['bottom'].set_linewidth(0.7)
# ax[0].spines['left'].set_linewidth(2)

# ax[2].spines['top'].set_linewidth(0.7)                
# ax[2].spines['right'].set_linewidth(2)
# ax[2].spines['bottom'].set_linewidth(2)
# ax[2].spines['left'].set_linewidth(2)

# ax[2].tick_params(axis='x', labelsize=16)
# ax[0].tick_params(axis='y', labelsize=16)

# ax[2].xaxis.set_minor_locator(AutoMinorLocator(n=5))
# ax[0].yaxis.set_minor_locator(AutoMinorLocator(n=5))
# ax[2].tick_params(which='major',axis='x',color='k',length=6,width=1.5)
# ax[0].tick_params(which='major',axis='y',color='k',length=6,width=1.5)

#idea : create my own timeframe
# axes.yaxis.major.formatter._useMathText = True
# axes.xaxis.major.formatter._useMathText = True
# axes.ticklabel_format(style='sci', axis='both', scilimits=scilimits)

# plt.grid('on')
# plt.title(v_double + ' RF - '+str(voltage_program_1)+' - '+impedance+' impedance (Color: Amplitude)')
        
# fig.subplots_adjust(left=0.15, bottom=0.13, right=0.96, top=0.96, wspace=0.4, hspace=0.3)
# fig.tight_layout()

fig.show()
fig.savefig('./data/btc_candlestick.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
#fig.savefig('./data/btc_candlestick.png', dpi=600, bbox_inches='tight', pad_inches=0.03, transparent=True)


