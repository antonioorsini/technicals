def main():
    ''' test '''
    import technicals.indicators as idc
    import matplotlib.pyplot as plt    
    import pandas as pd

    from misc.variables import path_single_series

    data = pd.read_csv(path_single_series + '/SPY.csv')    

    idc.RSI(data['Close'], n = 20).plot()
    plt.show()

if __name__ == '__main__':
    main()
    