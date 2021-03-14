def main():
    # Data Table
    symbols_list = ['LQD','VCSH']
    descriptions = symbols_list
    sources_list = ['local'] * len( symbols_list )

    for i in range(0,len(symbols_list)):
        plotSymbol(symbols_list[i], sources_list[i], descriptions[i])

if __name__ == '__main__':
    main()