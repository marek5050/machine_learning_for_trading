import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_max_price(symbol):
    tb=pd.read_csv("daily/table_{}.csv".format(symbol),names=["date","sm1","open","high","low","close","volume"])
    return tb["close"].mean()

def plot_historical(symbol):
    df=pd.read_csv(symbol_to_path(symbol),
                   names=["date","sm1","open","high","low","close","volume"])
    print(df[["high","low"]].head())
    df[["high","low"]].plot()
    plt.show()

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir,"table_{}.csv".format(str(symbol)))

def get_base_dataframe(dates):
    df1=pd.DataFrame(index=dates)
    dfSPY=pd.read_csv("table_spy.csv",
                      # names=["Date","Open","High","Low","Close","Volume"],
                      dtype={'Close': np.float64},
                      usecols=["Date","Close"],
                      index_col="Date",
                      na_values=['nan'],
                      parse_dates=True)
    dfSPY=dfSPY.iloc[::-1]
    dfSPY=dfSPY.rename(columns={'Close':'SPY'})
    df1=df1.join(dfSPY,how='inner')
    return df1

def get_data(symbols,dates):
    df1=get_base_dataframe(dates)
    for symbol in symbols:
        df_temp=pd.read_csv(symbol_to_path(symbol),
                          # names=["Date","Open","High","Low","Close","Volume","Adj. Close","Adj. Volume"],
                          # header=0,
                          usecols=["Date","Close"],
                          index_col="Date",
                          na_values=['nan'],
                          parse_dates=True)
        df_temp=df_temp.iloc[::-1]
        df_temp = df_temp.rename(columns={'Close':symbol})
        df1=df1.join(df_temp)

    df1.ffill(inplace=True)
    df1.bfill(inplace=True)
    df1.fillna(0,inplace=True)
    return df1


def plot_selected(df,symbols,start,end):
    df1=df.ix[start:end,symbols]
    ax = df1.plot(title="Plot Selected", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def test_run():
    a = np.random.randint(0,10,size=(2,8))
    a=a.reshape(4,4)
    print(a.shape)
    print(a)
    print(a[0:2,0:3])
    a[0,:]=1
    a[:,1]=0
    a[1,0:3]=2
    print(a)
    mean = a.mean()
    print(mean)
    print(a*2.0)

def rolling_mean(df):
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')
    rm_SPY= pd.rolling_mean(df['SPY'],window=20)
    rm_SPY.plot(label='Rolling Mean', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.show()

def plot_rolling(df,mean, upper_band,lower_band):
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')
    mean.plot(label='Rolling Mean', ax=ax)
    upper_band.plot(label="Upper band", ax=ax)
    lower_band.plot(label="Lower band",ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.show()


def plot_data(df,title="Stock prices"):
    ax = df.plot(title=title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def get_rolling_mean(values, window):
    return pd.rolling_mean(values['SPY'],window=window)

def get_rolling_std(values,window):
    return pd.rolling_std(values['SPY'],window=window)

def get_bands(rolling_mean, rolling_std):
    return (rolling_mean+2*rolling_std, rolling_mean-2*rolling_std)

def compute_daily_returns(df):
    return (df / df.shift(1))-1

def compute_cumulative_returns(df):
    return df.div(df.ix[0],axis="columns")-1

def normalize(df):
    return df.div(df.ix[0],axis="columns")

def make_scatter_plot(df,x,y):
    ax=df.plot(kind='scatter',x=x, y=y)
    beta_XOM,alpha_XOM = np.polyfit(df[x],df[y],1)
    plt.plot(df[x],beta_XOM*df[x]+alpha_XOM,'-',color='r')

def single_stock():
    start='2014-01-01'
    end='2015-12-31'
    dates=pd.date_range(start,end)
    df1=get_data(['IBM'],dates)
    df_daily_1=compute_daily_returns(df1)
    df_daily_1.ffill(inplace=True)
    df_daily_1.bfill(inplace=True)

    df2=get_data(['XOM'],dates)
    df_daily_2=compute_daily_returns(df2)
    df_daily_2.ffill(inplace=True)
    df_daily_2.bfill(inplace=True)

    df3=get_data(['GOOG'],dates)
    df_daily_3=compute_daily_returns(df3)
    df_daily_3.ffill(inplace=True)
    df_daily_3.bfill(inplace=True)

    # mean = get_rolling_mean(df1,20)
    # std = get_rolling_std(df1,20)
    # upper_band,lower_band = get_bands(mean,std)
    # df_cum=compute_cumulative_returns(df1)
    make_scatter_plot(df_daily_1,"SPY","IBM")

    make_scatter_plot(df_daily_2,"SPY","XOM")
    make_scatter_plot(df_daily_3,"SPY","GOOG")

    plt.show()

def porfolio_return():
    start='2010-01-01'
    end='2013-12-31'
    dates=pd.date_range(start,end)
    symbols=['IBM','GOOGL','AAPL']
    allocs=[0.4,0.4,0.2]
    start_val=1000000
    df1=get_data(symbols,dates)

    del(df1['SPY'])
    df_norm=normalize(df1)
    df_alloced=df_norm.mul(allocs,axis='columns')
    pos_val=df_alloced*start_val
    port_val=pos_val.sum(axis=1)
    port_val.plot()
    plt.show()

def calculate_portfolio_return(symbols, allocs, start_val, dates):
    df1=get_data(symbols,dates)
    del(df1['SPY'])

    df_norm=normalize(df1)
    df_alloced=df_norm.mul(allocs,axis='columns')
    pos_val=df_alloced*start_val
    port_val=pos_val.sum(axis=1)
    return port_val

if __name__ == "__main__":
    start='2010-01-01'
    end='2013-12-31'
    dates=pd.date_range(start,end)
    start_val=1000000

    df_total=get_data([],dates)
    del(df_total['SPY'])

    symbols=['MSFT','GOOGL',"CSCO","FB"]
    allocs=[0.4,0.2,0.2,0.2]
    df_total[", ".join(symbols)]=calculate_portfolio_return(symbols,allocs, start_val,dates)

    df_total.plot()
    plt.title("Portfolio return",fontsize=15)
    plt.show()    # plot_data(df1,title="Closing")
    # plot_data(df_daily,title="Daily Returns")
    # plot_data(df_cum,title="Cummulative Returns")
    # mean= df_daily['spy'].mean()
    # std= df_daily['spy'].std()
    # print("MEAN: " , mean)
    # print("STD: ", std)
    # df_daily.hist()
    # plt.axvline(mean,color='w',linestyle='dashed',linewidth=2)
    # plt.axvline(-std,color='r',linestyle='dashed',linewidth=2)
    # plt.axvline(std,color='r',linestyle='dashed',linewidth=2)
    # plt.show()
    # rolling_mean(df1)
    # plot_rolling(df1,mean,upper_band,lower_band)
    # df1=get_data(['ibm','aapl','msft','csco','goog'],dates)
    # df2= df1 / df1.ix[0]
    # plot_selected(df2,['spy','ibm','aapl','msft','csco','goog'],start,end)
    # mean = df1.mean()
    # median = df1.median()
    # std = df1.std()
    # print("Mean: \n" , mean)
    # print("Median: \n", median)
    # print("STD: \n", std)
