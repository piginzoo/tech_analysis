from scipy.signal import argrelmax, argrelmin
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
from data_loader import load_stock
from utils import str2date

# python -m test.test_stock_kr
code = '600196'
start_date = '20080101'
end_date = '20240101'

df = load_stock(code)
start = str2date(start_date)
end = str2date(end_date)
df = df[(df.index > start) & (df.index < end)]
print(df)

idx = np.arange(len(df))
model = KernelReg(df.close, idx, 'c', reg_type='ll', bw='cv_ls')

# 请多平滑后的高低点
model.bw = model.bw * 8
close_pred, _ = model.fit(idx)
max_index = argrelmax(close_pred)[0]
min_index = argrelmin(close_pred)[0]
print("最大值索引：", max_index)
minmax_index = np.concatenate([max_index, min_index])
minmax_index.sort()
df_maxmin = df.iloc[minmax_index]


def plot():
    # 画图
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from pandas.plotting import table
    import mplfinance as mpf
    fig = plt.figure(figsize=(50, 200))
    ax = fig.add_subplot(1, 1, 1)
    mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
    s = mpf.make_mpf_style(gridaxis='both', gridstyle='-.', y_on_right=False, marketcolors=mc)
    kwargs = dict(type='candle', volume=False, title='基金的走势', ylabel='K线', ylabel_lower='')  # ,figratio=(15, 10)
    mpf.plot(df, ax=ax, style=s, type='candle', show_nontrading=True)
    mng = plt.get_current_fig_manager()
    ax.plot(df.index, df.close, color='red', linestyle='--', linewidth=1)
    ax.plot(df_maxmin.index, df_maxmin.close, color='#6495ED', linewidth=2)
    ax.plot(df.index, close_pred, color='#6495ED', linestyle='--', linewidth=1)
    ax.scatter(df_maxmin.index, df_maxmin.close, color='red', linewidths=5)
    fig.tight_layout()
    mng.full_screen_toggle()
    plt.show()


def is_5_waves(df, minmaxs, today):
    """
    是否是5浪？
    :param df: 数据
    :param minmaxs: 最大最小值的索引:int
    :param today_iloc: 今天的日期:date
    :return:
    """
    today_iloc = df.index.get_loc(today)
    past_minmaxs = minmaxs[minmaxs < today_iloc]
    last_5_minmaxs = past_minmaxs[:5]

    if len(last_5_minmaxs)!=5:
        print("[X] 没有5个高低点")
        return False

    c1 = df.iloc[last_5_minmaxs[0]].close
    c2 = df.iloc[last_5_minmaxs[1]].close
    c3 = df.iloc[last_5_minmaxs[2]].close
    c4 = df.iloc[last_5_minmaxs[3]].close
    c5 = df.iloc[last_5_minmaxs[4]].close
    c_today = df.loc[today].close

    # import pdb;pdb.set_trace()

    if not c1 > c3 > c5:
        print("[X] 不满足 c1 > c3 > c5")
        return False
    if not c2 > c4 > c_today:
        print("[X] 不满足 c2 > c4 > c_today")
        return False
    if not c1 > c3 > c2:
        print("[X] 不满足 c1 > c3 > c2")
        return False
    if not c3 > c5 > c4:
        print("[X] 不满足 c3 > c5 > c4")
        return False

    print("[√] 完美的五浪出现")
    return True

for date in df.index:
    is_5_waves(df,minmax_index,date)