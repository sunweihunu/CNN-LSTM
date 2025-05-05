import numpy as np
import pandas as pd
import PySimpleGUI as sg


def normalization(mat):
    _range = np.max(mat) - np.min(mat)
    return (mat - np.min(mat)) / _range


if __name__ == '__main__':
    time_steps = 20
    forecast_period = 4

    data = pd.read_csv(r'data/2012.7_2015.4_6h.csv')

    date = data['date'].values
    hour = data['hour'].values
    water_l = data['runoff_lin'].values
    # water_l = data['poly2'].values

    pre = np.load(r"data/prec6×6.npy")
    tem = np.load(r"data/tem6×6.npy")

    num = 0
    label = []
    date_ = []
    tem_time = []
    pre_time = []
    water_l_time = []

    for n, w_l in enumerate(water_l):
        sg.one_line_progress_meter('处理进度', n + 1, len(water_l), '-key-')
        if time_steps <= n < len(water_l) - forecast_period:
            label.append(water_l[n:n + forecast_period])
            pre_time.append(pre[n - time_steps: n, :, :])
            tem_time.append(tem[n - time_steps: n, :, :])
            water_l_time.append(water_l[n - time_steps: n])

            # 记录时间和样本总数
            date_.append(str(date[n]) + "-" + str(hour[n]))
            num += 1

    pre_time = np.expand_dims(np.asarray(pre_time), axis=4)
    tem_time = np.expand_dims(np.asarray(tem_time), axis=4)
    water_l_time = np.expand_dims(np.asarray(water_l_time), axis=2)
    label = np.asarray(label)

    fac = np.concatenate([normalization(pre_time), normalization(tem_time)], axis=4)
    # fac = fac[:, :, 2:8, 0:6, :]

    np.save(r"data/fac_ConvLSTM_A2.npy", fac)
    np.save(r"data/wal_ConvLSTM.npy", water_l_time)
    np.save(r"data/label_ConvLSTM.npy", label)

    date_ = np.expand_dims(np.asarray(date_), axis=1)
    date_ = np.concatenate([date_, label], axis=1)
    date_ = pd.DataFrame(date_)
    date_.columns = ["date", "water_level_t1", "water_level_t2", "water_level_t3", "water_level_t4"]
    date_.to_csv(r'data/label_ConvLSTM.csv', index=False)
    print("可用样本：{}".format(num))
    print('All end.')
