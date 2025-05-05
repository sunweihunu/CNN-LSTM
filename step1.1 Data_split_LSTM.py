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

    site = ["cz", 'lz', 'sg']
    site_name = ["郴州", "连州", "韶关"]
    num = 0
    for i, s in enumerate(site):
        print("共{}个站点，现在处理第{}个站点：{}".format(len(site), i+1, site_name[i]))
        label = []
        date_ = []
        tem_time = []
        pre_time = []
        water_l_time = []

        tem = data[s + '_tem'].values
        pre = data[s + '_prec'].values

        for n, w_l in enumerate(water_l):
            sg.one_line_progress_meter(site_name[i] + '处理进度', n + 1, len(water_l), '-key-')
            if time_steps <= n < len(water_l) - forecast_period:
                label.append(water_l[n:n + forecast_period])
                pre_time.append(pre[n - time_steps: n])
                tem_time.append(tem[n - time_steps: n])
                water_l_time.append(water_l[n - time_steps: n])

                # 记录时间和样本总数
                date_.append(str(date[n]) + "-" + str(hour[n]))
                num += 1

        pre_time = np.expand_dims(np.asarray(pre_time), axis=2)
        tem_time = np.expand_dims(np.asarray(tem_time), axis=2)
        water_l_time = np.expand_dims(np.asarray(water_l_time), axis=2)
        label = np.asarray(label)

        fac = np.concatenate([normalization(pre_time), normalization(tem_time), water_l_time], axis=2)

        np.save(r"data/" + s + "_fac_LSTM.npy", fac)
        np.save(r"data/label.npy", label)

        date_ = np.expand_dims(np.asarray(date_), axis=1)
        date_ = np.concatenate([date_, label], axis=1)
        date_ = pd.DataFrame(date_)
        date_.columns = ["date", "water_level_t1", "water_level_t2", "water_level_t3", "water_level_t4"]
        # date_.columns = ["date", "runoff"]
        date_.to_csv(r'data/label.csv', index=False)
    print("可用样本：{}".format(int(num/3)))
    print('All end.')
