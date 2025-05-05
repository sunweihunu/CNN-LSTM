import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error
from tensorflow_core.python.keras.utils import plot_model
from tensorflow.keras.layers import Dense, ConvLSTM2D, MaxPooling2D, LSTM, MaxPooling3D
from tensorflow.keras.layers import Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import losses
import sys

sys.path.append('E:/Python/rainfall_predict_v1/CIKM-Cup-2017-master/code/TOOLS')
from CIKM_TOOLS import *

import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd


def cre_model(fil=4):
    # first
    x = ConvLSTM2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu",
                   return_sequences=True)(fac_in)
    x = ConvLSTM2D(filters=8, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu",
                   return_sequences=False)(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)

    w = LSTM(units=10, activation="relu", return_sequences=False)(wal_in)
    w = Flatten()(w)

    x = Concatenate()([x, w])
    x = Dense(x.shape[1]*2, activation='relu')(x)
    x = Dense(x.shape[1], activation='relu')(x)
    output = Dense(1, use_bias=True, activation='relu')(x)
    adam = Adam(learning_rate=0.001)
    model_ = Model(inputs=[fac_in, wal_in], outputs=output)
    model_.compile(optimizer=adam, loss=losses.mean_squared_error)

    return model_


if __name__ == '__main__':
    tf.disable_v2_behavior()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu90%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)

    # *****************************参数设置******************************* #
    forecast_period = 2
    time_steps = 15
    seed_ = 110
    seed_tensorflow(seed_)

    fac = np.load(r'data/fac_ConvLSTM_A1.npy')[:, 20 - time_steps:, :, :, :]
    wal = np.load(r'data/wal_ConvLSTM.npy')[:, 20 - time_steps:, :]
    date = pd.read_csv(r"data/label.csv")['date'].values
    label = np.load(r'data/label_ConvLSTM.npy')[:, forecast_period - 1]

    fac_in = Input(shape=(time_steps, np.shape(fac)[2], np.shape(fac)[3], np.shape(fac)[4]))
    wal_in = Input(shape=(time_steps, 1))

    model = cre_model()
    model.summary()
    plot_model(model, to_file=r'output/ConvLSTM/ConvLSTM.png', show_shapes=True, show_layer_names=True, rankdir='TB',
               expand_nested=False, dpi=196)

    k_folder = 5  # 作几轮交叉验证
    bat_s = 512
    epoch = 200
    trn_rmse_sum = 0
    trn_r_sum = 0
    trn_nse_sum = 0
    val_rmse_sum = 0
    val_r_sum = 0
    val_nse_sum = 0
    print(tf.test.is_gpu_available())
    for kk in range(1):
        k = 1  # 第几折  手动控制
        print('********************************第{}折*************************************'.format(k + 1))
        trn_fac, val_fac = k_val(k, k_folder, fac)
        trn_wal, val_wal = k_val(k, k_folder, wal)
        trn_label, val_label = k_val(k, k_folder, label)
        trn_date, val_date = k_val(k, k_folder, date)

        stop = EarlyStopping(patience=100, min_delta=5, mode='min')
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200, min_delta=100, min_lr=0.00005,
                                   verbose=1)
        hist = model.fit([trn_fac, trn_wal], trn_label, epochs=epoch, batch_size=bat_s, verbose=2, shuffle=False,
                         validation_data=([val_fac, val_wal], val_label), callbacks=[reduce])  # , callbacks=[stop]
        train_pre = model.predict([trn_fac, trn_wal], batch_size=bat_s, verbose=2)
        val_pre = model.predict([val_fac, val_wal], batch_size=bat_s, verbose=2)

        train_rmse, train_r, train_nse = score_runoff(trn_label, train_pre[:, 0])
        val_rmse, val_r, val_nse = score_runoff(val_label, val_pre[:, 0])

        trn_rmse_sum = trn_rmse_sum + train_rmse
        trn_r_sum = trn_r_sum + train_r
        trn_nse_sum = trn_nse_sum + train_nse

        val_rmse_sum = val_rmse_sum + val_rmse
        val_r_sum = val_r_sum + val_r
        val_nse_sum = val_nse_sum + val_nse

        if forecast_period == 1 and time_steps == 5:
            log = open(r'output/ConvLSTM/k_val' + str(k + 1) + '/score_kfold' + str(k + 1) + '.txt', mode="w",
                       encoding="utf-8")
        else:
            log = open(r'output/ConvLSTM/k_val' + str(k + 1) + '/score_kfold' + str(k + 1) + '.txt', mode="a",
                       encoding="utf-8")
            print('\n', file=log)

        print('时间步长为{}，提前期{:^2}小时的  Training  RMSE:{:^5}   R:{:^5}   NSE:{:^5}'.format(time_steps, forecast_period * 6,
                                                                                       round(train_rmse, 3),
                                                                                       round(train_r, 3),
                                                                                       round(train_nse, 3)),
              file=log)
        print('时间步长为{}，提前期{:^2}小时的 Validation RMSE:{:^5}   R:{:^5}   NSE:{:^5}'.format(time_steps, forecast_period * 6,
                                                                                       round(val_rmse, 3),
                                                                                       round(val_r, 3),
                                                                                       round(val_nse, 3)),
              file=log)
        print('Training RMSE:{},R:{},NSE:{}'.format(train_rmse, train_r, train_nse))
        print('Validation RMSE:{},R:{},NSE:{}'.format(val_rmse, val_r, val_nse))

        train_pre = np.vstack([train_pre[:, 0], trn_label])
        train_pre = pd.DataFrame(train_pre.T)
        train_pre.columns = ["pre", "truth"]
        train_pre.index = pd.Series(trn_date)
        train_pre.to_csv(
            r'output/ConvLSTM/k_val' + str(k + 1) + '/trn_kfold' + str(k + 1) + '_fp' + str(
                forecast_period) + "_ts" + str(
                time_steps) + ".csv")

        val_pre = np.vstack([val_pre[:, 0], val_label])
        val_pre = pd.DataFrame(val_pre.T)
        val_pre.columns = ["pre", "truth"]
        val_pre.index = pd.Series(val_date)
        val_pre.to_csv(
            r'output/ConvLSTM/k_val' + str(k + 1) + '/val_kfold' + str(k + 1) + '_fp' + str(
                forecast_period) + "_ts" + str(
                time_steps) + ".csv")

        log.close()
        # 模型保存和初始化
        del model
        seed_tensorflow(seed_)
        model = cre_model()

    if forecast_period == 1 and time_steps == 5:
        log = open(r'output/ConvLSTM/Average_score_f' + str(forecast_period) + '.txt', mode="w", encoding="utf-8")
    else:
        log = open(r'output/ConvLSTM/Average_score_f' + str(forecast_period) + '.txt', mode="a", encoding="utf-8")
        print('\n', file=log)
    print('时间步长为{}，提前期{:^2}小时的  Training  RMSE_AVE:{:^5}   R_AVE:{:^5}   NSE_AVE:{:^5}'.format(time_steps,
                                                                                               forecast_period * 6,
                                                                                               round(
                                                                                                   trn_rmse_sum / k_folder,
                                                                                                   3),
                                                                                               round(
                                                                                                   trn_r_sum / k_folder,
                                                                                                   3),
                                                                                               round(
                                                                                                   trn_nse_sum / k_folder,
                                                                                                   3)),
          file=log)
    print('时间步长为{}，提前期{:^2}小时的 Validation RMSE_AVE:{:^5}   R_AVE:{:^5}   NSE_AVE:{:^5}'.format(time_steps,
                                                                                               forecast_period * 6,
                                                                                               round(
                                                                                                   val_rmse_sum / k_folder,
                                                                                                   3),
                                                                                               round(
                                                                                                   val_r_sum / k_folder,
                                                                                                   3),
                                                                                               round(
                                                                                                   val_nse_sum / k_folder,
                                                                                                   3)),
          file=log)
    print('All end')
