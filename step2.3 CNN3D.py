import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 忽略 warning 和 Error
from tensorflow_core.python.keras.utils import plot_model
from tensorflow.keras.layers import Dense, MaxPooling3D, \
    Conv3D, LSTM
from tensorflow.keras.layers import Flatten, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import losses
from tensorflow_core.python.keras import regularizers
import sys

sys.path.append('E:/Python/rainfall_predict_v1/CIKM-Cup-2017-master/code/TOOLS')
from CIKM_TOOLS import *

import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd

def plt_pic_save(train_list_, val_list_, tit, k_):
    plt.figure()
    plt.rc('font', family='Times New Roman')
    plt.plot(train_list_, label="Training")
    plt.plot(val_list_, label="Validation")
    # plt.title(tit + str(k_))
    plt.legend(fontsize=20)

    x = MultipleLocator(300)
    y = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x)
    ax.yaxis.set_major_locator(y)
    plt.ylim(0, 183357)

    plt.text(0.12, 0.85, "(b) 1D CNN-2-30", transform=plt.gca().transAxes, size=20)
    output = r'E:\Python\rainfall_predict_v1\Pingshi_ConvLSTM\6h_code\output\pic/'
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("MSE", fontsize=20)
    plt.tight_layout()
    plt.savefig(output + tit + str(k_) + ".jpg", dpi=500, bbox_inches='tight')
    plt.savefig(output + tit + str(k_) + ".svg", dpi=500, bbox_inches='tight')
    plt.show()

def cre_model(fil=4):
    # first
    # x = Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.01))(fac_in)
    x = Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same", activation='relu')(fac_in)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    # x = Conv3D(filters=8, kernel_size=(2, 2, 2), strides=(2, 1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Conv3D(filters=8, kernel_size=(2, 2, 2), strides=(2, 1, 1), padding="same", activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    x = Flatten()(x)

    # w = Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(wal_in)
    w = LSTM(units=10, activation="relu", return_sequences=False)(wal_in)
    w = Flatten()(w)

    x = Concatenate()([x, w])
    x = Dense(x.shape[1] * 2, activation='relu')(x)
    x = Dense(x.shape[1], activation='relu')(x)
    output = Dense(1, use_bias=True, activation='relu')(x)
    adam = Adamax(learning_rate=0.01)
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
    forecast_period = 1
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
    plot_model(model, to_file=r'output/CNN3D/CNN3D.png', show_shapes=True, show_layer_names=True, rankdir='TB',
               expand_nested=False, dpi=196)

    k_folder = 5  # 作几轮交叉验证
    bat_s = 512
    epoch = 1000
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

        stop = EarlyStopping(patience=200, min_delta=1, mode='min')
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200, min_delta=100, min_lr=0.00005, verbose=1)
        hist = model.fit([trn_fac, trn_wal], trn_label, epochs=epoch, batch_size=bat_s, verbose=2, shuffle=False,
                         validation_data=([val_fac, val_wal], val_label), callbacks=[reduce, stop])  # , callbacks=[stop]

        # 保存损失函数曲线
        loss_ = list(hist.history.values())
        trn_loss_ = np.array(loss_[0])
        val_loss_ = np.array(loss_[1])
        output = r'E:\Python\rainfall_predict_v1\Pingshi_ConvLSTM\6h_code\output\pic/'
        # 将trn_loss_和val_loss_转换为DataFrame
        trn_loss_sqrt = np.sqrt(trn_loss_)
        val_loss_sqrt = np.sqrt(val_loss_)
        # 将平方根处理后的损失值转换为DataFrame
        loss_df = pd.DataFrame({'Training Loss (sqrt)': trn_loss_sqrt, 'Validation Loss (sqrt)': val_loss_sqrt})
        # 指定CSV文件的保存路径
        csv_file_path = output + 'loss_data.csv'
        # 将DataFrame保存为CSV文件
        loss_df.to_csv(csv_file_path, index=False)

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
            log = open(r'output/CNN3D/k_val' + str(k + 1) + '/score_kfold' + str(k + 1) + '.txt', mode="w",
                       encoding="utf-8")
        else:
            log = open(r'output/CNN3D/k_val' + str(k + 1) + '/score_kfold' + str(k + 1) + '.txt', mode="a",
                       encoding="utf-8")
            print('\n', file=log)

        print(
            '时间步长为{}，提前期{:^2}小时的  Training  RMSE:{:^5}   R:{:^5}   NSE:{:^5}'.format(time_steps, forecast_period * 6,
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
            r'output/CNN3D/k_val' + str(k + 1) + '/trn_kfold' + str(k + 1) + '_fp' + str(forecast_period) + "_ts" + str(
                time_steps) + ".csv")

        val_pre = np.vstack([val_pre[:, 0], val_label])
        val_pre = pd.DataFrame(val_pre.T)
        val_pre.columns = ["pre", "truth"]
        val_pre.index = pd.Series(val_date)
        val_pre.to_csv(
            r'output/CNN3D/k_val' + str(k + 1) + '/val_kfold' + str(k + 1) + '_fp' + str(forecast_period) + "_ts" + str(
                time_steps) + ".csv")

        log.close()
        # 模型保存和初始化
        model.save(
            r'output/CNN3D/k_val' + str(k + 1) + '/model_kfold' + str(k + 1) + '_fp' + str(forecast_period) + '.h5')
        del model
        seed_tensorflow(seed_)
        model = cre_model()

    if forecast_period == 1 and time_steps == 5:
        log = open(r'output/CNN3D/Average_score_f' + str(forecast_period) + '.txt', mode="w", encoding="utf-8")
    else:
        log = open(r'output/CNN3D/Average_score_f' + str(forecast_period) + '.txt', mode="a", encoding="utf-8")
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

# kernel_regularizer=regularizers.l2(0.01)
