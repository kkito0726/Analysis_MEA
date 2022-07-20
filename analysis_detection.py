import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import griddata
import statistics


class ReadData():
    def __init__(self, hed_path, start, end):
        import os
        # hedファイルからサンプリングレートとゲインを取得
        self.samp, self.gain = self.decode_hed(hed_path)
        
        bio_path = os.path.splitext(hed_path)[0] + "0001.bio"
        self.data = self.read_bio(bio_path, start, end, sampling_rate=self.samp, gain=self.gain)
    
    # hedファイルの解読メソッド
    def decode_hed(self, file_name):

        # hedファイルを読み込む。
        hed_data = np.fromfile(file_name, dtype='<h', sep='')

        # rate（サンプリングレート）、gain（ゲイン）の解読辞書。
        rates = {0:100000, 1:50000, 2:25000, 3:20000, 4:10000, 5:5000}
        gains ={16436:20, 16473:100, 16527:1000, 16543:2000,\
                16563:5000, 16579:10000, 16595:20000, 16616:50000}

        # サンプリングレートとゲインを返す。
        # hed_dataの要素16がrate、要素3がgainのキーとなる。
        return [rates[hed_data[16]], gains[hed_data[3]]]
    
    # bioファイルを読み込むメソッド
    def read_bio(self, file_name, start, end, sampling_rate=10000, gain=50000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
        import numpy as np

        electrode_number = 64
        data_unit_length = electrode_number + 4

        bytesize = np.dtype("<h").itemsize
        data = np.fromfile(file_name, dtype="<h", sep='', offset=start*sampling_rate*bytesize * data_unit_length, count=(end-start)*sampling_rate*data_unit_length) * (volt_range / (2**16-2)) * 4
        data = data.reshape(int(len(data) / data_unit_length), data_unit_length).T
        data = np.delete(data, range(4), 0)
        
        # Gainの値に合わせてデータを増幅させる。
        if gain != 50000:
            amp = 50000 / gain
            data *= amp
            
        t = np.arange(len(data[0])) / sampling_rate
        t = t.reshape(1, len(t))
        t = t + start
        data = np.append(t, data, axis=0)
    
        return data
    
# 64電極すべての電極の波形を出力
def showAll(MEA_raw, start=0, end=5, volt_min=-200, volt_max=200):

    sampling_rate = 10000
    
    sampling_rate = 10000 # サンプリングレート (Hz)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=(16,16))
    for i in range(1, 65, 1):
        plt.subplot(8, 8, i)
        plt.plot(MEA_raw[0][start_frame:end_frame], MEA_raw[i][start_frame:end_frame])
        plt.ylim(volt_min, volt_max)
        
    plt.show()


def detect_peak_neg(data, distance=5000):

    peak_index = np.array([None for _ in range(len(data))])
    for i in range(1, len(data)):
        height = np.std(data[i]) * 3
        detect_peak_index = find_peaks(-data[i], height=height, distance=distance)
        
        peak_index[i] = detect_peak_index[0]
        peak_index[i] = np.sort(peak_index[i])
    peak_index[0] = np.array([])
        
    return peak_index

def draw(peak_index):
    time = [data[0][peak_index[i]] for i in range(1, 65)]
    peaks = [len(peak_index[i]) for i in range(1, 65)]
    remove_ch = []
    for i in range(len(time)):
        if len(time[i]) != statistics.mode(peaks):
            remove_ch.append(i)
    print(np.array(remove_ch))
    time_del = np.delete(time, remove_ch, 0)
    
    df = pd.read_csv("./electrode.csv")

    # データ範囲を取得
    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()

    # 取得したデータ範囲で新しく座標にする配列を作成
    new_x_coord = np.linspace(x_min, x_max, 100)
    new_y_coord = np.linspace(y_min, y_max, 100)

    # ベクトル描画用の座標も作成
    new_x_coord_vec = np.linspace(x_min, x_max, 8)
    new_y_coord_vec = np.linspace(y_min, y_max, 8)

    # x, yのグリッド配列作成
    xx, yy = np.meshgrid(new_x_coord, new_y_coord)
    yy_vec, xx_vec = np.meshgrid(new_x_coord_vec, new_y_coord_vec)
    # 既知のx, y座標, その値取得
    knew_xy_coord = df[['X', 'Y']].values
    knew_xy_coord = np.delete(knew_xy_coord, remove_ch, 0)
    l = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    for f in range(len(time_del[0])):
        knew_values = [time_del[i][f] for i in range(len(time_del))]
        knew_values -= np.min(knew_values)

        result = griddata(points=knew_xy_coord, values=knew_values, xi=(xx, yy), method='cubic')
        result_vec = griddata(points=knew_xy_coord, values=knew_values, xi=(xx_vec, yy_vec), method='cubic')
        gradx, grady = np.gradient(result_vec, 1, 1)
        
        # グラフ表示
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        c = ax.contourf(xx, yy, result, cmap='jet')
        ax.contour(xx, yy, result,colors="k", linewidths = 0.5, linestyles = 'solid')
        plt.scatter(df["X"], df["Y"], marker=",", color="w")
        plt.scatter(knew_xy_coord[:,0], knew_xy_coord[:,1],marker=",", color="gray")
        plt.quiver(xx_vec, yy_vec, gradx , grady)
        plt.colorbar(c)
        plt.show()

if __name__ == "__main__":
    
    hed_path = input("ヘッダファイルのパスを入力: ")
    start = input("読み込み開始時刻を入力: ")
    end = input("読み込み終了時刻を入力: ")
    
    data = ReadData(hed_path, int(start), int(end)).data
    showAll(data)
    peak_index = detect_peak_neg(data)
    draw(peak_index)