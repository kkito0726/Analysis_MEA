import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# hedファイルの解読関数
def decode_hed(file_name):
  import numpy as np

  # hedファイルを読み込む。
  hed_data = np.fromfile(file_name, dtype='<h', sep='')

  # rate（サンプリングレート）、gain（ゲイン）の解読辞書。
  rates = {0:100000, 1:50000, 2:25000, 3:20000, 4:10000, 5:5000}
  gains ={16436:20, 16473:100, 16527:1000, 16543:2000,\
          16563:5000, 16579:10000, 16595:20000, 16616:50000}

  # サンプリングレートとゲインを返す。
  # hed_dataの要素16がrate、要素3がgainのキーとなる。
  return [rates[hed_data[16]], gains[hed_data[3]]]

# bioファイルを読み込む関数
def read_bio(file_name, start, end, sampling_rate=10000, gain=50000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
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

# hedファイルの情報からbioファイルを一気に読み込む
def hed2array(file_name, start, end):
    import os
    # hedファイルからサンプリングレートとゲインを取得
    samp, gain = decode_hed(file_name)
    
    bio_path = os.path.splitext(file_name)[0] + "0001.bio"
    return read_bio(bio_path, start, end, sampling_rate=samp, gain=gain)

def detect_peak_neg(data, distance=5000):

    peak_index = np.array([None for _ in range(len(data))])
    for i in range(1, len(data)):
        height = np.std(data[i]) * 3
        detect_peak_index = find_peaks(-data[i], height=height, distance=distance)
        
        peak_index[i] = detect_peak_index[0]
        peak_index[i] = np.sort(peak_index[i])
    peak_index[0] = np.array([])
        
    return peak_index


def isi_IRlaser(hed_path, interval_length, interval_num):
    start= [0, 34, 64, 94, 124, 154, 184, 214, 244, 274] 
    end = [i*interval_length for i in range(1, interval_num+1)]
    x, y = [], []
    ele = 28

    for t in range(len(start)):
        data = hed2array(hed_path, start[t], end[t])
        peak_index = detect_peak_neg(data)
        time = [data[0][peak_index[i]] for i in range(1, 65)]
        
        isi_x = time[ele][:-1]
        isi = time[ele][1:]-time[ele][:-1]
        
        x.append(isi_x)
        y.append(isi)
    
    return x, y

def graph(x, y):
    import itertools
    
    irradiation_x, irradiation_y = [], []
    no_irradiation_x, no_irradiation_y = [], []

    '''
    レーザー照射したデータとしていないデータを仕分けする。
    奇数番目のデータは照射、偶数番目は照射なし。
    '''
    for i in range(len(x)):
        if i % 2 != 0:
            irradiation_x.append(x[i])
            irradiation_y.append(y[i])
        if i % 2 == 0:
            no_irradiation_x.append(x[i])
            no_irradiation_y.append(y[i])
            
    # 配列を一次元化
    no_irradiation_x = list(itertools.chain.from_iterable(no_irradiation_x))
    no_irradiation_y = list(itertools.chain.from_iterable(no_irradiation_y))

    irradiation_x = list(itertools.chain.from_iterable(irradiation_x))
    irradiation_y = list(itertools.chain.from_iterable(irradiation_y))

    # グラフの描画
    plt.figure(dpi=300)
    plt.rcParams["font.size"] = 12
    plt.rcParams [ "font.family" ]  =  "century"
    plt.plot(no_irradiation_x, no_irradiation_y, ".", label="No irradiation")
    plt.plot(irradiation_x, irradiation_y, ".", label="0.5 W irradiation")
    plt.legend(fontsize=11)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("ISI (s)", fontsize=14)
    plt.show()
    
if __name__ == "__main__":
    hed_path = "G:\マイドライブ\研究\心筋シート\赤外線レーザー損傷/220907_解剖_9日胚\dish3/220910_day3_5point_0.5W_5min_.hed"
    interval_length = 30
    interval_num = 10
    
    x, y = isi_IRlaser(hed_path, interval_length, interval_num)
    graph(x, y)