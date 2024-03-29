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


def filter_MEA(data, sampling_rate=10000):
    import numpy as np
    # import filter_function

    data_filt = np.zeros_like(data)
    data_filt[0] = data[0].copy()
    
    # フィルタの設定（50 Hzの周期的ノイズ）
    fp_50 = np.array([45, 55]) # 通過域端周波数[Hz]
    fs_50 = np.array([30, 100]) # 阻止域端周波数[Hz]
    gpass_50 = 3 # 通過域端最大損失[dB]
    gstop_50 = 40 # 阻止域端最小損失[dB]
    
   # フィルタの設定（2 kHzの周期的ノイズ）
    fp_2k = np.array([1900, 2100]) # 通過域端周波数[Hz]
    fs_2k = np.array([1000, 4000]) # 阻止域端周波数[Hz]
    gpass_2k = 3 # 通過域端最大損失[dB]
    gstop_2k = 40 # 阻止域端最小損失[dB]
    
    # 移動平均
    num = 5  # 移動平均のフレーム数
    b = np.ones(num)/num
    
    # フィルタ処理
    for i in range(1, len(data)):
        # 平均が0になるように値をシフト
        data_filt[i] = data[i] - np.mean(data[i])
        
#         # 50 Hzでバンドストップ
#         data_filt[i] = filter_function.bandstop(data_filt[i], sampling_rate, fp_50, fs_50, gpass_50, gstop_50)
        
#         # 2 kHzでバンドストップ
#         data_filt[i] = filter_function.bandstop(data_filt[i], sampling_rate, fp_2k, fs_2k, gpass_2k, gstop_2k)
        
        # 移動平均
        data_filt[i] = np.convolve(data_filt[i], b, mode='same')
        
    return data_filt

# 64電極すべての電極の波形を出力
def showAll(MEA_raw, start=0, end=5, volt_min=-200, volt_max=200):
    import matplotlib.pyplot as plt

    sampling_rate = 10000
    volt_range = 100
    
    sampling_rate = 10000 # サンプリングレート (Hz)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=(16,16))
    for i in range(1, 65, 1):
        plt.subplot(8, 8, i)
        plt.plot(MEA_raw[0][start_frame:end_frame], MEA_raw[i][start_frame:end_frame])
        # plt.xlim(start, end)
        plt.ylim(volt_min, volt_max)
        
    plt.show()

# 64電極すべての下ピークを取得
def detect_peak_neg(data, distance=5000, width=None, prominence=None):
    import numpy as np
    from scipy.signal import find_peaks
    
    peak_index = np.array([None for _ in range(len(data))])
    for i in range(1, len(data)):
        height = np.std(data[i]) * 3
        detect_peak_index = find_peaks(-data[i], height=height, distance=distance, width=width, prominence=prominence)
        
        peak_index[i] = detect_peak_index[0]
        peak_index[i] = np.sort(peak_index[i])
    peak_index[0] = np.array([])
        
    return peak_index

# 64電極すべての上ピークを取得
def detect_peak_pos(data, distance=10000, width=None, prominence=None, height=(10, 80)):
    import numpy as np
    from scipy.signal import find_peaks
    
    peak_index = np.array([None for _ in range(len(data))])
    for i in range(1, len(data)):
        # height = np.std(data[i]) * 3
        detect_peak_index = find_peaks(data[i], height=height, distance=distance, width=width, prominence=prominence)
        
        peak_index[i] = detect_peak_index[0]
        peak_index[i] = np.sort(peak_index[i])
    peak_index[0] = np.array([])
        
    return peak_index

#外周のデータを表示
def circuit(file_name, start=0, end=5, sampling_rate=10000):
    import matplotlib.pyplot as plt
    import numpy as np

    eles = [
        1,2,3,4,5,6,7,8,
        16,24,32,40,48,56,64,
        63,62,61,60,59,58,57,
        49,41,33,25,17,9
    ]

    MEA_raw = read_bio(file_name)

    MEA_data = []
    for ele in eles:
        MEA_data.append(MEA_raw[ele])

    data = np.array(MEA_data)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=(8, 8))
    for i, index in enumerate(np.array(data)):
        tmp_volt = (index - np.mean(index)) / 200
        plt.plot(MEA_raw[0][start_frame:end_frame], tmp_volt[start_frame:end_frame] + i)

    plt.xlim(start, end)
    plt.yticks(range(0,len(eles),1))
    plt.ylim(-1, len(eles),1)
    plt.xlabel("Time(s)")
    plt.ylabel("Electrode Number")
    plt.show()


#任意の電極データを一つのグラフに表示
def showDetection(MEA_raw, eles, start=0, end=5, sampling_rate=10000, figsize=(12, 12)):
    import matplotlib.pyplot as plt
    import numpy as np


    MEA_data = []
    for ele in eles:
        MEA_data.append(MEA_raw[ele])

    data = np.array(MEA_data)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=figsize)
    for i, index in enumerate(np.array(data)):
        tmp_volt = (index - np.mean(index)) / 50
        plt.plot(MEA_raw[0][start_frame:end_frame], tmp_volt[start_frame:end_frame] + i)

    ele_label = [str(eles[i]) for i in range(len(eles))]
    plt.xlim(start, end)
    plt.yticks(range(0,len(eles),1), ele_label)
    plt.ylim(-1, len(eles),1)
    plt.xlabel("Time(s)")
    plt.ylabel("Electrode Number")
    plt.show()

def cv(data, first, second, length=450, threshold=5):
    import numpy as np
    from scipy import signal

    std_volt_second = np.std(data[second])
    volt_high_second = data[second].copy()
    volt_high_second[volt_high_second < std_volt_second * threshold] = 0
    positive_peak_index_second = signal.argrelmax(volt_high_second, order = 3)
    when_second = data[0][positive_peak_index_second]

    detect_second = []
    for i  in range(0, len(when_second)-1):
        if when_second[i+1] - when_second[i] > 0.2:
            detect_second.append(when_second[i])
    if when_second[len(when_second) - 1] - when_second[len(when_second) - 2] >0.2:
        detect_second.append(when_second[len(when_second) -1])
    detect_second = np.array(detect_second)

    std_volt_first = np.std(data[first])
    volt_high_first = data[first].copy()
    volt_high_first[volt_high_first < std_volt_first * threshold] = 0
    positive_peak_index_first = signal.argrelmax(volt_high_first, order = 3)
    when_first = data[0][positive_peak_index_first]

    detect_first = []
    for i  in range(0, len(when_first)-1):
        if when_first[i+1] - when_first[i] > 0.2:
            detect_first.append(when_first[i])
    if when_first[len(when_first) - 1] - when_first[len(when_first) - 2] >0.2:
        detect_first.append(when_first[len(when_first) -1])
    detect_first = np.array(detect_first)

    cv = (length/1000000)/(detect_second - detect_first)
    return cv

def negPeak(data, ele, isFig=True, threshold=5, height=20, distance=5000):
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    std_volt = np.std(data[ele])
    volt_low = data[ele].copy()
    volt_low[volt_low > - std_volt * threshold] = 0
    peaks,_ = find_peaks(-volt_low, height=height, distance=distance)
    time = data[0][peaks]

    if isFig == True:
        plt.figure(figsize=(100,10))
        plt.plot(data[0], volt_low)
        plt.plot(data[0][peaks], data[ele][peaks], 'ro')
        plt.show()

    return time

def negPeaks(data, ele, height=20, distance=5000, isFig = True):
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    peaks,_ = find_peaks(-data[ele], height=height, distance=distance)
    time = data[0][peaks]

    if isFig == True:
        plt.figure(figsize=(100,10))
        plt.plot(data[0], data[ele])
        plt.plot(data[0][peaks], data[ele][peaks], 'ro')
        plt.show()

    return time

def posPeak(data, ele, isFig=True, threshold=5, height=20,distance=5000):
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    std_volt = np.std(data[ele])
    volt_high = data[ele].copy()
    volt_high[volt_high < std_volt * threshold] = 0
    peaks,_ = find_peaks(volt_high, height=height, distance=distance)
    time = data[0][peaks]

    if isFig == True:
        plt.figure(figsize=(100,10))
        plt.plot(data[0], volt_high)
        plt.plot(data[0][peaks], data[ele][peaks], 'ro')
        plt.show()

    return time

def posPeaks(data, ele, height=20, distance=5000, isFig = True):
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

    peaks,_ = find_peaks(data[ele], height=height, distance=distance)
    time = data[0][peaks]

    if isFig == True:
        plt.figure(figsize=(100,10))
        plt.plot(data[0], data[ele])
        plt.plot(data[0][peaks], data[ele][peaks], 'ro')
        plt.show()

    return time

def min_length(time, i):
    if len(time[i]) > len(time[i+1]):
        return len(time[i+1])
    else:
        return len(time[i])

def calc_velocity(data, peak_index):
    time = [data[0][peak_index[i]] for i in range(1, 65)]
    detect_times = []

    for i in range(63):
        length = min_length(time, i)
        if time[i+1][0] > time[i][0]:
            detect_time = time[i+1][:length] - time[i][:length]
        else:
            detect_time = time[i][:length] - time[i+1][:length]
        detect_time = detect_time[detect_time > 0]
        velocity = 450 / detect_time / (10**6)
        detect_times.append(velocity)

    return detect_times

def allTime(data, order=100):
    eles = [
        1,2,3,4,5,6,7,8,
        16,24,32,40,48,56,64,
        63,62,61,60,59,58,57,
        49,41,33,25,17,9
    ]

    allTime = []
    for ele in eles:
        time = negPeak(data, ele, False, order=order)
        allTime.append(time)
    
    return allTime

def allTimes(data, height=20, distance=5000):
    eles = [
        1,2,3,4,5,6,7,8,
        16,24,32,40,48,56,64,
        63,62,61,60,59,58,57,
        49,41,33,25,17,9
    ]

    allTime = []
    for ele in eles:
        time = negPeaks(data, ele, height=height, distance=distance, isFig=False)
        allTime.append(time)
    
    return allTime

def normalize(filename):
    import pandas as pd
    import numpy as np

    df = pd.read_excel(filename)
    a = np.array(df)
    a = np.delete(a,0,axis=1)
    normalize = []
    for i in range(len(a)):
        b = a[i]/a[i][0]
        normalize.append(b)
    return normalize