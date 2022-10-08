def read_bio(file_name, sampling_rate=10000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
    import numpy as np

    electrode_number = 64
    data_unit_length = electrode_number + 4

    data = np.fromfile(file_name, dtype="<h", sep='') * (volt_range / (2**16-2)) * 40
    data = data.reshape(int(len(data) / data_unit_length), data_unit_length).T
    data = np.delete(data, range(4), 0)
    t = np.arange(len(data[0])) / sampling_rate
    t = t.reshape(1, len(t))
    data = np.append(t, data, axis=0)
    
    return data

# データの一部分のみを読み込む
def edit_bio(file_name, start, end, sampling_rate=10000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
    import numpy as np

    electrode_number = 64
    data_unit_length = electrode_number + 4

    bytesize = np.dtype("<h").itemsize
    data = np.fromfile(file_name, dtype="<h", sep='', offset=start*sampling_rate*bytesize * data_unit_length, count=(end-start)*sampling_rate*data_unit_length) * (volt_range / (2**16-2)) * 40
    data = data.reshape(int(len(data) / data_unit_length), data_unit_length).T
    data = np.delete(data, range(4), 0)
    t = np.arange(len(data[0])) / sampling_rate
    t = t.reshape(1, len(t))
    t = t + start
    data = np.append(t, data, axis=0)
    
    return data


# 読み込んだデータを任意の範囲の時間に編集する。
def editTime(file_name, start, end):
    import numpy as np

    sampling_rate = 10000
    MEA_raw = read_bio(file_name)
    # MEA_data = []
    # for i in range(65):
    #     time = MEA_raw[i][start * sampling_rate : end * sampling_rate]
    #     MEA_data.append(time)
    MEA_raw = MEA_raw[:65, start * sampling_rate:end * sampling_rate]
    return MEA_raw

# 64電極すべての電極の波形を出力
# 第一引数はbioファイルのパス
def showAll(file_name, start=0, end=5, volt_min=-200, volt_max=200):
    import matplotlib.pyplot as plt

    sampling_rate = 10000
    volt_range = 100

    MEA_raw = edit_bio(file_name,start=start,end=end, sampling_rate=sampling_rate, volt_range=volt_range)

    sampling_rate = 10000 # サンプリングレート (Hz)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=(16,16))
    for i in range(1, 65, 1):
        plt.subplot(8, 8, i)
        plt.plot(MEA_raw[0][start_frame:end_frame], MEA_raw[i][start_frame:end_frame])
        plt.xlim(start, end)
        plt.ylim(volt_min, volt_max)
        
    plt.show()

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