def read_bio(file_name, sampling_rate=10000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
    import numpy as np

    electrode_number = 64
    data_unit_length = electrode_number + 4

    data = np.fromfile(file_name, dtype="<h", sep='') * (volt_range / (2**16)) * 4
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
    data = np.fromfile(file_name, dtype="<h", sep='', offset=start*sampling_rate*bytesize * data_unit_length, count=(end-start)*sampling_rate*bytesize*data_unit_length) * (volt_range / (2**16-2)) * 40
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
    MEA_raw = MEA_raw[:64, start * sampling_rate:end * sampling_rate]
    return MEA_raw

# 64電極すべての電極の波形を出力
# 第一引数はbioファイルのパス
def showAll(file_name, start=0, end=5, volt_min=-50, volt_max=50):
    import matplotlib.pyplot as plt

    sampling_rate = 10000
    volt_range = 100

    MEA_raw = read_bio(file_name, sampling_rate, volt_range)

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

def showDetection(file_name, eles, start=0, end=5, sampling_rate=10000):
    import matplotlib.pyplot as plt
    import numpy as np

    MEA_raw = read_bio(file_name)

    MEA_data = []
    for ele in eles:
        MEA_data.append(MEA_raw[ele])

    data = np.array(MEA_data)
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    plt.figure(figsize=(12, 12))
    for i, index in enumerate(np.array(data)):
        tmp_volt = (index - np.mean(index)) / 50
        plt.plot(MEA_raw[0][start_frame:end_frame], tmp_volt[start_frame:end_frame] + i)

    plt.xlim(start, end)
    plt.yticks(range(0,len(eles),1))
    plt.ylim(-1, len(eles),1)
    plt.xlabel("Time(s)")
    plt.ylabel("Electrode Number")
    plt.show()

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

# フィルタ処理
def filter_MEA(data, sampling_rate=10000):
    import numpy as np
    import filter_function

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
        
        # 50 Hzでバンドストップ
        data_filt[i] = filter_function.bandstop(data_filt[i], sampling_rate, fp_50, fs_50, gpass_50, gstop_50)
        
#         # 2 kHzでバンドストップ
#         data_filt[i] = filter_function.bandstop(data_filt[i], sampling_rate, fp_2k, fs_2k, gpass_2k, gstop_2k)
        
#         # 移動平均
#         data_filt[i] = np.convolve(data_filt[i], b, mode='same')
        
    return data_filt

# ピーク検出
def detect_peak_index(data, threshold=[5., 5.], order=[3, 3]):
    import numpy as np
    from scipy import signal
    
    peak_index = [None for _ in range(len(data))]
    for i in range(1, len(data)):
        std_volt = np.std(data[i])
        
        volt_high = data[i].copy()
        volt_high[volt_high <  std_volt * threshold[0]] = 0.
        positive_peak_index = signal.argrelmax(volt_high, order=order[0])

        volt_low = data[i].copy()
        volt_low[volt_low >  -std_volt * threshold[0]] = 0.
        negative_peak_index = signal.argrelmin(volt_low, order=order[1])

        peak_index[i] = np.append(positive_peak_index[0], negative_peak_index[0])
        peak_index[i] = np.sort(peak_index[i])
        
    return peak_index
    

# def save_as_bio:

# def save_peaks_as_csv:

# def read_peaks_from_bio:

# def read_peaks_from_csv:
"""
# ラスタプロット
def raster_plot(data, peak_index):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        positive_peak_index = peak_index[i][data[i][peak_index[i]] > 0]
        plt.plot(data[0][positive_peak_index], np.ones(len(positive_peak_index)) * i, "|", color='black', markersize=4)
    plt.title("Positive peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    plt.xlabel("Time (s)")
    plt.ylabel("Electrode ID")
    plt.savefig("positive_peaks.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        negative_peak_index = peak_index[i][data[i][peak_index[i]] < 0]
        plt.plot(data[0][negative_peak_index], np.ones(len(negative_peak_index)) * i, "|", color='black', markersize=4)
    plt.title("Negative peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    plt.xlabel("Time (s)")
    plt.ylabel("Electrode ID")
    plt.savefig("negative_peaks.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        plt.plot(data[0][peak_index[i]], np.ones(len(peak_index[i])) * i, "|", color='black', markersize=4)
    plt.title("Positive peaks & Negative peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    plt.xlabel("Time (s)")
    plt.ylabel("Electrode ID")
    plt.savefig("all_peaks.png", dpi=300)
    plt.show()
"""
# ラスタプロット
def raster_plot(data, peak_index, file_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        positive_peak_index = peak_index[i][data[i][peak_index[i]] > 0]
        plt.plot(data[0][positive_peak_index], np.ones(len(positive_peak_index)) * i, "|", color='black', markersize=4)
    plt.title("Positive peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    file_name_pos = os.path.splitext(file_name)[0] + "_pos.png"
    plt.savefig(file_name_pos, dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        negative_peak_index = peak_index[i][data[i][peak_index[i]] < 0]
        plt.plot(data[0][negative_peak_index], np.ones(len(negative_peak_index)) * i, "|", color='black', markersize=4)
    plt.title("Negative peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    file_name_neg = os.path.splitext(file_name)[0] + "_neg.png"
    plt.savefig(file_name_neg, dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 8))    
    for i in range(1, len(data)):
        plt.plot(data[0][peak_index[i]], np.ones(len(peak_index[i])) * i, "|", color='black', markersize=4)
    plt.title("Positive peaks & Negative peaks")
    plt.xticks(range(0, int(np.max(data[0])), 30))
    plt.xlim(0, np.max(data[0]))
    plt.ylim(0, 65)
    plt.yticks(range(4, 65, 4))
    file_name_all = os.path.splitext(file_name)[0] + "_all.png"
    plt.savefig(file_name_all, dpi=300)
    plt.show()
    
# 一気通貫
def bio_to_raster(file_name, sampling_rate=10000, volt_range=100):
    MEA_raw = read_bio(file_name, sampling_rate=10000, volt_range=100)
    MEA_filt = filter_MEA(MEA_raw)
    peak_index = detect_peak_index(MEA_filt)
#     raster_plot(MEA_filt, peak_index)
    raster_plot(data=MEA_filt, peak_index=peak_index, file_name=file_name)