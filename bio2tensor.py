def edit_neuron(file_name, start, end, sampling_rate=10000, volt_range=100): # sampling_rate (Hz), volt_range (mV)
    import numpy as np

    electrode_number = 64
    data_unit_length = electrode_number + 4

    bytesize = np.dtype("<h").itemsize
    data = np.fromfile(file_name, dtype="<h", sep='', offset=start*sampling_rate*bytesize * data_unit_length, count=(end-start)*sampling_rate*data_unit_length) * (volt_range / (2**16-2)) * 4
    data = data.reshape(int(len(data) / data_unit_length), data_unit_length).T
    data = np.delete(data, range(4), 0)
    t = np.arange(len(data[0])) / sampling_rate
    t = t.reshape(1, len(t))
    t = t + start
    data = np.append(t, data, axis=0)
    
    return data


def detect_peak_index(data, threshold=[5., 5.], order=[3, 3]):
    import numpy as np
    from scipy import signal
    
    peak_index = np.array([None for _ in range(len(data))])
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
    peak_index[0] = np.array([])
        
    return peak_index


def mkTensor(data, peak_index, cycle, time_window=0.04,tensor_duration=2):
    '''
    data=bio_data, peak_index=ピークの時刻データ, time_window=1ピクセルの時間 [s], tensor_duration=1画像の時間 [s]
    縦軸方向に時間、横軸方向に電極の縦(50)*横(64)の二次元テンソルを出力する。
    縦の1ピクセルは40ms間でのピークの数を表している。よって40ms*50=2s間分のデータを
    '''
    import numpy as np

    time_window = 0.04 # 1ピクセルの時間 [s]
    comp = []

    num = [time_window * i + tensor_duration * cycle for i in range(50)]
    for i in num:
        ele_num = []
        for ele in range(1, 65):
            peak_time = data[0][peak_index[ele]]
            peak_time = peak_time[peak_time<i+time_window]
            peak_time = peak_time[peak_time>i-time_window]
            ele_num.append(len(peak_time))
        comp.append(ele_num)
    
    return np.array(comp)

def mkTensorVolt(data, cycle, time_window=0.04,tensor_duration=2, samplimg=10000):
    import numpy as np
    comp = []
    peak_index = detect_peak_index(data)

    num = [time_window * i + tensor_duration * cycle for i in range(50)]
    for n in num:
        peak_volt = []
        for ele in range(1, len(data)):
            index = peak_index[ele][peak_index[ele]<(n+time_window)*samplimg]
            index = index[index>n*samplimg]
            peak = abs(data[ele][index])

            if len(peak) == 0:
                peak_volt.append(0)
            else:
                peak_volt.append(peak.max())
        comp.append(peak_volt)

    return np.array(comp)




if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    import numpy as np

    file_path = input("bioファイルのパスを入力: ")
    time = input("何秒間のデータを変換しますか？: ")

    print("変換中です!!")

    tensors = []
    for cycle, t in enumerate([i for i in range(0, int(time), 2)]):
        data = edit_neuron(file_path,t,t+2)
        peak_index = detect_peak_index(data)
        ten = mkTensor(data, peak_index, cycle)
        tensors.append(ten)
    tensors = np.array(tensors)

    print(tensors.shape)

    save_dir = input("保存先のディレクトリのパスを入力: ")
    save_file = input("ファイル名を入力: ")
    np.save(save_dir+"/"+save_file, tensors)