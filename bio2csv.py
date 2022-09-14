#!/usr/bin/env python
# -*-:coding:utf-8 -*-
#
# bio2csv.py
# 2020.6.9, Masahito Hayashi
# ver 1.0
#
# MEAシステムの出力する.bioファイルを.csvファイルに変換する。
#
# 使い方
# python bio2csv.py 入力フォルダ名 [-v 最大電圧] [-r サンプリングレート] [-c チャンネルリスト]
# 
# あらかじめ.bioファイルをbio2csv.pyと同じ階層に作成した入力フォルダに入れておく。
# 複数の.bioファイルを入れておくと一括変換する。
# 出力される.csvファイルは同じフォルダ内に拡張子だけ変えたファイル名で保存される。
# （例）aaaaa.bio　→　aaaaa.csv
#
# []内の引数はオプション。指定しなくても良い。
# 最大電圧：測定時にGainで決まる測定範囲の最大値を整数で指定。指定しないと-32768 ~ 32767の整数で出力される。
# サンプリングレート：整数で指定。指定しないと1として扱われる。
# チャンネルリスト：チャンネル番号をスペースで区切って並べる。指定しないと全電極のデータが変換される。
# （例）python bio2csv.py aaaaa -v 100 -r 10000 -c 1 3 45 67
# →フォルダaaaaa内のすべての.bioファイルを最大電圧100、サンプリングレート10000として、1、3、45、67番目の電極のみ変換して.csvファイルを作成する。
# 　出力された.csvファイルの1列目は時刻、2，3，4，5列目は1，3，45，67番目の電極の電圧。

import numpy as np
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('-r', '--rate', type=int,default=1)
parser.add_argument('-v', '--voltmax', type=int,default=None)
parser.add_argument('-c', '--channels', type=int, nargs='*')

args = parser.parse_args()
input_files = glob.glob(os.path.join(args.input_dir, "*.bio"))
sampling_rate = args.rate
if args.voltmax == None:
    max_voltage = 1
else:
    max_voltage = args.voltmax / (2**16 - 2)

if args.channels == None:
    channels = np.arange(64)+4
else:
    channels = np.array(args.channels) + 4 - 1

electrode_number = 64
data_unit_length = electrode_number + 4

for i,input_file in enumerate(input_files):
    data = np.fromfile(input_file, dtype = "<h", sep='')
    data = data.reshape(int(len(data) / data_unit_length), data_unit_length).T
    data = data[channels] * max_voltage
    t = [np.arange(len(data[0])) / sampling_rate]
    data = np.concatenate((t, data))

    np.savetxt(os.path.splitext(input_file)[0] + '.csv', data.T, delimiter=',')