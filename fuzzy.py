#fuzzy extension principle +
# ファジィ工学特論 課題1 拡張原理 + - * / ソースコード

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
# use latex for font rendering
plt.rcParams["text.usetex"] = True

# ~3の値を返す関数 1 <= x <= 5
def three_value(x) :
    if x < 1 :
        return 0
    elif 1 <= x < 3 :
        return (x-1) / 2
    elif 3 <= x < 5 :
        return (5-x) / 2
    else :
        return 0

# narray配列を~3の値に変換する関数
def three_array(x) :
    f_x = np.where((1 <= x) & (x < 3), (x-1) / 2 , 0)
    fz_x = np.where((3 <= x) & (x < 5), (5-x) / 2 , 0)
    return fz_x + f_x

# ~5の値を返す関数 2<= y <= 8
def five_value(y) :
    if y < 2 :
        return 0
    elif 2 <= y < 5 :
        return (y-2) / 3
    elif 5 <= y < 8 :
        return (8-y) / 3
    else :
        return 0

# narray配列を~5の値に変換する関数
def five_array(y) :
    f_y = np.where((2 <= y) & (y < 5), (y-2) / 3, 0)
    fz_y = np.where((5 <= y) & (y < 8), (8-y) / 3, 0)
    return fz_y + f_y

# z = x + y のファジィ演算の結果を出力
def z_fuzzy_plus(z,x) :
    z_list = []
    z_array = np.zeros(len(z))

    # z(ファジィ数)の値を演算する.
    for i in range(len(z)) :
        if 3 <= z[i] <= 13 :
            for j in range(len(x)) :
                y = z[i] - x[j]
                z_list.append(min(three_value(x[j]),five_value(y)))

            z_array[i] = max(z_list)
            z_list.clear()

        else :
            z_array[i] = 0

    return z_array

# z = x - y のファジィ演算の結果を出力
def z_fuzzy_minus(z,x) :
    z_list = []
    z_array = np.zeros(len(z))

    # z(ファジィ数)の値を演算する.
    for i in range(len(z)) :
        if -7 <= z[i] <= 3 :
            for j in range(len(x)) :
                y = x[j] - z[i]
                z_list.append(min(three_value(x[j]),five_value(y)))

            z_array[i] = max(z_list)
            z_list.clear()

        else :
            z_array[i] = 0

    return z_array

# z = x * y のファジィ演算の結果を出力
def z_fuzzy_times(z,x) :
    z_list = []
    z_array = np.zeros(len(z))

    # z(ファジィ数)の値を演算する.
    for i in range(len(z)) :
        if 2 <= z[i] <= 40 :
            for j in range(len(x)) :
                y = z[i] / x[j]
                z_list.append(min(three_value(x[j]),five_value(y)))

            z_array[i] = max(z_list)
            z_list.clear()

        else :
            z_array[i] = 0

    return z_array

# z = x / y のファジィ演算の結果を出力
def z_fuzzy_divide(z,x) :
    z_list = []
    z_array = np.zeros(len(z))

    # z(ファジィ数)の値を演算する.
    for i in range(len(z)) :
        if 1 / 8 <= z[i] <= 5 / 2 :
            for j in range(len(x)) :
                y = x[j] / z[i]
                z_list.append(min(three_value(x[j]),five_value(y)))

            z_array[i] = max(z_list)
            z_list.clear()

        else :
            z_array[i] = 0

    return z_array

# x + y = zの演算に必要な配列を生成する. 3 <= z <= 13 *0.01刻み
z_p = np.arange(0,15.1,0.01)
x_p = np.arange(0,15.1,0.01)

# x - y = zの演算に必要な配列を生成する. -7 <= z <= 3 *0.01刻み
z_m = np.arange(-10,10.1,0.01)
x_m = np.arange(-10,10.1,0.01)

# x * y = zの演算に必要な配列を生成する. 2 <= z <= 40 *0.01刻み
z_t = np.arange(0,45,0.01)
x_t = np.arange(0,45,0.01)

# x / y = zの演算に必要な配列を生成する. 1/8 <= z <= 5/2 *0.01刻み
z_d = np.arange(0,10,0.01)
x_d = np.arange(0,10,0.01)

#グラフの設定
fig = plt.figure(figsize=(15,13))
plus = plt.subplot2grid((3,2), (0,0))
minus = plt.subplot2grid((3,2), (0,1))
times = plt.subplot2grid((3,2), (1,0),colspan=2)
divide = plt.subplot2grid((3,2), (2,0),colspan=2)

#plotの設定

# x + y の結果をグラフにプロットする
# ~3のグラフをプロット
plus.fill_between(x_p, three_array(x_p),color='deepskyblue',label=r'$\mu_3$',alpha=0.5)
# ~5のグラフをプロット
plus.fill_between(x_p, five_array(x_p),color='hotpink',label=r'$\mu_5$',alpha=0.5)
# ~3 + ~5のグラフをプロット
plus.fill_between(x_p, z_fuzzy_plus(z_p,x_p),color='orangered',label=r'$\mu_{3+5}$',alpha=0.5)
# グラフにグリッドを描画
plus.grid(which='major')
# 凡例の設定
plus.legend(loc='upper right', borderaxespad=0.3,fontsize=8)
# タイトルを設定
plus.set_title(r'$\mu_{3} + \mu_{5}$',fontsize=16)
# x軸のメモリの設定
plus.set_xticks(np.arange(0,16.1,2))
plus.set_xticks(np.arange(0,16.1,0.5), minor=True)


# x - y の結果をグラフにプロットする
# ~3のグラフをプロット
minus.fill_between(x_m, three_array(x_m),color='deepskyblue',label=r'$\mu_3$',alpha=0.5)
# ~5のグラフをプロット
minus.fill_between(x_m, five_array(x_m),color='hotpink',label=r'$\mu_5$',alpha=0.5)
# ~3 - ~5のグラフをプロット
minus.fill_between(x_m, z_fuzzy_minus(z_m,x_m),color='steelblue',label=r'$\mu_{3-5}$',alpha=0.5)
# グラフにグリッドを描画
minus.grid(which='major')
# 凡例の設定
minus.legend(loc='upper right', borderaxespad=0.3,fontsize=8)
# タイトルを設定
minus.set_title(r'$\mu_{3} - \mu_{5}$',fontsize=16)
# x軸のメモリの設定
minus.set_xticks(np.arange(-10,11,2))
minus.set_xticks(np.arange(-10,11,0.5), minor=True)


# x * y の結果をグラフにプロットする
# ~3のグラフをプロット
times.fill_between(x_t, three_array(x_t),color='deepskyblue',label=r'$\mu_3$',alpha=0.5)
# ~5のグラフをプロット
times.fill_between(x_t, five_array(x_t),color='hotpink',label=r'$\mu_5$',alpha=0.5)
# ~3 * ~5のグラフをプロット
times.fill_between(x_t, z_fuzzy_times(z_t,x_t),color='forestgreen',label=r'$\mu_{3 \times 5}$',alpha=0.5)
# グラフにグリッドを描画
times.grid(which='major')
# 凡例の設定
times.legend(loc='upper right', borderaxespad=0.3,fontsize=8)
# タイトルを設定
times.set_title(r'$\mu_{3} \times \mu_{5}$',fontsize=16)
# x軸のメモリの設定
times.set_xticks(np.arange(0,46.5,2))
times.set_xticks(np.arange(0,46.5,0.5), minor=True)


# x / y の結果をグラフにプロットする
# ~3のグラフをプロット
divide.fill_between(x_d, three_array(x_d),color='deepskyblue',label=r'$\mu_3$',alpha=0.5)
# ~5のグラフをプロット
divide.fill_between(x_d, five_array(x_d),color='hotpink',label=r'$\mu_5$',alpha=0.5)
# ~3 / ~5のグラフをプロット
divide.fill_between(x_d, z_fuzzy_divide(z_d,x_d),color='slateblue',label=r'$\mu_{3 \div 5}$',alpha=0.5)
# グラフにグリッドを描画
divide.grid(which='major')
# 凡例の設定
divide.legend(loc='upper right', borderaxespad=0.3,fontsize=8)
# タイトルを設定
divide.set_title(r'$\mu_{3} \div \mu_{5}$',fontsize=16)
# x軸のメモリの設定
divide.set_xticks(np.arange(0,11,1))
divide.set_xticks(np.arange(0,11,0.1), minor=True)

# グラフを表示する
fig.show()

# save as pdf
fig.savefig('figure.pdf',bbox_inches="tight", pad_inches=0.05)
