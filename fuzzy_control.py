# -*- coding: utf-8 -*-
#
#   fuzzy control
#   ファジィ工学特論 課題3 ファジィ制御 ソースコード

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)
# use latex for font rendering
plt.rcParams["text.usetex"] = True
sns.set()
sns.set_style('whitegrid',{'grid.linestyle':'--'})
sns.set_palette("GnBu_r",3)
#sns.set_palette("hls",2)
'''
# 求めた解をグラフにプロットする為のクラス
class Graphplot() :
'''
# 前件部
# ekやd_ekのメンバーシップ関数を定義するクラス
class Condition() :
    # コンストラクタを実行
    @classmethod
    def __init__(cls,S) :
        # S = [NB,NS,ZE,PS,PB]
        # 例1 S_ek = [-r,-r/2,0,r/2,r]
        # 例2 S_d_ek = [-1,-0.05,0,0.05,1]
        cls.S = S
        #print(S)
        #print(cls.S)

    # NBのメンバーシップ関数
    @classmethod
    def NB(cls,e) :
        #print(cls.S)
        if e < cls.S[0] :
            return 1
        elif cls.S[0] <= e < cls.S[1] :
            return (cls.S[1] - e) / (cls.S[1] - cls.S[0])
        else :
            return 0

    # NSのメンバーシップ関数
    @classmethod
    def NS(cls,e) :
        if cls.S[0] <= e < cls.S[1] :
            return (e-cls.S[0]) / (cls.S[1] - cls.S[0])
        elif cls.S[1] <= e < cls.S[2] :
            return (cls.S[2] - e) / (cls.S[2] - cls.S[1])
        else :
            return 0

    # ZEのメンバーシップ関数
    @classmethod
    def ZE(cls,e) :
        if cls.S[1] <= e < cls.S[2] :
            return (e-cls.S[1]) / (cls.S[2] - cls.S[1])
        elif cls.S[2] <= e < cls.S[3] :
            return (cls.S[3]-e) / (cls.S[3] - cls.S[2])
        else :
            return 0

    # PSのメンバーシップ関数
    @classmethod
    def PS(cls,e) :
        if cls.S[2] <= e < cls.S[3] :
            return (e-cls.S[2]) / (cls.S[3] - cls.S[2])
        elif cls.S[3] <= e < cls.S[4] :
            return (cls.S[4]-e) / (cls.S[4] - cls.S[3])
        else :
            return 0

    # PBのメンバーシップ関数
    @classmethod
    def PB(cls,e) :
        if e < cls.S[3] :
            return 0
        elif cls.S[3] <= e < cls.S[4] :
            #print("PB = {}".format(e-cls.S[4]))
            #print("PB = {}".format(cls.S[4]-cls.S[3]))
            return (e-cls.S[3]) / (cls.S[4] - cls.S[3])
        else :
            return 1

    # ekやd_ekのメンバーシップ関数をプロットするメソッド
    @classmethod
    def Graph_plt(cls) :
        pass

'''
# 重心法でファジィ推論をする場合に使用するクラス
class Conclusion() :
'''

# ファジィ推論を行うクラス
class Fuzzy_inference() :
    # コンストラクタを実行する
    def __init__(self,S) :
        # d_ukのメンバーシップ関数
        # S = [NB,NM,NS,ZE,PS,PM,PB]
        # 例1 S_d_uk = [-1.5,-1,-0.5,0,0.5,1,1.5]
        #self.S = S
        self.NB = S[0]
        self.NM = S[1]
        self.NS = S[2]
        self.ZE = S[3]
        self.PS = S[4]
        self.PM = S[5]
        self.PB = S[6]

    # ファジィ制御器の制御ルールを作成するメソッド
    def makerule(self) :
        # それぞれの条件値を引き出す
        NB = self.NB
        NM = self.NM
        NS = self.NS
        ZE = self.ZE
        PS = self.PS
        PM = self.PM
        PB = self.PB

        # 行ラベル e(k) = [NB,NS,ZE,PS,PB]
        # 列ラベル d_e(k) = [NB,NS,ZE,PS,PB]
        self.rule = np.array([[ZE,PS,PM,PB,PB],
                              [NS,ZE,PS,PM,PB],
                              [NM,NS,ZE,PS,PM],
                              [NB,NM,NS,ZE,PS],
                              [NB,NB,NM,NS,ZE]])

        return self.rule

# 離散化して数値シミュレーションをするクラス
class Discretization() :
    # グローバル変数の定義
    u_pre = 0
    # x(k+1) = Px(k) + qu(k)
    # 離散化の為の関数 Pの演算
    @classmethod
    def P(cls,T,N,I,A) :
        # 階乗の計算 math.factorial()
        # 単位行列を代入
        P = I
        for i in range(1,N+1) :
            P = P + ((A**i)*(T**i))/math.factorial(i)
            #print(math.factorial(i))
        #print(P)
        # P = e^{AT} ~= I+AT+1/(2!)*A^2*T^2+...+1/(N!)*A^N*T^N
        cls.P = P

    # 離散化の為の関数 qの演算
    @classmethod
    def q(cls,T,N,I,A,b) :
        # 階乗の計算 math.factorial()
        # 単位行列を代入
        R = I
        for i in range(1,N+1) :
            #print(i)
            R = R + ((A**i)*(T**i))/math.factorial(i+1)
        R = R*T
        # R ~= T{I+1/(2!)*AT+1/(3!)*A^2*T^2+...+1/((N+1)!)*A^N*T^N}
        q = np.dot(R,b)
        #print(q)
        cls.q = q

    # メンバーシップ関数を定義するメソッド
    @classmethod
    def define_membership(cls,r) :

        # e(k)の前件部を生成する
        S_ek1 = [-r, -r/2, 0, r/2, r]
        S_ek2 = [-r+10, -r/2, 0, r/2, r-10]
        # e(k)のメンバーシップ関数を生成する
        cls.member1 = Condition(S_ek1)

        # Δe(k)の前件部を生成する
        S_d_ek1 = [-r, -r/3, 0, r/3, r]
        S_d_ek2 = [-0.02, -0.018, 0, 0.018, 0.02]
        # Δe(k)のメンバーシップ関数を生成する
        cls.member2 = Condition(S_d_ek1)

        # Δu(k)の後件部を生成する
        S_d_uk1 = [-0.004,-0.0008,-0.00001,0,0.00001,0.0008,0.004]
        S_d_uk2 = [-0.0008,-0.0004,-0.00001,0,0.00001,0.0004,0.0008]
        # Δu(k)のメンバーシップ関数を生成する
        cls.member3 = Fuzzy_inference(S_d_uk1)
        # ファジィ推論用の配列を生成する
        cls.member3.makerule()

    # 入力 u(t)=1 単位ステップ入力
    @classmethod
    def u_step(cls,t) :
        return 1

    # ファジィ推論で推定した入力u(t)をreturn
    @classmethod
    def u_fuzzy(cls,k,e,d_e) :
        # d_eの符号を反転させる
        d_e = -d_e
        # fuzzy推論用のルールをローカル変数に代入する
        rule = cls.member3.rule
        # ek用のndarrayを生成する
        array_e = np.zeros(5)
        # NBの値を代入する
        array_e[0] = cls.member1.NB(e)
        # NSの値を代入する
        array_e[1] = cls.member1.NS(e)
        # ZEの値を代入する
        array_e[2] = cls.member1.ZE(e)
        # PSの値を代入する
        array_e[3] = cls.member1.PS(e)
        # PBの値を代入する
        array_e[4] = cls.member1.PB(e)

        # d_ek用のndarrayを生成する
        array_d_e = np.zeros(5)
        # NBの値を代入する
        array_d_e[0] = cls.member2.PB(d_e)
        # NSの値を代入する
        array_d_e[1] = cls.member2.PS(d_e)
        # ZEの値を代入する
        array_d_e[2] = cls.member2.ZE(d_e)
        # PSの値を代入する
        array_d_e[3] = cls.member2.NS(d_e)
        # PBの値を代入する
        array_d_e[4] = cls.member2.NB(d_e)

        # 荷重平均法の分子を求める numerator : 分子
        numerator = 0
        for i in range(5) :
            for j in range(5) :
                numerator += array_d_e[i] * array_e[j] * rule[i][j]

        # 荷重平均法の分母を求める denominator : 分母
        denominator = 0
        for i in range(5) :
            for j in range(5) :
                denominator += array_d_e[i] * array_e[j]

        # Δu(k)を求める
        d_uk = numerator / denominator

        # uを計算する
        u = d_uk + cls.u_pre

        # cls.u_preに現在のuを代入する
        cls.u_pre = u
        '''
        # d_ukをreturnする
        if k % 10 == 0 :
            print("---------------------------------")
            print("{}[s]".format(k))
            print("e(k) = {}".format(e))
            print("e(k)のメンバーシップ関数値 \n= {}".format(array_e))
            print("Δe(k) = {}".format(-d_e))
            print("Δe(k)のメンバーシップ関数値 \n= {}".format(array_d_e))
            #print("分子 = {}".format(numerator))
            #print("分母 = {}".format(denominator))
            print("入力 u(k) = {}".format(u))
        '''
        return u


    # x(k+1) = Px(k) + qu(k)
    # u(k) = 1
    @classmethod
    def discretize(cls,T,kf,N,I,A,b,x,r,mode) :
        # ファジィ制御用のメンバーシップ関数を定義する
        cls.define_membership(r)
        # システムの入力u(t) u:操作量(V)
        cls.P(T,N,I,A) # Pの演算
        cls.q(T,N,I,A,b) # qの演算
        e_pre = 0 # e(k-1)
        #print(cls.P,cls.q)
        k = np.arange(0,kf,T) # 離散時間分の配列を生成する
        y = np.arange(0,kf,T) # グラフ用のx1を格納していく配列
        # print(x1[0])
        #離散化して2次システムを解く
        if mode == 'step' :
            for i in range(len(k)) :
                #print(x1[i])
                y[i] = x[0]
                # a.ravel()で生成された2次元配列を1次元化する
                x = (np.dot(cls.P,x) + np.dot(cls.q,cls.u_step(k[i]))).ravel()

        elif mode == 'fuzzy' :
            for i in range(len(k)) :
                #print("{}step".format(i))
                y[i] = x[0]
                e = r - y[i]
                d_e = e - e_pre
                #u = cls.u_fuzzy(k[i],e,d_e) + u_pre
                x = (np.dot(cls.P,x) + np.dot(cls.q,cls.u_fuzzy(k[i],e,d_e))).ravel()
                e_pre = e
                #u_pre = u

        # X軸の範囲
        plt.xlim(0,kf)
        # Y軸の範囲
        plt.ylim(0,65)
        # グラフのプロット
        plt.plot(k,y,label='Discretization')
        # 凡例の表示
        plt.legend()
        # 目標値に破線を引く
        plt.axhline(r, color="b", linestyle="--")
        # プロット表示(設定の反映)
        plt.show()

# 4次のRunge-Kutta法を使って数値シミュレーションをするクラス
class Rungekutta() :
    # グローバル変数の定義
    u_pre = 0
    # メンバーシップ関数を定義するメソッド
    @classmethod
    def define_membership(cls,r) :

        # e(k)の前件部を生成する
        S_ek1 = [-r, -r/2, 0, r/2, r]
        #S_ek2 = [-r+10, -r/2, 0, r/2, r-10]
        # e(k)のメンバーシップ関数を生成する
        cls.member1 = Condition(S_ek1)

        # Δe(k)の前件部を生成する
        S_d_ek1 = [-r, -r/3, 0, r/3, r]
        #S_d_ek2 = [-0.02, -0.018, 0, 0.018, 0.02]
        # Δe(k)のメンバーシップ関数を生成する
        cls.member2 = Condition(S_d_ek1)

        # Δu(k)の後件部を生成する
        S_d_uk1 = [ -0.001, -0.00021, -0.0000008, 0, 0.0000008, 0.00021, 0.001]
        #S_d_uk2 = [-0.0008,-0.0004,-0.00001,0,0.00001,0.0004,0.0008]
        # Δu(k)のメンバーシップ関数を生成する
        cls.member3 = Fuzzy_inference(S_d_uk1)
        # ファジィ推論用の配列を生成する
        cls.member3.makerule()

    # 入力u(t)
    @classmethod
    def u_step(cls,t) :
        return 1

    # ファジィ推論で推定した入力u(t)をreturn
    @classmethod
    def u_fuzzy(cls,k,e,d_e) :
        # d_eの符号を反転させる
        d_e = -d_e
        # fuzzy推論用のルールをローカル変数に代入する
        rule = cls.member3.rule
        # ek用のndarrayを生成する
        array_e = np.zeros(5)
        # NBの値を代入する
        array_e[0] = cls.member1.NB(e)
        # NSの値を代入する
        array_e[1] = cls.member1.NS(e)
        # ZEの値を代入する
        array_e[2] = cls.member1.ZE(e)
        # PSの値を代入する
        array_e[3] = cls.member1.PS(e)
        # PBの値を代入する
        array_e[4] = cls.member1.PB(e)

        # d_ek用のndarrayを生成する
        array_d_e = np.zeros(5)
        # NBの値を代入する
        array_d_e[0] = cls.member2.PB(d_e)
        # NSの値を代入する
        array_d_e[1] = cls.member2.PS(d_e)
        # ZEの値を代入する
        array_d_e[2] = cls.member2.ZE(d_e)
        # PSの値を代入する
        array_d_e[3] = cls.member2.NS(d_e)
        # PBの値を代入する
        array_d_e[4] = cls.member2.NB(d_e)

        # 荷重平均法の分子を求める numerator : 分子
        numerator = 0
        for i in range(5) :
            for j in range(5) :
                numerator += array_d_e[i] * array_e[j] * rule[i][j]

        # 荷重平均法の分母を求める denominator : 分母
        denominator = 0
        for i in range(5) :
            for j in range(5) :
                denominator += array_d_e[i] * array_e[j]

        # Δu(k)を求める
        d_uk = numerator / denominator

        # uを計算する
        u = d_uk + cls.u_pre

        # cls.u_preに現在のuを代入する
        cls.u_pre = u

        '''
        # d_ukをreturnする
        if k % 10 == 0 :
            print("---------------------------------")
            print("{}[s]".format(k))
            print("e(k) = {}".format(e))
            print("e(k)のメンバーシップ関数値 \n= {}".format(array_e))
            print("Δe(k) = {}".format(-d_e))
            print("Δe(k)のメンバーシップ関数値 \n= {}".format(array_d_e))
            #print("分子 = {}".format(numerator))
            #print("分母 = {}".format(denominator))
            print("入力 u(k) = {}".format(u))
        '''
        return u

    # 4次のRunge-Kutta法を使って単位ステップ応答を確認する
    @classmethod
    def rungekutta(cls,h,kf,A,b,x,r,mode) :
        # ファジィ制御用のメンバーシップ関数を定義する
        cls.define_membership(r)
        k = np.arange(0,kf,h) # 離散時間分の配列を生成する
        y = np.arange(0,kf,h) # グラフ用のx1を格納していく配列
        y2 = np.arange(0,kf,h) # グラフ用のx2を格納していく配列
        e_pre = 0
        # 4次のRunge-Kutta法を使って2次システムを解く
        if mode == 'step' :
            for i in range(len(k)) :
                y[i] = x[0]
                k1 = h*(np.dot(A,x) + np.dot(b,cls.u_step(k[i]))) # k1
                k2 = h*(np.dot(A,x+k1/2) + np.dot(b,cls.u_step(k[i]+h/2))) # k2
                k3 = h*(np.dot(A,x+k2/2) + np.dot(b,cls.u_step(k[i]+h/2))) # k3
                k4 = h*(np.dot(A,x+k3) + np.dot(b,cls.u_step(k[i]+h))) # k4
                x = x + (k1 + 2*k2 + 2*k3 + k4)/6

        elif mode == 'fuzzy' :
            for i in range(len(k)) :
                y[i] = x[0]
                y2[i] = x[1]
                e = r - y[i]
                d_e = e - e_pre
                k1 = h*(np.dot(A,x) + np.dot(b,cls.u_fuzzy(k[i],e,d_e))) # k1
                k2 = h*(np.dot(A,x+k1/2) + np.dot(b,cls.u_fuzzy(k[i]+h/2,e,d_e))) # k2
                k3 = h*(np.dot(A,x+k2/2) + np.dot(b,cls.u_fuzzy(k[i]+h/2,e,d_e))) # k3
                k4 = h*(np.dot(A,x+k3) + np.dot(b,cls.u_fuzzy(k[i]+h,e,d_e))) # k4
                x = x + (k1 + 2*k2 + 2*k3 + k4)/6
                e_pre = e

        # X軸の範囲
        plt.xlim(0,kf)
        # Y軸の範囲
        plt.ylim(0,65)
        # Y軸のラベル
        # 目標値に破線を引く
        plt.axhline(r, color="hotpink", linestyle="--")
        # グラフのプロット
        plt.plot(k,y,label= r"4thRunge-kutta")
        plt.plot(k,y2)
        # 凡例の表示
        plt.legend()
        # プロット表示(設定の反映)
        plt.show()


# fuzzy_control.pyのソースコード内で呼び出された場合のみ実行.
if __name__ == '__main__' :

    # 状態方程式 d/dt{x(t)} = Ax(t) + bu(t)
    # G = (35.78)/{(1.73s+1)(16.85s+1)}
    # 初期値 x(0)
    x = np.array([0.0,0.0]) # x1:速度(km/h),x2:加速度
    kf = 120 # k[s]までサンプリングする

    '''--------------------------------------
        A = np.array([[     0.0,     1.0],
                      [ -0.0343, -0.6374]])
    --------------------------------------'''
    A = np.array([[          0.0,             1.0],
                  [-1/(29.1505), -18.58/(29.1505)]])

    '''--------------------------------------
        b = np.array([0.0,1.227])
    --------------------------------------'''
    b = np.array([0.0,35.78/(29.1505)])

    # 厳密な離散化
    N = 20 # 無限級数の有限値
    T = 0.01 # サンプリング周期
    # 単位行列
    I = np.array([[ 1.0, 0.0],
                  [ 0.0, 1.0]])

    # 4次のRunge-kutta
    h = 0.01 # サンプリング周期

    # Discretization()の生成
    Dis_step = Discretization()
    Dis_fuzzy = Discretization()
    # Rungekutta()の生成
    Run_step = Rungekutta()
    Run_fuzzy = Rungekutta()

    '''
    # 状態方程式の離散化し,単位ステップ応答を確認する
    mode = 'step'
    r = 35.78
    Dis_step.discretize(T,kf,N,I,A,b,x,r,mode)
    '''

    # 状態方程式の離散化をし,ファジィ制御を行う
    mode = 'fuzzy'
    r = 60.0
    Dis_fuzzy.discretize(T,kf,N,I,A,b,x,r,mode)

    '''
    # 4次のRunge-Kutta法を使って,単位ステップ応答を確認する
    mode = 'step'
    r = 35.78
    Run.rungekutta(h,kf,A,b,x,r,mode)
    '''

    # 4次のRunge-Kutta法を使って,ファジィ制御を行う
    mode = 'fuzzy'
    r = 60.0
    Run_fuzzy.rungekutta(h,kf,A,b,x,r,mode)
