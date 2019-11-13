import matplotlib.pyplot as plt

#ファジイ数生成
class fuzzyset:
    def __init__(self,t):
        self.t = t

    def makeA(self):
        ua = []
        for i in self.t:
            if i < 1.0:
                ua.append(0)
            elif i >= 1.0 and i < 3.0:
                ua.append((i-1) / 2.0)
            elif i >= 3.0 and i < 5.0:
                ua.append((5-i) / 2.0)
            elif i >= 5.0:
                ua.append(0)

        return ua

    def makeB(self):
        ub = []
        for i in self.t:
            if i < 2.0:
                ub.append(0)
            elif i >= 2.0 and i < 5.0:
                ub.append((i-2) / 3.0)
            elif i >= 5.0 and i < 8.0:
                ub.append((8-i) / 3.0)
            elif i >= 8.0:
                ub.append(0)

        return ub

#各演算を行うクラス
class operation:
    def __init__(self):
        self.mu_min = []

    def sum(self,t,ua,ub):
        mu_max = []
        for k in t:
            for i in range(100,501):
                self.mu_min.append(min(ua[i],ub[int(100*k) - i]))

            if self.mu_min != []:
                mu_max.append(max(self.mu_min))
            else:
                mu_max.append(0)
            self.mu_min.clear()

        return mu_max

    def subtruct(self,t,ua,ub):
        mu_max = []

        for k in t:
            for i in range(800,1201):
                if i - int(100*k) <= 1500:
                    self.mu_min.append(min(ua[i],ub[i - int(100*k)]))
            if self.mu_min != []:
                mu_max.append(max(self.mu_min))
            else:
                mu_max.append(0)
            self.mu_min.clear()

        return mu_max

    def multiply(self,t,ua,ub):
        mu_max = []
        for k in t:
            for i in range(100,501):
                self.mu_min.append(min(ua[i],ub[int(int(10000*k) / i)]))
            if self.mu_min != []:
                mu_max.append(max(self.mu_min))
            else:
                mu_max.append(0)
            self.mu_min.clear()

        return mu_max

    def divide(self,t,ua,ub):
        mu_max = []
        for k in t:
            for i in range(100,501):
                if k == 0:      # i / 0 を避ける
                    break
                if int(i / k) <= 800:
                    self.mu_min.append(min(ua[i],ub[int(i / k)]))
            if self.mu_min != []:
                mu_max.append(max(self.mu_min))
            else:
                mu_max.append(0)
            self.mu_min.clear()

        return mu_max

t_sum = [i / 100.0 for i in range(0,1301)]
t_sub = [i / 100.0 for i in range(-700,801)]
t_mtp = [i / 100.0 for i in range(0,4001)]
t_div = [i / 100.0 for i in range(0,801)]

A_sum,B_sum = fuzzyset(t_sum),fuzzyset(t_sum)
A_sub,B_sub = fuzzyset(t_sub),fuzzyset(t_sub)
A_mtp, B_mtp = fuzzyset(t_mtp),fuzzyset(t_mtp)
A_div, B_div = fuzzyset(t_div), fuzzyset(t_div)

ua_sum, ub_sum = A_sum.makeA(), B_sum.makeB()
ua_sub, ub_sub = A_sub.makeA(), B_sub.makeB()
ua_mtp, ub_mtp = A_mtp.makeA(), B_mtp.makeB()
ua_div, ub_div = A_div.makeA(), B_div.makeB()

ope = operation()

uapb = ope.sum(t_sum,ua_sum,ub_sum)
uasb = ope.subtruct(t_sub,ua_sub,ub_sub)
uamb = ope.multiply(t_mtp,ua_mtp,ub_mtp)
uadb = ope.divide(t_div,ua_div,ub_div)

plt.figure(figsize=(6,4))
plt.subplot(2,2,1)
plt.plot(t_sum,ua_sum,label=r"$\mu_\tilde{3}$")
plt.plot(t_sum,ub_sum,label=r"$\mu_\tilde{5}$")
plt.plot(t_sum,uapb,label=r"$\mu_{\tilde{3}+\tilde{5}}$")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2,2,2)
plt.plot(t_sub,ua_sub,label=r"$\mu_\tilde{3}$")
plt.plot(t_sub,ub_sub,label=r"$\mu_\tilde{5}$")
plt.plot(t_sub,uasb,label=r"$\mu_{\tilde{3}-\tilde{5}}$")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2,2,3)
plt.plot(t_mtp,ua_mtp,label=r"$\mu_\tilde{3}$")
plt.plot(t_mtp,ub_mtp,label=r"$\mu_\tilde{5}$")
plt.plot(t_mtp,uamb,label=r"$\mu_{\tilde{3}\times\tilde{5}}$")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2,2,4)
plt.plot(t_div,ua_div,label=r"$\mu_\tilde{3}$")
plt.plot(t_div,ub_div,label=r"$\mu_\tilde{5}$")
plt.plot(t_div,uadb,label=r"$\mu_{\tilde{3}\div\tilde{5}}$")
plt.legend(loc="upper right")
plt.grid()
plt.show()
