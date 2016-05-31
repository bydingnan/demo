#encoding=utf-8
import os
import random

#切词共有4个状态，Begin、Middle、End、Single，代表词的开头、中间、结尾和单字词

def baum_welch(pi, A, B):
    f = file("input")
    sentence = f.read()[3:].decode('utf-8') #跳过文件头
    f.close()
    T = len(sentence)  #观测序列
    alpha = [[0 for i in range(4)] for t in range(T)]
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T-1)]

    for time in range(100):
        calc_alpha(pi, A, B, sentence, alpha) #alpha(t, i): 前向算法,给定lambda
        calc_beta(pi, A, B, sentence, beta) #beta(t, i): 后向算法,  给定lambda和t时刻状态i
        calc_gamma(alpha, beta, gamma)  # gamma(t, i): 给定lambda和O，t时刻状态为i的概率
        calc_ksi(alpha, beta, A, B, sentence, ksi) #ksi(t, i, j): 给定lambda和O,i到j的转换概率
        bw(pi, A, B, alpha, beta, gamma, ksi, sentence) #baum_welch 算法


def calc_alpha(pi, A, B, o, alpha):
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t-1][j] + A[j][i])
            alpha[t][i] = log_sum[temp]
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T-2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
            beta[t][i] += log_sum(temp)

def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T-1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s

def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T-1)]
    s2 = [0 for x in range(T-1)]
    for i in range(4):
        for j in range(4):
            for t in range(T-1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        for k in range(65536):
            valid = 0
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)


def viterbi(pi, A, B, o):
    T = len(o)   #观测序列
    delta = [[0 for i in range(4)] for t in range(T)] #保存极值
    pre = [[0 for i in range(4)] for t in range(T)]  #前一个状态
    for i in range(4):
        delta[0][i] = pi[i] + B[i][ord(o[0])]
    for t in range(1, T):
        for i in range(4):
            delta[t][i] = delta[t-1][0] + A[0][i]
            for j in range(1,4):
                vj = delta[t-1][j] + A[j][i]
                if delta[t][i] < vj:
                    delta[t][i] = vj
                    pre[t][i] = j
            delta[t][i] += B[i][ord(o[t])]
    decode = [-1 for t in range(T)]  # 解码，回溯回去倒推最大路径
    q = 0
    for i in range(1, 4):
        if delta[T-1][i] > delta[T-1][q]:
            q = i
    decode[T-1] = q
    for t in range(T-2, -1, -1):
        q = pre[t+1][q]
        decode[t] = q
    return decode


#分词
def segment(sentence, decode):
    N = len(sentence)
    i = 0
    while i < N:   #B/M/E/S
        if decode[i] == 0 or decode[i] == 1:    #Begin
            j = i + 1
            while j < N:
                if decode[j] == 2:
                    break
                j += 1
            print sentence[i:j+1], "/"
            i = j + 1
        elif decode[i] == 3 or decode[i] == 2:  #Single
            print sentence[i:i+1], "/"
            i += 1
        else:
            print "Error:", i, decode[i]
            i += 1


if __name__ == '__main__':
    #初始化pi, A, B
    pi = [random.random]
    pi = [random.random() for x in range(4)]  #初始分布
    log_normalize(pi)
    A = [[random.random() for y in range(4)] for x in range(4)]   #状态转移矩阵
    A[0][0] = A[0][3] = A[1][0] = A[1][3] = A[2][1] = A[2][2] = A[3][1] = A[3][2] = 0 #不可能事件
    B = [[random.random() for y in range(65536)] for x in range(4)]  #观察矩阵
    for i in range(4):
        log_normalize(A[i])
        log_normalize(B[i])
    baum_welch(pi, A, B)
    save_parameter(pi, A, B)
    
    
