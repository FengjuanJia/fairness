# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from parameters import parameters
import torch
import os.path
import torch
import numpy as np

from scipy import integrate
import scipy.stats as stats
import math
import random
import pandas as pd

def basic(data,PrA):
    result = np.zeros(parameters.test)
    swA = np.zeros(parameters.test)
    swB = np.zeros(parameters.test)
    obj = np.zeros(parameters.test)
    PrB=1-PrA
    for i in range(parameters.test):
        WA = data[i][parameters.nA - 1]
        WB = data[i][parameters.n - 1]
        pA = data[i][parameters.nA - 2]
        pB = data[i][parameters.n - 2]
        swA[i] = WA * PrA
        swB[i] = WB * PrB
        obj[i] = pA * PrA + pB * PrB
        result[i] = swA[i] - swB[i]
    return swA, swB, result, obj


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # indexa = torch.randint(0,parameters.nA+1,(150,))
    # # indexa = torch.randint(low=0,high=parameters.nA+1,sizes=(150,))
    # indexb = torch.randint(parameters.nA,parameters.n+1,(350,))
    # indexa=torch.sort(indexa)
    # indexb=torch.sort(indexb)
    # print(indexa[0])
    index=torch.rand(parameters.m, parameters.n)
    print(index.shape)
    testset = pd.read_csv('C:/Users/Administrator/Desktop/testset1-normal.csv')
    test_data = np.array(testset)
    testset = torch.tensor(test_data).to(torch.float32)
    test_data = testset.numpy()  # 获取测试数
    j=0;
    i=0;
    k1=0;
    k2=0;
    # data=[[]for i in range(parameters.test)]
    # train=[[]for i in range(parameters.test)]
    m = int(parameters.test)
    n = 500
    PrA = [0]*parameters.test;
    data=[[0]* int(n) for _ in range(m)]
    train=[[0]* int(n) for _ in range(m)];
    for t in range(parameters.test):
        k1 = 0;
        k2 = 0;
        for i in range(parameters.nA):
            if ((index[t][i]<0.5) & (k1 < parameters.nA/2))|(k2 >= parameters.nA/2):
                train[t][k1]=testset[t][i];
                k1 = k1 + 1;
            else:
                if(k2 < parameters.nA/2):
                    data[t][k2]=testset[t][i];
                    k2 = k2 + 1;
        for i in range(parameters.nB):
            if ((index[t][i+parameters.nA]<0.5) & (k1<parameters.n/2))|(k2 >= parameters.n/2):
                train[t][k1]=testset[t][i+parameters.nA];
                k1 = k1 + 1;
            else:
                if (k2 <parameters.n/2):
                     data[t][k2]=testset[t][i+parameters.nA];
                     k2 = k2 + 1;
        nAindex= int(parameters.nA / 2 -1);
        nBindex= int(parameters.n / 2 -1);
        # print("*************************")
        # print(train[t][nAindex])
        # print(train[t][nBindex])
        PrA[t] = train[t][nAindex] + train[t][nBindex]
        PrA[t] = (parameters.epsilon + train[t][nBindex]) / PrA[t]
        # print(PrA[t])

    result = np.zeros(m)
    swA = np.zeros(m)
    swB = np.zeros(m)
    obj = np.zeros(m)
    PrB = [0]*parameters.test;
    index = int(parameters.nA/2)-1
    for i in range(int(m)):
        PrB[i] = 1 - PrA[i]
        WA = data[i][index]
        WB = data[i][index]
        pA = data[i][index-1]
        pB = data[i][index-1]
        swA[i] = WA * PrA[i]
        swB[i] = WB * PrB[i]
        obj[i] = pA * PrA[i] + pB * PrB[i]
        result[i] = swA[i] - swB[i]

    print(np.mean(np.abs(result)))
    print(np.mean(obj))
    print(np.mean(swA)+np.mean(swB))
    print(np.mean(swA))

    different_sw_basic = pd.DataFrame(result, columns=["different_basic"])
    obj_S_basic = pd.DataFrame(obj, columns=["obj_S_basic"])
    B_basic_sw = pd.DataFrame( swA, columns=["B_basic_sw"])
    A_basic_sw = pd.DataFrame( swB, columns=["A_basic_Sw"])
    parameters.PrA = np.mean(PrA);
    fair = pd.concat([different_sw_basic], axis=1)
    save_path = parameters.save_path + "data=" + str(parameters.sum_dataset_path) + ",sum/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print("File exists.")
    f_para = open(save_path + "/parameters.txt", "w");
    f_para.write("PrA = " + str(parameters.PrA) + "\n")
    fair.to_csv(save_path + '/fair.csv')
    UA = pd.concat(
        [A_basic_sw], axis=1)
    UA.to_csv(save_path + '/UA.csv')
    UB = pd.concat(
        [ B_basic_sw], axis=1)
    UB.to_csv(save_path + '/UB.csv')
    OBJ = pd.concat(
        [obj_S_basic], axis=1)
    OBJ.to_csv(save_path + '/OBJ.csv')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
