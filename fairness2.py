import csv

import matplotlib.pyplot as plt
import numpy
import torch
import pandas as pd
import numpy as np
import seaborn as sn
import time
import torch.nn as nn
# import numba as nb
import os
import torch.optim.lr_scheduler as lr_scheduler
from parameters import  parameters
from Monotonic import Monotonic
from Monotonic_exp import Monotonic_exp
from Monotonic_log import Monotonic_log
from Monotonic_n2 import Monotonic_n2
# import  torch.tensorflow as tf
from torch.utils.data import DataLoader



#计算积分的矩形划分
def trainset_to_x(m, n, trainset):
    flat_trainset = trainset.flatten()
    flat_trainset = np.repeat(flat_trainset[np.newaxis, :], parameters.n, axis=0)
    zeros = flat_trainset.astype(np.float32)
    # columns=zeros.shape[1]
    np.fill_diagonal(zeros[:, 0:], 0)
    for i in range(parameters.m):
        np.fill_diagonal(zeros[:, i*parameters.n:], 0)
    ts = np.linspace(zeros, flat_trainset, parameters.integral_space,True,False,np.float32,1) #将从0到 flat_trainset划分成parameters.integral_space段区间
    x=ts.flatten()
    x = x.T.reshape((parameters.n*parameters.integral_space*m),parameters.n) #(n*int*m)*n
    return x
# # n*int*m*n-->m*n*int
# def train_to_zoid(x,m):
#     x_flatten=x.flatten()
#     x_N_INT_M_N=x_flatten.reshape(parameters.n,parameters.integral_space,m,parameters.n)
#     x_N_M_INT_N=torch.transpose( x_N_INT_M_N, 1, 2)
#     x_M_N_INT_N=torch.transpose(x_N_M_INT_N, 0, 1)
#     column= x_M_N_INT_N.shape[0]
#     # print(x_M_N_INT_N.shape)
#     for i in range(column):
#         for j in range(parameters.n):
#             x_M_N_INT_N[i,0,:,j]=x_M_N_INT_N[i,j,:,j]
#     result=x_M_N_INT_N[:,0,:,:]
#     return results


def  Calutility(x,m,log):
    flat_trainset = x.flatten()
    zeros = np.zeros(len(flat_trainset)).astype(np.float32)
    ts = np.linspace(flat_trainset, zeros, parameters.integral_space).astype(np.float32)
    x_v = ts.T.reshape((m, parameters.n, parameters.integral_space))
    x_v = torch.tensor(x_v)
    if (log == 1):
        x = torch.log(x_v)
    elif (log == 2):
        x = torch.exp(x_v)
    elif (log == 3):
        x =torch.square(x_v)
    else:
        x =x
    x_flatten=x.flatten()
    y = np.zeros(len(x_flatten)).astype(np.float32)
    res = y.T.reshape((m, parameters.n, parameters.integral_space))
    # print(res)
    for k in range (m):
        sum=0
        for j in range (parameters.nA):
            sum+= parameters.W2A[k]*(parameters.W1A[k]*x[k][j][0]+parameters.b1A[k])+parameters.b2A[k]
        for j in range (parameters.nB):
            sum+=parameters.W2B[k]*(parameters.W1B[k]*x[k][j+parameters.nA][0]+parameters.b1B[k])+parameters.b2B[k]
        row=parameters.integral_space * parameters.m
        for j in range(parameters.nA):
            sum_i=sum-parameters.W2A[k]*(parameters.W1A[k]*x[k][j][0]+parameters.b1A[k])+parameters.b2A[k]
            for i in range (parameters.integral_space):
                temp=parameters.W2A[k] * (parameters.W1A[k] * x[k][j][i] + parameters.b1A[k]) + parameters.b2A[k]
                res[k][j][i]= temp/(sum_i+temp)
        for j in range(parameters.nB):
            sum_i=sum-parameters.W2B[k]*(parameters.W1B[k]*x[k][j+parameters.nA][0]+parameters.b1B[k])+parameters.b2B[k]
            for i in range (parameters.integral_space):
                temp=parameters.W2B[k] * (parameters.W1B[k] * x[k][j+parameters.nA][i] + parameters.b1B[k]) + parameters.b2B[k]
                res[k][j+parameters.nA][i]= temp/(sum_i+temp)
    res = torch.tensor(res)
    z = torch.trapezoid(torch.transpose(res, 2, 1), torch.transpose(x_v, 2, 1))
    print(z)
    integral = z.sum(1)
    return integral

def train():
    start = time.time()
    save_path = "C:/Users/Administrator/Desktop/Monotonic/" + "n=" + str(parameters.n) + "m=" + str(parameters.m) + "train=" \
                + str(parameters.train) + "t=" + str(parameters.t) + "b=" + str(parameters.beta) + 'lamb=' \
                + str(parameters.lamb) + "lamb_rate=" + str(parameters.lamb_rate) + "model_rate=" \
                + str(parameters.model_rate) + "embed=" + str(parameters.embed) + "epsilonfor_Uitility=" + str(parameters.epsilonfor_Uitility)+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print("File exists.")
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    trainseta = torch.sort((torch.rand(parameters.m, parameters.nA))*10)[0]
    trainsetb = torch.sort((torch.rand(parameters.m, parameters.nB))*8)[0]
    # trainseta = torch.sort(torch.normal(5.0, 1, (parameters.m, parameters.nA)))[0]
    # trainsetb = torch.sort(torch.normal(4.0, 1, (parameters.m, parameters.nB)))[0]
    #把a和b拼接起来
    trainset_value = torch.tensor(np.concatenate((trainseta, trainsetb), axis=1))

    #放置在设备上
    trainset_value_np = trainset_value.to('cpu')

    x_value = torch.tensor(trainset_to_x(parameters.m, parameters.n, trainset=trainset_value_np.numpy())).to(device)
    train_data = trainset_value_np.numpy()
    pd.DataFrame(train_data).to_csv(
        save_path + str(int(parameters.beta)) + "," + str(parameters.proportion_of_0) + ' trainset.csv')

    # train the network model
    lamb = torch.full((parameters.m,), parameters.lamb).to(device)

    net = Monotonic()
    net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=parameters.model_rate)
    scheduler = lr_scheduler.StepLR(trainer, step_size=500, gamma=0.5)
    obj_list = []
    m_obj_list = []
    output_list = []
    loss_list = []
    cons_list = []
    m_cons_list = []
    lag_list = []
    lamb_list = []

    for iter in range(parameters.t):
        # print("****************************************" + str(iter))
        # train model
        iteration = 0
        while iteration < parameters.train:
            output = net(trainset_value)
            # obj= -(output * trainset_value).sum(1)
            sw = output*trainset_value
            swA=sw[:, 0:parameters.nA].sum(1)
            swB=sw[:, parameters.nA:].sum(1)
            #计算积分 即 卖家的utility
            integralz=Calutility(trainset_value,parameters.m,0)
            utility= sw.sum(1)-integralz
            obj = -utility
            cons=torch.abs(swA-swB)-parameters.epsilonfor_Uitility
            lag= torch.sum(obj + lamb * cons)
            #loss =lag
            loss = torch.exp(lag / (parameters.m * parameters.n))

            obj_list.append(float(torch.mean(-obj)))
            m_obj_list.append(float(torch.max(-obj)))  # the highest obj among all exp
            output_list.append(float(torch.max(output)))  # the highest q among all agents and exp

            cons_list.append(float(torch.mean(cons)))  # should be negative
            m_cons_list.append(float(torch.max(cons)))
            lag_list.append(float(lag))
            loss_list.append(float(loss))

            if iter == 0:
                obj_best = torch.mean(-obj)
                inte_bestA = swA
                inte_bestB = swB

                obj_best_bf = 0
                inte_best_bfA = swA
                inte_best_bfB = swB

                obj_bf = 0
                inte_bfA = swA
                inte_bfB = swB

                torch.save(net, save_path + "/best_model")
                # torch.save(net, save_path + "/best_bf_model")
                # torch.save(net, save_path + "/bf_model")
            else:
                if torch.mean(-obj) >= obj_best:
                    obj_best = torch.mean(-obj)
                    inte_bestA = swA
                    inte_bestB = swB
                    torch.save(net, save_path + "/best_model")
                if (torch.mean(-obj) >= obj_best_bf) and (torch.mean(cons) <= 0):
                    obj_best_bf = torch.mean(-obj)
                    inte_best_bfA = swA
                    inte_best_bfB = swB
                    torch.save(net, save_path + "/best_bf_model")
                if (torch.mean(-obj) >= obj_bf) and (torch.max(cons) <= 0):
                    obj_bf = torch.mean(-obj)
                    inte_bfA = swA
                    inte_bfB = swB
                    torch.save(net, save_path + "/bf_model")

            trainer.zero_grad()
            loss.backward(retain_graph=True)
            #loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # clip parameter
            trainer.step()
            scheduler.step()

            torch.cuda.empty_cache()

            iteration += 1

        if (iter + 1) % 1000 == 0:
            print("Loss for iteration %d is %f" % (iter + 1, loss))
            pass
        torch.cuda.empty_cache()


        y0 = net(trainset_value).detach()
        y= y0*trainset_value
        swA=y[:,0:parameters.nA].sum(1)
        swB=y[:,parameters.nA:].sum(1)
        lamb = lamb + parameters.lamb_rate * (torch.abs(swA - swB)  - parameters.epsilonfor_Uitility)


        for i in range(len(lamb)):
            if lamb[i] <= 0:
                lamb[i] = 1e-4
        lamb_list.append(float(torch.mean(lamb)))

        torch.cuda.empty_cache()

    end = time.time()
    print("time: ", end - start)

    last_allocation = net(trainset_value).detach()
    last_integralz = Calutility(trainset_value,parameters.m,0)
    last_utility = sw.sum(1) - last_integralz
    last_obj = (last_allocation * trainset_value).sum(1) - last_utility  #这里记录的是正的obj =social welfare
    last_cons = torch.abs(swA - swB) - parameters.epsilonfor_Uitility

    #last_allocation = net(trainset).detach()
    #last_obj = ((last_allocation * torch.log((1 + last_allocation) / (1 - last_allocation))).sum(1))
    #last_cons = (trainset * last_allocation * torch.log((1 + last_allocation) / (1 - last_allocation))).sum(1) + integral #这里为什么不用重新计算integral

    print("(Trainset) (last) Total (expected) social welfare: ", torch.mean(last_obj))
    print("(Trainset) (last) Total (expected) fairness difference : ", torch.mean(last_cons))

    best_model = torch.load(save_path + 'best_model')
    best_allocation = best_model(trainset_value).detach()
    best_integralz = Calutility(trainset_value,parameters.m,0)
    best_utility = sw.sum(1) - best_integralz
    best_obj = (best_allocation * trainset_value).sum(1) -  best_utility
    best_cons = inte_bestA-inte_bestB

    print("(Trainset) (best) Total (expected) social welfare: ", torch.mean(best_obj))
    print("(Trainset) (best) Total (expected) fairness difference: ", torch.mean(best_cons))

    best_bf_model = torch.load(save_path + 'best_bf_model')
    best_allocation_bf = best_bf_model(trainset_value).detach()
    best_bf_integralz = Calutility(trainset_value,parameters.m,0)
    best_bf_utility = sw.sum(1) - best_bf_integralz
    best_obj_bf =( best_allocation_bf * trainset_value).sum(1) - best_bf_utility
    best_cons_bf = inte_best_bfA-inte_best_bfB

    print("(Trainset) (best bf) Total (expected) social welfare: ", torch.mean(best_obj_bf))
    print("(Trainset) (best bf) Total (expected) fairness difference: ", torch.mean(best_cons_bf))
    print("(Trainset) (best bf) Total (expected) fairness difference: ", best_cons_bf)

    bf_model = torch.load(save_path + 'bf_model')
    allocation_bf = bf_model(trainset_value).detach()
    bf_integralz = Calutility(trainset_value,parameters.m,0)
    bf_utility = sw.sum(1) - bf_integralz
    obj_bf = (allocation_bf * trainset_value).sum(1) - bf_utility
    cons_bf = inte_bfA- inte_bfB

    print("(Trainset) (bf) Total (expected) social welfare:", torch.mean(obj_bf))
    print("(Trainset) (bf) Total (expected) fairness difference: ", cons_bf)
    print(torch.cuda.max_memory_allocated())

    torch.save(net, save_path + "/model")
    torch.save(loss_list, save_path + "/loss")
    torch.save(obj_list, save_path + "/mean_obj")
    torch.save(m_obj_list, save_path + "/max_obj")
    torch.save(output_list, save_path + "/allocation")
    torch.save(cons_list, save_path + "/constraint")
    torch.save(m_cons_list, save_path + "/max constraint")
    torch.save(lag_list, save_path + "/lag")
    torch.save(lamb_list, save_path + "/lambda")
    torch.save(trainset_value, save_path + "/trainset_log")
    torch.save(trainset_value, save_path + "/trainset_value")

    with open(save_path + "/obj,mobj,cons,lag,loss,lambda.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Interation', 'mean obj', 'max obj', 'constraint', 'max cons', 'lag', "loss", "lambda"])
        for i in range(parameters.t):
            writer.writerow(
                [i, obj_list[i], m_obj_list[i], cons_list[i], m_cons_list[i], lag_list[i], loss_list[i], lamb_list[i]])

    return net

if __name__== "__main__":
    train()

