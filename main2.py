
"""

"""

import os.path
import torch
import numpy as np
from scipy import integrate
import scipy.stats as stats
import math
import random
import pandas as pd
import seaborn as sn
from parameters import parameters
from fairness2 import Calutility
from Monotonic import Monotonic
from Monotonic_log import Monotonic_log
import matplotlib.pyplot as plt

def accuracy(network, testset, save_path):
    # get allocation
    allocation = network(testset).detach()

    # obj= -(output * trainset_value).sum(1)
    sw = allocation *  testset
    swA = sw[:, 0:parameters.nA].sum(1)
    swB = sw[:, parameters.nA:].sum(1)
    # 计算积分 即 卖家的utility
    integralz = Calutility(testset,parameters.test,0)
    utility = sw.sum(1) - integralz
    obj_test = utility
    # cons = torch.abs(swA - swB) - parameters.epsilonfor_Uitility
    # loss =lag
    # get obj(socialwelfare)
    # print("(Testset) Total (expected) obj (social welfare): ", obj_test)
    expected_different_utility=torch.abs(swA-swB)

    # print("Group A utility: ", integralzA)
    # print("Group B utility: ", integralzB)
    print("(Testset) Total (expected) different utility: ",  expected_different_utility)

    return swA.numpy(), swB.numpy(), expected_different_utility.numpy(), obj_test.numpy(), allocation
def accuracy_log(network, testset, save_path,testset_v):
    # get allocation
    allocation = network(testset).detach()

    # obj= -(output * trainset_value).sum(1)
    sw = allocation *  testset
    swA = sw[:, 0:parameters.nA].sum(1)
    swB = sw[:, parameters.nA:].sum(1)
    print(allocation)
    # 计算积分 即 卖家的utility
    integralz = Calutility(testset_v,parameters.test,1)
    utility = sw.sum(1) - integralz
    print("***")
    print(torch.mean(integralz))
    print(torch.mean(sw.sum(1)))
    obj_test = utility
    # cons = torch.abs(swA - swB) - parameters.epsilonfor_Uitility
    # loss =lag
    # get obj(socialwelfare)
    # print("(Testset) Total (expected) obj (social welfare): ", obj_test)
    expected_different_utility=torch.abs(swA-swB)

    # print("Group A utility: ", integralzA)
    # print("Group B utility: ", integralzB)
    print("(Testset) Total (expected) different utility: ",  expected_different_utility)

    return swA.numpy(), swB.numpy(), expected_different_utility.numpy(), obj_test.numpy(), allocation

def accuracy_exp(network, testset, save_path,testset_v):
    # get allocation
    allocation = network(testset).detach()

    # obj= -(output * trainset_value).sum(1)
    sw = allocation *  testset
    swA = sw[:, 0:parameters.nA].sum(1)
    swB = sw[:, parameters.nA:].sum(1)
    print(allocation)
    # 计算积分 即 卖家的utility
    integralz = Calutility(testset_v,parameters.test,2)
    utility = sw.sum(1) - integralz
    print("***")
    print(torch.mean(integralz))
    print(torch.mean(sw.sum(1)))
    obj_test = utility
    # cons = torch.abs(swA - swB) - parameters.epsilonfor_Uitility
    # loss =lag
    # get obj(socialwelfare)
    # print("(Testset) Total (expected) obj (social welfare): ", obj_test)
    expected_different_utility=torch.abs(swA-swB)

    # print("Group A utility: ", integralzA)
    # print("Group B utility: ", integralzB)
    print("(Testset) Total (expected) different utility: ",  expected_different_utility)

    return swA.numpy(), swB.numpy(), expected_different_utility.numpy(), obj_test.numpy(), allocation
def accuracy_n2(network, testset, save_path,testset_v):
    # get allocation
    allocation = network(testset).detach()

    # obj= -(output * trainset_value).sum(1)
    sw = allocation *  testset
    swA = sw[:, 0:parameters.nA].sum(1)
    swB = sw[:, parameters.nA:].sum(1)
    print(allocation)
    # 计算积分 即 卖家的utility
    integralz = Calutility(testset_v,parameters.test,3)
    utility = sw.sum(1) - integralz
    print("***")
    print(torch.mean(integralz))
    print(torch.mean(sw.sum(1)))
    obj_test = utility
    # cons = torch.abs(swA - swB) - parameters.epsilonfor_Uitility
    # loss =lag
    # get obj(socialwelfare)
    # print("(Testset) Total (expected) obj (social welfare): ", obj_test)
    expected_different_utility=torch.abs(swA-swB)

    # print("Group A utility: ", integralzA)
    # print("Group B utility: ", integralzB)
    print("(Testset) Total (expected) different utility: ",  expected_different_utility)

    return swA.numpy(), swB.numpy(), expected_different_utility.numpy(), obj_test.numpy(), allocation
def equalFairness(data):
    result = np.zeros(parameters.test)
    swA = np.zeros(parameters.test)
    swB = np.zeros(parameters.test)
    obj = np.zeros(parameters.test)
    for i in range(parameters.test):
       WA = data[i][parameters.nA - 1]
       WB = data[i][parameters.n - 1]
       pA= data[i][parameters.nA - 2]
       pB= data[i][parameters.n - 2]
       PrA= WB/(WA+WB)
       PrB= WA/(WA+WB)
       swA[i]=WA*PrA
       swB[i]=WB*PrB
       obj[i]=pA*PrA+pB*PrB
       result[i]=np.abs(swA[i]-swB[i])
    return swA, swB, result, obj


def get_Second_Auction_answer(data):
  result = np.zeros(parameters.test)
  uA = np.zeros(parameters.test)
  uB = np.zeros(parameters.test)
  obj = np.zeros(parameters.test)
  for i in range(parameters.test):
        if (data[i][parameters.nA-1]>data[i][parameters.n-1]):
          result[i]=data[i][parameters.nA-1]
          uA[i]=data[i][parameters.nA-1]
          uB[i]=0
          obj[i]= data[i][parameters.nA-2]
        else:
          result[i] = data[i][parameters.n-1]
          uA[i] = 0
          uB[i] = data[i][parameters.n-1]
          obj[i] = data[i][parameters.n-2]
  return uA, uB, result, obj

# def basic_model(data,basicnet):
#
#     uA=(data[0]-data[1])*PrA
#     uB = (data[parameters.nA] - data[parameters.nA+1]) * PrB
#     result = uA-uB
#     if result<0:
#         result=-result
#     obj=data[0]*PrA+data[parameters.nA]*PrB
#     return uA, uB, result, obj
def answer_query(query):
    # testset = torch.sort(torch.rand(parameters.n))[0]#生成测试数据
    # trainseta = torch.sort((torch.rand(parameters.test,parameters.nA)) * 10)[0]
    # trainsetb = torch.sort((torch.rand(parameters.test,parameters.nB)) * 8)[0]
    # # trainseta = torch.sort(torch.normal(5.0, 1, (parameters.test, parameters.nA)))[0]
    # # trainsetb = torch.sort(torch.normal(4.0, 1, (parameters.test, parameters.nB)))[0]
    # # 把a和b拼接起来
    # testset = torch.tensor(np.concatenate((trainseta, trainsetb), axis=1))

    testset = pd.read_csv('C:/Users/Administrator/Desktop/testset5(1).csv')
    test_data = np.array(testset)
    testset = torch.tensor(test_data).to(torch.float32)

    test_data = testset.numpy()  # 获取测试数据
    # print(test_data)
    # print(testset.shape)

    # save_path = "C:/Users/Administrator/Desktop/Monotonic/" + "n=" + str(parameters.n) + "m=" + str(
    #     parameters.m) + "basictrain=" \
    #             + str(parameters.train) + "t=" + str(parameters.t) + "b=" + str(parameters.beta) + 'lamb=' \
    #             + str(parameters.lamb) + "lamb_rate=" + str(parameters.lamb_rate) + "model_rate=" \
    #             + str(parameters.model_rate) + "embed=" + str(parameters.embed) + "epsilonfor_Uitility=" + str(
    #     parameters.epsilonfor_Uitility) + "/best_bf_model"
    # parameters.PrA = torch.load(save_path, map_location=torch.device('cpu'))

    # record parameters
    save_path = parameters.save_path + "data=" + str(parameters.sum_dataset_path) + "," + str(query) + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(save_path+"File exists.")
    f_para = open(save_path + "/parameters.txt", "w")
    f_para.write("n = " + str(parameters.n) + "\n")
    f_para.write("nA="+str(parameters.nA) + "\n")
    f_para.write("nB=" + str(parameters.nB) + "\n")
    f_para.write("t = " + str(parameters.t) + "\n")
    f_para.write("m = " + str(parameters.m) + "\n")
    f_para.write("beta = " + str(parameters.beta) + "\n")
    f_para.write("lambda = " + str(parameters.lamb) + "\n")
    f_para.write("lambda rate = " + str(parameters.lamb_rate) + "\n")
    f_para.write("mu rate = " + str(parameters.mu_rate) + "\n")
    f_para.write("lambda for basic = " + str(parameters.lambforbaisc) + "\n")
    f_para.write("model rate = " + str(parameters.model_rate) + "\n")
    f_para.write("proportion of 0 = " + str(parameters.proportion_of_0) + "\n")
    f_para.write("space = " + str(parameters.integral_space) + "\n")
    f_para.write("d = " + str(parameters.d_sum) + "\n")
    f_para.write("test = " + str(parameters.test) + "\n")
    f_para.write("train = " + str(parameters.train) + "\n")
    f_para.write("query = " + str(query) + "\n")
    f_para.write("dataset = " + str(parameters.sum_dataset_path) + "\n")
    f_para.write("epsilon for basic = " + str(parameters.epsilon) + "\n")
    f_para.write("epsilon for sigma = " + str(parameters.epsilonfor_Uitility) + "\n")
    f_para.write("PRA= " + str(parameters.PrA) + "\n")
    f_para.close()
    # # load private data
    # private_data = pd.read_csv(parameters.sum_dataset_path, header=None)
    # private_data = private_data.iloc[:parameters.n].to_numpy()  # only read n rows
    # private_data = private_data.flatten()
    # np.random.shuffle(private_data)

    pd.DataFrame(test_data).to_csv(save_path + '/testset.csv')


    # model = 'best_model'
    # model2 = 'best_bf_model'
    # model3 = 'bf_model'
    # network = torch.load(parameters.save_path+parameters.model, map_location=torch.device('cpu'))
    network2 = torch.load(parameters.save_path+parameters.model2, map_location=torch.device('cpu'))
    # network3 = torch.load(parameters.save_path+parameters.model3, map_location=torch.device('cpu'))
    network_log = torch.load(parameters.save_path_log + parameters.model2, map_location=torch.device('cpu'))
    network_exp = torch.load(parameters.save_path_exp + parameters.model2, map_location=torch.device('cpu'))
    network_n2 = torch.load(parameters.save_path_n2 + parameters.model2, map_location=torch.device('cpu'))
    # network(testset).detach()
    # network_log(testset).detach()


    # basicnet= torch.load(parameters.basicmodel, map_location=torch.device('cpu'))
    # test
    # get true answer TRUE UTILITY DIFFERENT IN SECTION AUCTION
    A_utilityT, B_utilityT, different_utilityT, objT \
        = get_Second_Auction_answer(test_data)
    print("true answer: ",  different_utilityT)
    print("")
    f_Second_Auction_different_utility = pd.DataFrame(different_utilityT, columns=["Second_Auction_different_utility"])
    obj_Second_Auction = pd.DataFrame(objT, columns=["obj_Second_Auction "])
    Second_Auction_B_utility = pd.DataFrame(B_utilityT, columns=["Second_Auction_B_utility"])
    Second_Auction_A_utility = pd.DataFrame(A_utilityT, columns=["Second_Auction_A_utility"])


    print("network model2: ")
    A_utility2, B_utility2, different_utility2, obj2, allocation2 \
        = accuracy(network2, testset,save_path)
    pd.DataFrame(allocation2).to_csv(save_path + '/allocation2.csv')
    # np.savez(save_path + "/network_data2", n_privacy=n_privacy2, n_payments=n_payments2,
    #          n_num_of_selected=n_num_of_selected2, n_answer_list=n_answer_list2, true_answer=Second_Auction_answer)
    f_Network_different_utility2 = pd.DataFrame(different_utility2, columns=["Network2_different_utility2"])
    obj_Network_obj2 = pd.DataFrame(obj2, columns=["Network2_obj2"])
    Network_B_utility2 = pd.DataFrame(B_utility2, columns=["Network2_B_utility2"])
    Network_A_utility2 = pd.DataFrame(A_utility2, columns=["Network2_A_utility2"])
    # df_net2 = pd.DataFrame(n_stats2, columns=["Network2"])
    # an_net2 = pd.DataFrame(n_answer_list2, columns=["Network2"])
    print("")


    print("network model_log: ")
    A_utility_log, B_utility_log, different_utility_log, obj_log, allocation_log \
        = accuracy_log(network_log, testset, save_path,testset)
    pd.DataFrame(allocation_log).to_csv(save_path + '/allocation_log.csv')
    f_Network_different_utility_log = pd.DataFrame(different_utility_log, columns=["Network_log_different_utility"])
    obj_Network_obj_log = pd.DataFrame(obj_log, columns=["Network_log_obj"])
    Network_B_utility_log = pd.DataFrame(B_utility_log, columns=["Network_log_B_utility"])
    Network_A_utility_log = pd.DataFrame(A_utility_log, columns=["Network_log_A_utility"])
    print("")
    print("network model_exp: ")
    A_utility_exp, B_utility_exp, different_utility_exp, obj_exp, allocation_exp \
        = accuracy_exp(network_exp, testset, save_path,testset)
    pd.DataFrame(allocation_exp).to_csv(save_path + '/allocation_exp.csv')
    f_Network_different_utility_exp = pd.DataFrame(different_utility_log, columns=["Network_exp_different_utility"])
    obj_Network_exp = pd.DataFrame(obj_exp, columns=["Network_exp_obj"])
    Network_B_utility_exp = pd.DataFrame(B_utility_exp, columns=["Network_exp_B_utility"])
    Network_A_utility_exp = pd.DataFrame(A_utility_exp, columns=["Network_exp_A_utility"])
    print("")

    print("network model_n2: ")
    A_utility_n2, B_utility_n2, different_utility_n2, obj_n2, allocation_n2 \
        = accuracy_n2(network_n2, testset, save_path, testset)
    pd.DataFrame(allocation_log).to_csv(save_path + '/allocation_n2.csv')
    f_Network_different_utility_n2 = pd.DataFrame(different_utility_log, columns=["Network_n2_different_utility"])
    obj_Network_obj_n2 = pd.DataFrame(obj_n2, columns=["Network_n2_obj"])
    Network_B_utility_n2 = pd.DataFrame(B_utility_n2, columns=["Network_n2_B_utility"])
    Network_A_utility_n2 = pd.DataFrame(A_utility_n2, columns=["Network_n2_A_utility"])
    print("")

    print("equalFairness")
    A_ef, B_ef, different_ef, obj_ef \
        = equalFairness(testset)
    # pd.DataFrame( allocation_ef).to_csv(save_path + '/ allocation_ef.csv')
    different_sw_ef = pd.DataFrame(different_ef, columns=["different_ef"])
    obj_S_ef = pd.DataFrame(obj_ef, columns=["obj_S_ef"])
    B_ef_sw = pd.DataFrame(B_ef, columns=["B_ef_sw"])
    A_ef_sw = pd.DataFrame(A_ef, columns=["A_ef_Sw"])



    # get and store results
    fair = pd.concat([f_Second_Auction_different_utility, f_Network_different_utility2,
                      f_Network_different_utility_log,f_Network_different_utility_exp,f_Network_different_utility_n2,different_sw_ef], axis=1)
    fair.to_csv(save_path + '/fair.csv')
    UA = pd.concat([ Second_Auction_A_utility, Network_A_utility2, Network_A_utility_log, Network_A_utility_exp, Network_A_utility_n2,A_ef_sw],axis=1)
    UA.to_csv(save_path + '/UA.csv')
    UB = pd.concat([ Second_Auction_B_utility, Network_B_utility2, Network_B_utility_log,Network_B_utility_exp,Network_B_utility_n2, B_ef_sw], axis=1)
    UB.to_csv(save_path + '/UB.csv')
    OBJ = pd.concat([obj_Second_Auction, obj_Network_obj2,obj_Network_obj_log,obj_Network_exp,obj_Network_obj_n2,obj_S_ef], axis=1)
    OBJ.to_csv(save_path + '/OBJ.csv')



def __main__():
    answer_query("sum")

if __name__ == "__main__":
    __main__()
