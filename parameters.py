
class parameters:
    nA = 500 # number of group A
    nB = 500  # number of group B
    n = nA + nB
    t = 1000  # number of epchos
    m = 100 # number of queries
    embed = 8  # hidden layer
    beta = round(0.7 * n)
    lamb = 0.1
    lamb_rate = 0.005
    lambforbaisc=0.005
    mu_rate=0.0001
    model_rate = 0.010  # trainer learning rate  可以放大
    proportion_of_0 = 0.2
    test =100
    epsilon=1
    epsilonfor_Uitility = 1
    d_sum = 2  # the cardinality of the private data 15
    d_lp = 2
    d_median = 40
    median_low = 0   #
    median_high = 13
    PrA=0.35
    train = 1  # the number of rounds of training in one iteration

    model = "/best_model"
    model2 = "/best_bf_model"
    model3 = "/bf_model"

    integral_space = 10
    sum_dataset_path = "Customers_count.csv"
    median_dataset_path = "Customers_median.csv"

    # save_path = "C:/Users/Administrator/Desktop/Monotonic/" + "n=" + str(n) + "m=" + str(m) + "train=" + str(train) \
    #             + "t=" + str(t) + "b=" + str(beta) + 'lamb=' + str(lamb) + "lamb_rate=" + str(lamb_rate) \
    #             + "model_rate=" + str(model_rate) + "emded=" + str(embed)+"epsilonfor_Uitility=" + str(epsilonfor_Uitility)
    save_path = "C:/Users/Administrator/Desktop/Monotonic/" + "n=" + str(n) + "m=" + str(m) + "train=" \
                + str(train) + "t=" + str(t) + "b=" + str(beta) + 'lamb=' \
                + str(lamb) + "lamb_rate=" + str(lamb_rate) + "model_rate=" \
                + str(model_rate) + "embed=" + str(embed) + "epsilonfor_Uitility=" + str(epsilonfor_Uitility)
    save_path_log = "C:/Users/Administrator/Desktop/Monotonic_log/" + "n=" + str(n) + "m=" + str(m) + "train=" \
                + str(train) + "t=" + str(t) + "b=" + str(beta) + 'lamb=' \
                + str(lamb) + "lamb_rate=" + str(lamb_rate) + "model_rate=" \
                + str(model_rate) + "embed=" + str(embed) + "epsilonfor_Uitility=" + str(epsilonfor_Uitility)
    save_path_exp = "C:/Users/Administrator/Desktop/Monotonic_exp/" + "n=" + str(n) + "m=" + str(m) + "train=" \
                    + str(train) + "t=" + str(t) + "b=" + str(beta) + 'lamb=' \
                    + str(lamb) + "lamb_rate=" + str(lamb_rate) + "model_rate=" \
                    + str(model_rate) + "embed=" + str(embed) + "epsilonfor_Uitility=" + str(epsilonfor_Uitility)
    save_path_m2 = "C:/Users/Administrator/Desktop/Monotonic_n2/" + "n=" + str(n) + "m=" + str(m) + "train=" \
                    + str(train) + "t=" + str(t) + "b=" + str(beta) + 'lamb=' \
                    + str(lamb) + "lamb_rate=" + str(lamb_rate) + "model_rate=" \
                    + str(model_rate) + "embed=" + str(embed) + "epsilonfor_Uitility=" + str(epsilonfor_Uitility)


    W1A =0
    W2A =0
    b1A =0
    b2A =0

    W1B =0
    W2B =0
    b1B =0
    b2B =0