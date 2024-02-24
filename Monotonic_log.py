from parameters import parameters
import torch
import torch.nn as nn

class Monotonic_log(nn.Module):
    def __init__(self):
        super().__init__()

        self.hyper_w_1A = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))
        self.hyper_w_finalA = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))

        # State dependent bias for hidden layer
        self.hyper_b_1A = nn.Linear(parameters.n, 1)

        # V(s) instead of a bias for the last layers
        self.VA = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                               nn.ReLU(),
                               nn.Linear(parameters.embed, 1))


        self.hyper_w_1B = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))
        self.hyper_w_finalB = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))

        # State dependent bias for hidden layer
        self.hyper_b_1B = nn.Linear(parameters.n, 1)

        # V(s) instead of a bias for the last layers
        self.VB = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                               nn.ReLU(),
                               nn.Linear(parameters.embed, 1))
    def forward(self, theta_value):
        # 把theta 分成两部分计算 thetaA thetaB
        # First layer
        print("exp")
        theta=torch.log(theta_value)

        #theta = theta_value
        w1A = torch.abs(self.hyper_w_1A(theta))
        b1A =torch.abs(self.hyper_b_1A(theta))
        hiddenA = w1A * theta + b1A
        # hiddenA = w1A * thetaA + b1A

        w1B = torch.abs(self.hyper_w_1B(theta))
        b1B  = torch.abs(self.hyper_b_1B(theta))
        hiddenB = w1B * theta + b1B
        # hiddenB = w1B * thetaB + b1B

        # Second layer
        w_finalA = torch.abs(self.hyper_w_finalA(theta))

        w_finalB = torch.abs(self.hyper_w_finalB(theta))
        # State-dependent bias
        vA = torch.abs(self.VB(theta))
        vB =  torch.abs(self.VB(theta))

        # Compute final output
        yA = hiddenA * w_finalA + vA
        yB = hiddenB * w_finalB + vB
        yA_=yA[:,0:parameters.nA]
        yB_=yB[:,parameters.nA:]

        # 拼接y
        y= torch.cat((yA_, yB_),1)

        column_sum = y.sum(dim=1)
        y = y /  column_sum.unsqueeze(1) # 用torch.tensor的方法
        parameters.W1A=w1A
        parameters.W2A=w_finalA
        parameters.W1B=w1B
        parameters.b1A=b1A
        parameters.b2A=vA
        parameters.b1B=b1B
        parameters.b2B=vA
        parameters.W2B=w_finalB
        if ((any(w1A) < 0) | (any(b1A) < 0) | (any(w1B) < 0) | (any(b1B) < 0)):
            print("不符合要求")
        if ((any(w_finalB) < 0) | (any(vA) < 0) | (any(vB) < 0) | (any(w_finalA) < 0)):
            print("不符合要求")
        # 切分yAyB
        return y