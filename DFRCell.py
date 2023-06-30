import torch
import numpy as np
import torch.nn as nn


class OriDFRCell(nn.Module):
    def __init__(self, n_input, n_hidden=10):
        super(OriDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_input, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        self.n_hidden = n_hidden
        #nn.init.xavier_uniform_(self.mask)
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        output = self.act3(vec_x + 0.25 * prev_output)
        return output, output 

class OriQDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(OriQDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        self.n_hidden = n_hidden
        self.mask_q = torch.quantization.FakeQuantize(observer=torch.quantization.observer.MovingAverageMinMaxObserver, 
                                                      quant_min=0, quant_max=255)
        self.output_q = torch.quantization.FakeQuantize(observer=torch.quantization.observer.MovingAverageMinMaxObserver, 
                                                        quant_min=0, quant_max=255)
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x.unsqueeze(-1), self.mask_q(self.mask))
        output = self.act3(vec_x + 0.125 * prev_output)
        output = self.output_q(output)
        return output, output 
    
class NewDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(NewDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        self.n_hidden = n_hidden
        #nn.init.xavier_uniform_(self.mask)
                
    def forward(self, x, prev_output):
        x = x.view(-1, 1)
        #vec_x = torch.matmul(x, self.mask)
        #output = self.act3(vec_x + 0.5 * prev_output)
        #prev_output = output
        output = []
        for i in range(self.n_hidden):
            cur = x * self.mask[0, i]
            cur = cur + 0.8 * prev_output
            cur = self.act3(cur)
            prev_output = cur
            output.append(cur)
        output = torch.cat(output, dim=1)
        return output, prev_output 

class DFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(DFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=0.0, b=1.0)
        #nn.init.xavier_uniform_(self.mask)
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        #vec_x = self.act1(vec_x)
        output = self.act(vec_x + 0.8 * prev_output)
        return output

class ParallelDFRCell(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(ParallelDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_in, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        #nn.init.xavier_uniform_(self.mask)
        #nn.init.xavier_uniform_(self.W)
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        output = self.act1(vec_x + prev_output)
        return output

class QParallelDFRCell(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(QParallelDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.act1_1 = nn.Hardtanh(min_val=0.0, max_val=60.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_in, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        self.register_buffer('layer', torch.zeros(2))
        self.layer[0] = -1 
        self.layer[1] = 1
        self.register_buffer('mask_p', torch.zeros(2))
        self.mask_p[0] = -0.5
        self.mask_p[1] = 0.5
        self.register_buffer('vec_p', torch.zeros(2))
        self.vec_p[0] = -30
        self.vec_p[1] = 30
        self.register_buffer('in_p', torch.zeros(2))
        self.in_p[0] = -60
        self.in_p[1] = 60

                
    def forward(self, x, prev_output):
        #print(torch.max(x), torch.min(x))
        #x = self.bn(x)
        x = Quantize(x, self.in_p[0], self.in_p[1])
        #x = Quantize(x, num_bits=16)
        mask = Quantize(self.mask, self.mask_p[0], self.mask_p[1])
        vec_x = torch.matmul(x, mask)
        output = self.act1(vec_x + prev_output)
        output = Quantize(output, self.layer[0], self.layer[1])
        return output

class ADFRCell(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(ADFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.act4 = nn.Sigmoid()
        self.act2 = nn.Softsign()
        self.act3 = nn.ReLU()
        self.act = nn.Tanh()
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_in, n_hidden), requires_grad=True)
        #nn.init.uniform_(self.mask, a=0.5, b=0.5)
        nn.init.xavier_uniform_(self.mask)
                
    def forward(self, x, prev_output):
        vec_x = torch.matmul(x, self.mask)
        output = self.act(vec_x + 0.5 * prev_output)
        #print(output)
        return output

class NarmaDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(NarmaDFRCell, self).__init__()
        self.act1 = torch.nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.act = torch.nn.Tanh()
        self.act2 = torch.nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-0.5, b=0.5)
        #nn.init.xavier_uniform_(self.mask)
                
    def forward(self, x, prev_output):
        vec_x = self.act1(torch.matmul(x, self.mask))
        output = self.act1(vec_x + 0.8 * prev_output)
        return output

class QDFRCell(nn.Module):
    def __init__(self, n_hidden=10):
        super(QDFRCell, self).__init__()
        self.act = torch.nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(1, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=0.0, b=1.0)
        self.register_buffer('in_min_max', torch.zeros(2))
        self.in_min_max[1] = 0.497 
        self.register_buffer('l1_min_max', torch.zeros(2))
        self.l1_min_max[1] = 1.9 
        self.register_buffer('maskout_min_max', torch.zeros(2))
        self.maskout_min_max[1] = 0.95
                
    def forward(self, x, prev_output):
        #update min and max of input
        x = Quantize(x, self.in_min_max[0], self.in_min_max[1])
        q_mask = Quantize(self.mask)
        vec_x = torch.matmul(x, q_mask)
        q_vec_x = Quantize(vec_x, self.maskout_min_max[0], self.maskout_min_max[1])
        bias = Quantize(0.2 * prev_output, self.l1_min_max[0], self.l1_min_max[1])
        output = self.act(q_vec_x + prev_output - bias)
        output = Quantize(output, self.l1_min_max[0], self.l1_min_max[1])
        return output

class AQDFRCell(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(AQDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        #self.act_QW = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.act_QW = nn.Softsign()
        self.actQ = nn.Tanh()
        self.act = nn.ReLU()
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_in, n_hidden), requires_grad=False)
        nn.init.uniform_(self.mask, a=-1.0, b=1.0)
        #nn.init.xavier_uniform_(self.mask)
        self.register_buffer('In', torch.zeros(2))
        self.register_buffer('mask_param', torch.zeros(2))
        self.register_buffer('layer1', torch.zeros(2))
        self.register_buffer('test', torch.zeros(2))
        self.In[0] = -1.0
        self.In[1] = 1.0
        self.layer1[0] = 0.0
        self.layer1[1] = 2.0
        self.mask_param[0] = -1
        self.mask_param[1] = 1
        self.test[0] = -1.0
        self.test[1] = 1.0


    def forward(self, x, prev_output):
        #q_mask = self.act_QW(self.mask)
        q_x = Quantize(x, self.In[0], self.In[1])
        #q_mask = Quantize(self.mask, self.mask_param[0], self.mask_param[1], num_bits=8)
        q_mask = Quantize(self.mask, self.test[0], self.test[1])
        vec_x = torch.matmul(q_x, q_mask)
        #vec_x = Quantize(vec_x, self.layer1[0], self.layer1[1], num_bits=8)
        #vec_x = self.actQ(vec_x)
        #vec_x_q = Quantize(vec_x, self.In[0], self.In[1])
        #print(prev_output.shape, vec_x.shape)
        output = self.act1(vec_x + 0.5 * prev_output)
        output_q = Quantize(output, self.layer1[0], self.layer1[1])
        return output

class AQRefDFRCell(nn.Module):
    def __init__(self, n_in=1, n_hidden=10):
        super(AQRefDFRCell, self).__init__()
        self.act1 = nn.Hardtanh(min_val=0.0, max_val=2.0)
        self.act_QW = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        self.mask = torch.nn.Parameter(data=torch.Tensor(n_in, n_hidden), requires_grad=True)
        nn.init.xavier_uniform_(self.mask)
        self.register_buffer('In', torch.zeros(2))
        self.register_buffer('layer1', torch.zeros(2))
        self.layer1[1] = 2.0
        self.In[0] = -2.5
        self.In[1] = 2.5
                
    def forward(self, x, prev_output):
        x_q = Quantize(x, self.In[0], self.In[1])
        q_mask = Quantize(self.mask)
        vec_x = torch.matmul(x_q, q_mask)
        vec_x = self.act1(vec_x)
        vec_x_q = Quantize(vec_x, self.layer1[0], self.layer1[1])
        output = self.act1(vec_x_q + prev_output)
        output_q = Quantize(output, self.layer1[0], self.layer1[1])
        return output_q
