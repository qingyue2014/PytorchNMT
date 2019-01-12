from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 创建lookup表 input_size: 词典大小 hidden_size: 词向量维度
        self.embedding = nn.Embedding(input_size, embedding_size)
        # 此处设置词向量维度和rnn隐藏节点数一致
        self.gru = nn.GRU(embedding_size, hidden_size,bidirectional=True)

    def forward(self, input, hidden):#每次的input是一个时间步一个词
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded #embedded:[1,1,256]
        outputs, hidden = self.gru(output, hidden)
        return outputs, hidden # output:[1,1,256] hidden:[1,1,256]

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device) #创建一个维度为1*1*hidden_size的全0向量

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #embedding: look_up 表 [output_size,hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size)
        #Linear: y=xA+b A: [hidden_size,output_size]
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size #输出维度等于目标语言词典大小
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #[256*2,10]
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_w = nn.Linear(self.hidden_size*3, self.hidden_size)
        self.attn_v= nn.Linear(self.hidden_size, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        # embedded:[1,1,256] 输入词的embedding
        embedded = self.dropout(embedded)
        #torch.cat:上一时刻的hidden:[1,256] 和当前时刻的embedding:[1,256] 连接成为[1,512]
        #attn: 输入[1,512] A:[512,10] 输出：[1,10]
        #1.hidden[0]: [1,256]
        #2.hidden.expend(10,-1): [10,256] encoder_outputs:[10,256]
        #3.cat: [10,512]
        h1 = hidden[0].expand(encoder_outputs.shape[0], -1)
        att_cat = torch.cat((h1, encoder_outputs), 1)
        #self.attn_w(att_cat): [50, 256]
        #self.attn_v(self.attn_w(att_cat)): [50,1]
        #attn_weights:[1,50]
        attn_weights = F.softmax(
            self.attn_v(F.relu(self.attn_w(att_cat))).t(), dim=1)
        #encoder_outputs: [10,256]
        #[1,1,50] * [1,50,512] attn_applied:[1,1,512]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        #[1,256] [1,256] output:[1,512]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #output:[1,1,256]
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        #output:[1,1,256]
        output, hidden = self.gru(output, hidden)
        #output:[1,target_volum] log_softmax等价于log(softmax(x))
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

