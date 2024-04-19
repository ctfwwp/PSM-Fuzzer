import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import time
import torch.utils.data as Data
import torch.nn.functional as F
import random
import modbus_tk
import modbus_tk.defines as cst
import modbus_tk.modbus_tcp as modbus_tcp
from numpy import *
import TRM_text.WRD as similar
import TRM_text.check_tcp as ch
import TRM_text.adjust_lv as adjust
LOGGER = modbus_tk.utils.create_logger("console")
num_data = 0
mal_num  = 0
num0,num1,num2,num3,num4,num5 =0,0,0,0,0,0
def  build_key():
    key = {}
    str1 = ['a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num = {'SS': 256, 'EE': 257, 'PP': 258,"":259}
    n = 0
    for x in str1:
        for y in str1:
           keys = x+y
           key[keys] = n
           n= n + 1
    key.update(num)
    return key
src_vocab = build_key()
src_vocab_size = len(src_vocab)
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
tgt_vocab = build_key()
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for i, w in enumerate(tgt_vocab)}

device = 'cpu'
def make_batch(str1):
    input_batch = [[src_vocab[n] for n in str1.split()]]
    return input_batch
def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # print(sentences)
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[src_vocab[n] for n in sentences[i][1].split()]]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[src_vocab[n] for n in sentences[i][2].split()]]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

## 10
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):

        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ##输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        ## 输入进行的attn_mask形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)


        ##然后我们计算 ScaledDotProductAttention 这个函数，去7.看一下
        ## 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

## 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

## 3. PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

## 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

## 2. Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(d_model) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, enc_inputs):
        ## 这里我们的 enc_inputs 形状是： [batch_size x source_len]

        ## 下面这个代码通过src_emb，进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)

        ## 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现；3.
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            ## 去看EncoderLayer 层函数 5.
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

## 10.
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

## 9. Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]

        ## get_attn_pad_mask 自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        ## 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)


        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
## 1. 从整体网路结构来看，分为三个部分：编码层，解码层，输出层
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  ## 编码层
        self.decoder = Decoder()  ## 解码层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False) ## 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
    def forward(self, enc_inputs, dec_inputs):
        ## 这里有两个数据进行输入，一个是enc_inputs 形状为[batch_size, src_len]，主要是作为编码段的输入，一个dec_inputs，形状为[batch_size, tgt_len]，主要是作为解码端的输入

        ## enc_inputs作为输入 形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        ## enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        ## dec_outputs 是decoder主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        ## dec_outputs做映射到词表大小
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
def tenToTwo(num, sz):
    if sz == 0:
        return ''
    if num == 0:
        return '0' * sz
    return tenToTwo(num // 2, sz - 1) + str(num % 2)
def hextoten(data):   #在这里两个字符为1byte,1byte=8bit
    sz = len(data)
    res = []
    for i in range(0, sz, 2):
        #16 to 10
        asc_hight_num = ord(data[i]) - ord('0')
        if 97 <= ord(data[i]) <= 102:
            asc_hight_num = asc_hight_num + ord('0') - ord('a') + 10

        asc_low_num = ord(data[i + 1]) - ord('0')
        if 97 <= ord(data[i + 1]) <= 102:
            asc_low_num = asc_low_num + ord('0') - ord('a') + 10

        # 10 to 2
        h = tenToTwo(asc_hight_num, 4)
        l = tenToTwo(asc_low_num, 4)
        res += [h+l]
    return res
def mutation_two(data):  #data字段变异
    # print("二进制数据：", data)
    res = hextoten(data)
    bit = ""
    for str in res:
        bit = bit + str
    return bit
def bytes_to_hex(a):
    if len(a) == 1:
        return "0" + a
    else:
        return a
def towtohex(data):
    str = ""
    for i in range(0,len(data),8):
        str = str + bytes_to_hex(hex(int(data[i:i + 8], 2))[2:])
    return str
#二进制转换
def binary_change(data):
    data =list(data)
    for i in range(0,len(data)):
        # f = random.uniform(0,1)
        # if f <0.5:
            if data[i] == '0':
                data[i] = "1"
            else:
                data[i] = "0"
    return towtohex("".join(data))

def softmax1(data):
    data = list(data)
    max_tensor = torch.tensor(data)
    _, index1 = max_tensor.topk(10)
    index2 = sorted(list(index1.squeeze().numpy()))
    index2.append(-1)
    denominator = 0
    j = 1
    l = len(data)
    for i in data:
        denominator = denominator + math.exp(i / j)
    for i in range(0, l):
        data[i] = math.exp(data[i] / j) / denominator
    return data

def range_bit_decoder(model,enc_input,str0,lv,start_symbol):
    """变异采样编码
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    prob1 = [ ]
    d_lv = []
    if int(str0[-2:],16)==15:   #  10= 15 0f =16
         n = 7
    elif int(str0[-2:],16)==16:
         n = 8
    else:
         n=int(str0[-2:],16)
    d_lv.append(n)
    location = 0
    fuzz = random.uniform(0, 1)  #####注意这个fuzz的位置
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        soft_max=softmax1(projected.squeeze(0).data.numpy()[len(projected.squeeze(0).data)-1])
        max_data=max(soft_max)
        prob = soft_max.index(max_data)
        ture_output = soft_max.index(max(soft_max))
        # fuzz = random.uniform(0, 1)     #####注意这个fuzz的位置vx
        if torch.tensor(prob) == tgt_vocab["PP"]:
            prob1.append(tgt_vocab["PP"])
        elif torch.tensor(prob) == tgt_vocab["EE"]:
            prob1.append(tgt_vocab["EE"])
        else:
            src_word=src_idx2word[ture_output]
            sil = similar.similarity(str0,src_word)  #计算两个字符串的相似度
            sil = 0.5 + 0.5*sil                     #通过放缩法将相似度集中在0-1之间
            if sil*lv[n-1][location]>=fuzz:
                res = mutation_two(src_word)
                change_data = binary_change(res)
                prob1.append(src_vocab[change_data])
                str0 = str0 + change_data
                d_lv.append(location)
            else:
                str0 = str0 + src_word
                prob1.append(ture_output)

        location = location + 1
        next_symbol = torch.tensor(prob)
        if next_symbol == tgt_vocab["EE"]:
            terminal = True
    # greedy_dec_predict = dec_input[:, 1:]
    return torch.tensor(prob1),d_lv
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
def data_change(str1):
    data=""
    for i in range(0, len(str1), 2):
        data=data+str1[i] + str1[i + 1]+" "
    return data.strip()
def add_data(str1):
    data=""
    for i in range(0, len(str1), 2):
        data = data + str1[i] + str1[i + 1] + " "
    for i in range(0,int((max_len-len(str1))/2)):
        data = data+"PP "
    return data.strip()
def data_variation(str2):
    return str2
if __name__ == '__main__':
    ftxt=open("modbus_tcp.txt","r")
    ftxt1 = open("modbus_tcp.txt", "r")
    sentences = []
    lv = []
    fc_number = [0]*8
    fd = ftxt.readlines()
    max_len=len(fd[fd.index(max(fd, key=len))])-18
    for i in range(0,8):  #初始化lv
        lv.append([1]*max_len)
    random.shuffle(fd)
    for str1 in fd:
         input = []
         data = str1.split()
         input.append(data_change(data[0]))
         input.append("SS "+add_data(data[1]))
         input.append(add_data(data[1])+" EE")
         sentences.append(input)
    # print(sentences)
    ftxt.close()

    ## 句子的输入部分，
    #
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    #loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs),2, True)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs),32, True)
    ## 模型参数
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention
    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
#############################################################
    # 如果你要重新训练模型，就不要注释下边for循环的代码。
    # 但是如果你训练完了，模型是可以自动保存在本地的，这样你可以注释下边的，就可以节省训练的时间
#############################################################
    for epoch in range(9):
        s_time = time.time()
        l=[]
        for enc_inputs, dec_inputs, dec_outputs in loader:
            optimizer.zero_grad()
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs,dec_outputs.view(-1))
            l.append(loss.data.item())
            loss.backward()
            optimizer.step()
        e_time = time.time()
        epoch_mins, epoch_secs = epoch_time(s_time, e_time)
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(min(l)), f'Time: {epoch_mins}m {epoch_secs}s')
        torch.save(model, 'model_4096_'+str(9)+'_1.pth')
##################################################################################
    str_txt = open("error ModRSsim2.txt",'a')     #错误的数据（被正确接收）
    str_txt.write(time.strftime("%Y-%m-%d:%H:%M")+":\n")
    number = 0
    fd1 = ftxt1.readlines()
    b_time = time.time()
    try:
        # 连接从机地址,这里要注意端口号和IP与从机一致
        MASTER = modbus_tcp.TcpMaster(host="192.168.1.102", port=502)
        MASTER.set_timeout(0.5)
        LOGGER.info("connected")
        model = torch.load('model_4096_7_1.pth')

        record_all_text = []
        record_right_text = []
        print("开始生成：模型为 model_4096_7_1.pth")
        discount_lv = []
        unusual_data = 0
        ftxt_recode = open("ModRSsim_result_7.1.txt", 'a')
        while number <=16100:
            # time.sleep(0.5)
            uid = random.randint(0, 3999)
            length = random.randint(0, 3999)
            slave_fc = random.randint(0, 3999)
            str_uid = fd1[uid][0:4]
            str_len = fd1[length][8:12]
            str_slave = fd1[slave_fc][12:16]
            str0 = str_uid+"0000"+ str_len +str_slave
            enc_inputs= make_batch(data_change(str0))
            number = number + 1

            b,discount_lv= range_bit_decoder(model,torch.tensor(enc_inputs).view(1, -1).to(device),str0,lv,start_symbol=tgt_vocab["SS"])
            input_txt = " ".join('%s' % idx2word[id] for id in torch.tensor(enc_inputs).numpy()[0]).replace(" ", "")
            data1 = ''.join([idx2word[n.item()] for n in b.squeeze()]).replace("PP", "").replace("SS", "")
            ##########
            modbus_tcp1 = input_txt + "" + data1.replace("EE","")
            lenth = int(len(modbus_tcp1[12:]) / 2)
            length = hex(lenth).split("0x")
            length1 = length[1]
            for i in range(0, 4 - len(length[1])):
                length1 = "0" + length1
            string = list(modbus_tcp1)
            string[8:12] = length1
            modbus_tcp2 = "".join(string)

            flag,reasion= ch.check(modbus_tcp2)
            num_data = num_data + 1

            num = MASTER.execute1(modbus_tcp2[12:14], modbus_tcp2)
#####################################

            #num = MASTER.execute(modbus_tcp2[12:14], modbus_tcp2)
            #源代码这是个，但是我改成上边的，我在这个库里 多加了一个函数execute1()，这个函数放在execute下边。这个函数的代码，在 “代码.txt” 文件中。
            #你安装库之后，进去库里把 “代码.txt”的函数放到，相同的位置就行。

#####################################
            if flag==False and num ==0:
                num0 = num0 + 1
                state=1
                unusual_data = unusual_data + 1
            elif num==0:
                num0 = num0 + 1
                state=0
            else:
                mal_num = mal_num + 1
                state=-1

            lv, fc_number = adjust.adjvst_lv(lv, discount_lv, state, fc_number)
            target_numbers = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
            o_time = time.time()
            if num_data in target_numbers:
                epoch_mins, epoch_secs = epoch_time(b_time, o_time)
                print("###########################第", num_data, "测试用例################################")
                print("测试", num_data, "条数据", f'共用时: {epoch_mins}m {epoch_secs}s')
                print("      接收率：", format(num0 / num_data, '.4%'))
                print("      错误率：", format(1 - (num0 / num_data), '.4%'))
                print("非正常正确数据：", unusual_data, format((unusual_data / num_data), '.4%'))

                ftxt_recode.write("共测试"+ str(num_data)+"条数据,"+str(f'共用时: {epoch_mins}m {epoch_secs}s')+"\n")
                ftxt_recode.write("      接收率："+ str(num0)+"  "+ str(format(num0 / num_data, '.4%'))+"\n")
                ftxt_recode.write("      错误率："+ str(format(1 - (num0 / num_data), '.4%'))+"\n")
                ftxt_recode.write("非正常正确数据："+ str(unusual_data)+"  " +str(format((unusual_data / num_data), '.4%')) + "\n")

        ftxt_recode.close()

    except modbus_tk.modbus.ModbusError as err:
        LOGGER.error("%s- Code=%d" % (err, err.get_exception_code()))


