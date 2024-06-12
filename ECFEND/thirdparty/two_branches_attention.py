from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Dot(nn.Module):
    """Learn from """
    def __init__(self):
        super().__init__()

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, D)
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------

        """
        assert left.size(0) == right.size(0) and left.size(-1) == right.size(-1), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        left = left.unsqueeze(1)  # (B, 1, D)
        tmp = torch.bmm(left, right.permute(0, 2, 1))  # (B, 1, D) * (B, D, L) => (B, 1, L)
        tmp = tmp.squeeze(1)
        doc_mask = (mask == 0)
        out = tmp.masked_fill(doc_mask, -np.inf)
        attention_weights = F.softmax(out, dim=1)  # (B, L)
        avg = right * attention_weights.unsqueeze(-1) # (B, L, D) * (B, L, 1) => (B, L, D)
        assert len(avg.size()) == 3
        avg = torch.sum(avg, dim = 1)  # dim = 1 compute on middel dimension
        return avg, attention_weights


class BiLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, D)
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0) and left.size(-1) == right.size(-1), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        left = self.W(left)  # (B, D)
        left = left.unsqueeze(1)  # (B, 1, D)
        tmp = torch.bmm(left, right.permute(0, 2, 1))  # (B, 1, D) * (B, D, L) => (B, 1, L)
        tmp = tmp.squeeze(1)
        doc_mask = (mask == 0)
        out = tmp.masked_fill(doc_mask, -np.inf)
        attention_weights = F.softmax(out, dim=1)  # (B, L)
        avg = right * attention_weights.unsqueeze(-1)  # (B, L, D) * (B, L, 1) => (B, L, D)
        avg = torch.sum(avg, dim = 1)  # dim = 1 compute on middel dimension
        return avg, attention_weights


class ConcatSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, D)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class ConcatNotEqualSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights
import math
from d2l import torch as d2l

def transpose_qkv(X, num_heads):

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_mean(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行mean操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return X.mean(-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        # print(valid_lens, valid_lens.shape)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=0)
        # print(X, X.shape)
    xsum = X.reshape(shape).sum(-1)
    # print(xsum, xsum.shape)# 2 3
    valid_lens = valid_lens.reshape(xsum.shape)
    # print(valid_lens, valid_lens.shape)
    return xsum/valid_lens

import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

class ScaleUp(nn.Module):
    """ScaleUp"""

    def __init__(self, scale):
        super(ScaleUp, self).__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale
class DotProductCredibility(nn.Module):
    """缩放点积注意力"""
    def __init__(self, **kwargs):
        super(DotProductCredibility, self).__init__(**kwargs)
        self.scaleup = ScaleUp(scaling_factor(30))
        
    def forward(self, queries, keys, valid_lens=None):
        d = queries.shape[-1]
        # scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        
        # mean = torch.mean(queries, dim=-1)
        # mean = mean.unsqueeze(-1)
        # queries = queries-mean

        # mean = torch.mean(keys, dim=-1)
        # mean = mean.unsqueeze(-1)
        # keys = keys-mean
        
        # queries = F.normalize(queries, p=2, dim=-1)
        # keys = F.normalize(keys, p=2, dim=-1)
        scaleup = self.scaleup
        # scores = scaleup(torch.matmul(queries, keys.transpose(-2, -1)))
        scores = scaleup(torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d))
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)效果不好
        ####################
        self.credibility = masked_mean(scores, valid_lens)
        return self.credibility.squeeze(-1)
        # return scores
        
class DotProductCredibility2(nn.Module):
    """缩放点积注意力"""
    def __init__(self, num_heads, **kwargs):
        super(DotProductCredibility2, self).__init__(**kwargs)
        
        # self.W_c = nn.Linear(30, 1)
        self.W_c = nn.Linear(30, num_heads)
        self.scaleup = ScaleUp(scaling_factor(30))
        
    def forward(self, queries, keys, valid_lens=None):
        d = queries.shape[-1]
        
        # mean = torch.mean(queries, dim=-1)
        # mean = mean.unsqueeze(-1)
        # queries = queries-mean

        # mean = torch.mean(keys, dim=-1)
        # mean = mean.unsqueeze(-1)
        # keys = keys-mean
        
        # queries = F.normalize(queries, p=2, dim=-1)
        # keys = F.normalize(keys, p=2, dim=-1)
        scaleup = self.scaleup
        # scores = scaleup(torch.matmul(queries, keys.transpose(-2, -1)))
        scores = scaleup(torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d))
        # scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d)效果不好
        
        self.credibility = self.W_c(scores)
        return self.credibility
class Attention(nn.Module):
    """多头注意力"""
    def __init__(self, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.attention = DotProductCredibility()
        # self.W_o = nn.Linear(30, 2)
        self.W_o = nn.Linear(30, 30)
        #####################
        # self.qw = nn.Linear(1628, 1628)
        # self.kw = nn.Linear(1628, 1628)
        ###################
        # self.ln = nn.Linear(2, 2)
        # self.ln = nn.Linear(30, 30)


    def forward(self, queries, keys, valid_lens):

        # output = self.attention(torch.tanh(self.qw(queries)), torch.tanh(self.kw(keys)), valid_lens)
        output = self.attention(queries, keys, valid_lens)

        # output = torch.log(torch.sigmoid(self.W_o(output)))+1
        # output = torch.sigmoid(self.ln(torch.tanh(torch.exp(self.W_o(output)))))
        output = torch.tanh(torch.exp(self.W_o(output)))
        return output
class Attention2(nn.Module):
    """多头注意力"""
    def __init__(self, num_heads, bias=False, **kwargs):
        super(Attention2, self).__init__(**kwargs)

        self.attention = DotProductCredibility2(num_heads)
        #####################
        # self.qw = nn.Linear(1628, 1628)
        # self.kw = nn.Linear(1628, 1628)
        ###################
        # self.ln = nn.Linear(2, 2)


    def forward(self, queries, keys, valid_lens):

        # output = self.attention(torch.tanh(self.qw(queries)), torch.tanh(self.kw(keys)), valid_lens)
        output = self.attention(queries, keys, valid_lens)
        
        output = torch.tanh(torch.exp(output))
        # output = torch.sigmoid(self.ln(torch.tanh(torch.exp(output))))
        # output = torch.log(torch.sigmoid(output))+1
        
        return output
class EvdCred(nn.Module):
    def __init__(self, num_layers, num_heads):
        super(EvdCred, self).__init__()
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("EvdCred" + str(i), Attention())
 
        self.final = Attention2(num_heads)
        # self.final = Attention2(1)
        #############
        # self.qln = nn.Linear(1628, 1628)
        # self.qlinears = nn.Sequential()
        # for i in range(num_layers):
        #     self.qlinears.add_module("linear" + str(i), nn.Linear(1628, 1628))
        # self.kln = nn.Linear(1628, 1628)
        # self.klinears = nn.Sequential()
        # for i in range(num_layers):
        #     self.klinears.add_module("linear" + str(i), nn.Linear(1628, 1628))
        # # self.leakyrelu = nn.LeakyReLU(0.1)
        # self.q_prelu = nn.Sequential()
        # for i in range(num_layers):
        #     self.q_prelu.add_module("prelu" + str(i), nn.PReLU())
        # self.qprelu = nn.PReLU()
        # self.k_prelu = nn.Sequential()
        # for i in range(num_layers):
        #     self.k_prelu.add_module("prelu" + str(i), nn.PReLU())
        # self.kprelu = nn.PReLU()

    def forward(self, query, key, valid_lens):
        old_key = key
        for i, blk in enumerate(self.blks):
            # 加上线性变换
            # query = self.q_prelu[i](self.qlinears[i](query))
            # key = self.k_prelu[i](self.klinears[i](key))
            X = blk(query, key, valid_lens)#add value

            # key = X[:, :, 0].unsqueeze(-1).repeat(1, 1, key.shape[-1]) * key + X[:, :, 1].unsqueeze(-1).repeat(1, 1, key.shape[-1]) * key
            # key = X.unsqueeze(-1).repeat(1, 1, key.shape[-1]) * key
            key = X.unsqueeze(-1).repeat(1, 1, key.shape[-1]) * old_key
        #######################
        # query = self.qprelu(self.qln(query))
        # key = self.kprelu(self.kln(key))
        X = self.final(query, key, valid_lens)
        # key = torch.tanh(self.ln(X.repeat(1, 1, key.shape[-1]) * key))
        # return X, key
        return X

def new_cred(mean_cre_tmp, valid_lens):
    #[1, 30, 1], [1]
    evdnum = valid_lens[0]
    sumcred = 0
    for i in range(evdnum):
        sumcred = sumcred + mean_cre_tmp[0][i][0]
    meancred = sumcred / evdnum
    tmp = mean_cre_tmp[0]
    for i in range(30):
        if tmp[i][0] >= meancred:
            mean_cre_tmp[0][i][0] = 1
        else:
            mean_cre_tmp[0][i][0] = 0
    return mean_cre_tmp

class Fuse(nn.Module):
    def __init__(self, dim):
        super(Fuse, self).__init__()
        self.linear = nn.Linear(4 * dim, dim, bias=True).to('cuda:0')

    def forward(self, a, b):
        multi = a * b
        minus = a - b
        input = torch.cat((a, b, multi, minus), dim=-1)
        z = torch.sigmoid(self.linear(input))
        # one = torch.ones_like(a)
        return z * a + (1 - z) * b

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConcatNotEqualSelfAtt2(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1, iter_num: int = 4):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)
        self.evdCred = EvdCred(num_layers=iter_num, num_heads=num_heads).to("cuda:0")

        # self.rightlinear = nn.Linear(5248+num_heads, 5248)
        # self.rightlinear = nn.Linear(1628+num_heads, 1628)
                
        # self.rightlinear = nn.Linear(1028+num_heads, 1028)
        
        # self.gelu = nn.GELU()
        
        
        #aoa
        self.aoa_layer =  nn.Sequential(nn.Linear(num_heads * 2, num_heads * 2), nn.GLU())
        
        
        
        
        # self.rightlinear2 = nn.Linear(1628, 1628)
        
        # self.gelu2 = nn.GELU()
        # self.zlinear = nn.Linear(1630, 1628)
        
        ######################################
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=1628, nhead=4, batch_first=True)
        # self.transencoder = nn.TransformerEncoder(self.encoder_layer, 3)
#         self.mu_head = nn.Linear(num_heads, num_heads)
#         self.logvar_head = nn.Linear(num_heads, num_heads)
# #######################################
#     def _reparameterize(self, mu, logvar):
#         std = torch.exp(logvar).sqrt()
#         epsilon = torch.randn_like(std)
#         return mu + epsilon * std

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()#32, 30, 1628

        valid_lens = torch.tensor([(i==True).sum().long() for i in mask]).to("cuda:0")#32
 
        evd_cred = self.evdCred(right, right, valid_lens)#32,30,2
        #单头可信度输出+改变right（linear+gelu）
        # right = self.gelu(self.rightlinear(torch.cat([right, evd_cred], -1)))
        
        # right = self.gelu(self.rightlinear(evd_cred.repeat(1, 1, right.shape[-1]) * right))
        # z = torch.sigmoid(self.zlinear(torch.cat([right, evd_cred], -1)))
        # right1 = self.gelu(self.rightlinear(evd_cred[:,:,0].unsqueeze(-1).repeat(1, 1, right.shape[-1]) * right))
        # right2 = self.gelu2(self.rightlinear2(evd_cred[:,:,1].unsqueeze(-1).repeat(1, 1, right.shape[-1]) * right))
        # right = right1 + right2
        
        
        #self-att############################
        # doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        # mask2 = torch.zeros(mask.size(), device="cuda:0")
        # mask2 = mask2.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # right = self.transencoder(src=right, src_key_padding_mask=mask2)
        # mu = self.mu_head(right)
        # logvar = self.logvar_head(right)
        # right = self._reparameterize(mu, logvar)
        
        
        # evd_cred, right = self.evdCred(right, right, valid_lens)#32,30,2
        # mean cred per evidence
        # meanevd = torch.mean(evd_cred, dim=2)#32,30
        # doc_mask = (mask == 0)#32,30
        # meanevd = meanevd.masked_fill(doc_mask, 0)#32,30
        # maxevdid = torch.argmax(meanevd, dim=1)
        # evd = []

        # for i in range(right.shape[0]):
        #     evd.append(right[i][maxevdid[i]].tolist())
        # evd = torch.tensor(evd, device="cuda:0")

        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)

        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attention_weights = self.aoa_layer(torch.cat([attention_weights, evd_cred], -1))
        # attention_weights = F.softmax(attention_weights * evd_cred, dim=1)
        # attention_weights = attention_weights * evd_cred
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        # return attended, torch.cat((evd_cred, attention_weights), axis=-1), evd
        ######################################
        
        # return attended, attention_weights
        return attended, torch.cat((evd_cred, attention_weights), axis=-1)


class BiLinearTanh(nn.Module):

    def __init__(self, left_dim: int, right_dim: int, out_dim: int):
        """
        Implementation of equation v_s^T \tanh(W_1 * h_{ij} + W_s * x + b_s)
        Parameters
        ----------
        left_dim: `int` dimension of left tensor
        right_dim: `int` dimesion of right tensor
        out_dim
        """
        super().__init__()
        self.left_linear = nn.Linear(left_dim, out_dim, bias=True)
        self.right_linear = nn.Linear(right_dim, out_dim, bias=False)
        self.combine = nn.Linear(out_dim, 1, bias=False)

    def forward(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, mask: torch.Tensor):
        """
        compute attention weights on left tensor based on the right tensor.
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (B, L, H)
        right_tsr: `torch.Tensor` of shape (B, D)
        mask: `torch.Tensor` of shape (B, L) 1 is for real, 0 is for pad

        Returns
        -------

        """
        assert len(left_tsr.size()) == 3 and len(mask.size()) == 2
        left = self.left_linear(left_tsr)  # (B, L, O)
        right = self.right_linear(right_tsr).unsqueeze(1)  # (B, O)
        tmp = torch.tanh(left + right)  # (B, L, O)
        linear_out = self.combine(tmp).squeeze(-1)  # (B, L)  it is equal to v_s^T \tanh(W_1 * h_{ij} + W_2 * a + b_s)
        doc_mask = (mask == 0)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim = -1)  # (B, L)
        attended = left_tsr * attention_weights.unsqueeze(-1)  # (B, L, H)
        attended = torch.sum(attended, dim = 1)  # (B, H)
        return attended, attention_weights


class MultiHeadAttentionSimple(nn.Module):

    def __init__(self, num_heads: int, d_model: int, d_key: int, d_value: int,
                 # attention_type: int = AttentionType.ConcatNotEqual,
                 init_weights: bool = False,
                 use_layer_norm: bool = False):
        """
        Simple multi-head attention and customizable with layer-norm
        Parameters
        ----------
        num_heads: `int` the number of heads. how many aspects of the evidences you want to see
        d_model: `int` input embedding size
        d_key: `int` dimension of keys. We will set d_key = d_model
        d_value: `int` dimensions of key, d_value = d_model
        init_weights: `bool` whether we should init linear layers.
        use_layer_norm: `bool` whether we should use layer-norm
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model, self.d_key, self.d_value = d_model, d_key, d_value
        assert d_model == d_key == d_value
        self.use_layer_norm = use_layer_norm
        self.w_qs = nn.Linear(d_model, num_heads * d_key)  # gom tat ca head vo 1 matrix de nhan co de
        self.w_ks = nn.Linear(d_model, num_heads * d_key)
        self.w_vs = nn.Linear(d_model, num_heads * d_value)
        if init_weights:
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_key)))
            nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_key)))
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_value)))

        # if attention_type == AttentionType.ConcatNotEqual:
        self.attention_func = ConcatNotEqualSelfAttTransFormer(inp_dim=(d_key + d_key), out_dim=d_key)
        # else:
        #     self.attention_func = ScaledDotProductAttention(temperature=np.power(d_key, 0.5))

        self.fc = nn.Linear(num_heads * d_value, d_model)
        if init_weights: nn.init.xavier_normal_(self.fc.weight)
        if use_layer_norm: self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        B, L, D = right.size()
        assert D == self.d_model == self.d_key, "Must have same shape"
        len_q = 1
        # transform
        query = self.w_qs(left).view(B, len_q, self.num_heads, self.d_key)  # (B, 1, num_heads, d_key)
        key = self.w_ks(right).view(B, L, self.num_heads, self.d_key)  # (B, L, num_heads, d_key)
        value = self.w_vs(right).view(B, L, self.num_heads, self.d_value)  # (B, L, num_heads, d_value)
        # reshape
        q = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_key)  # (num_heads * B) x 1 x dk
        k = key.permute(2, 0, 1, 3).contiguous().view(-1, L, self.d_key)  # (num_heads * B) x L x dk
        v = value.permute(2, 0, 1, 3).contiguous().view(-1, L, self.d_value)  # (num_heads * B) x L x dv
        # compute attention weights
        mask = (mask == 0)
        mask = mask.unsqueeze(1).repeat(self.num_heads, 1, 1)  # (B * num_heads, 1, L)
        attended, attention_weights = self.attention_func(query=q, key=k, value=v, mask=mask)
        # concat all heads and push to MLP followed by optional layer_norm
        output = attended.view(self.num_heads, B, len_q, self.d_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(B, len_q, -1)  # b x lq x (n*dv)

        tmp = self.fc(output)
        if self.use_layer_norm: tmp = self.layer_norm(tmp)
        return tmp, attention_weights


class MultiHeadAttentionOriginal(nn.Module):
    ''' Multi-Head Attention module copied from PyTorch Transformer '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """

        Parameters
        ----------
        n_head: `int` number of attention layers or heads
        d_model: `int` what the fuck is d_model? is it word embedding size?
        d_k
        d_v
        dropout
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)  # gom tat ca head vo 1 matrix de nhan co de
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = ScaledDotProductAttention(temperature=np.power(1, 1))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """

        Parameters
        ----------
        q: `torch.Tensor` of shape (B, L, D)
        k: `torch.Tensor` of shape (B, R, D)
        v: `torch.Tensor` of shape (B, R, D)
        mask: `torch.Tensor` of shape (B, L, R) (very important, 1 is for

        Returns
        -------

        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..  (quite redundant here)
        output, _ = self.attention(q, k, v, mask = mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        # output = self.dropout(self.fc(output))
        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output, None


class ConcatNotEqualSelfAttTransFormer(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        # self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        query: `torch.Tensor` of shape (B, 1, X) X is not necessarily equal to D
        key: `torch.Tensor` of shape (B, L, D)
        value: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert query.size(0) == key.size(0), "Must same dimensions"
        # assert len(query.size()) == 2 and len(key.size()) == 3
        assert self.inp_dim == (query.size(-1) + key.size(-1))  # due to concat
        B, L, D = key.size()
        left_tmp = query.expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, key], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = mask.squeeze(1).unsqueeze(-1)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        # doc_mask = doc_mask.unsqueeze(-1).expand(B, L, 1)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(value.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)
        # self.softmax = nn.Softmax(dim=-1)  # are you sure the dimension is correct?

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """

        Parameters
        ----------
        query: `torch.Tensor` (n_heads * B, L, d_k)
        key: `torch.Tensor` (n_heads * B, L, d_k)
        value: `torch.Tensor` (n_heads * B, L, d_k)
        mask (n_heads * B, L, L) (this is I guess to remove padding tokens

        Returns
        -------

        """
        attn = torch.bmm(query, key.transpose(1, 2))
        # attn = attn / self.temperature

        if mask is not None: attn = attn.masked_fill(mask, -np.inf)
        attn = F.softmax(attn, dim = -1)  # exp of -np.inf would be zero (checked)
        attn = attn.masked_fill(mask, 0)  # reset nan
        # attn = self.dropout(attn)  # why there is a fucking shit dropout here???? (I've never seen this before)
        output = torch.bmm(attn, value)
        return output, attn


class CoDaAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, *input):
        pass