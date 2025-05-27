import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

torch.manual_seed(1)

START_TAG = '<START>'
STOP_TAG = '<STOP>'

EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def prepare_sequence(seq, to_ix):
    '''
    给定句长seq，和单词的索引映射to_ix，返回seq中的单词的索引
    '''
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs,dtype=torch.long)

def argmax(vec):
    '''
    给定张量vex，形状为(1, tagset_size)，返回最大值的索引
    '''
    # torch.max返回最大值和最大值的索引，只关注索引
    _, idx = torch.max(vec,1)

    return idx.item()

def log_sum_exp(vec):
    # 最大索引的得分，也就是最大得分,是一个值
    max_score = vec[0, argmax(vec)]
    # 广播为vec的形状，使得可以进行元素级的减法
    max_score_broadcast = max_score.view(1,-1).expand(1,vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size,embedding_dim)
        # 词嵌入维度，隐藏层单元（双向，所以一向要//2，一层lstm，双向lstm=true）
        self.lstm = nn.LSTM(embedding_dim,hidden_dim // 2,num_layers=1,bidirectional=True)
        # 全连接层，将lstm的输出映射到每个标签
        self.hidden2tag = nn.Linear(hidden_dim,self.tagset_size)
        # CRF转移矩阵，transition[i,j]表示从j转移到i的概率
        self.transitions = nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
        
        #其他标签到开始标签的概率与结束标签到其他标签的概率要很小，使得这样的情况不出现
        self.transitions.data[tag_to_ix[START_TAG],:] = -10000
        self.transitions.data[:,tag_to_ix[STOP_TAG]] = -10000 

        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        '''
        初始化bilstm的隐藏层状态为随机的标志正态数，形状为(num_layers*bidirection,batch_size,hidden_size)
        '''
        return (torch.randn(2,1,self.hidden_dim//2),torch.randn(2,1,self.hidden_dim//2))
    
    def _forward_alg(self, feats):
        '''
        CRF前向转播算法,计算可能的总分数,或者说是概率
        feats:发射矩阵(emissions score),有bilstm输出,表示每个词对应每个标签的得分
        return:
        所有可能路径和的对数
        '''
        # 其他标签初始值小，start标签初始值为0，表示从start开始
        init_alphas = torch.full((1,self.tagset_size),-10000.0)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_alphas

        for feat in feats:
            alpha_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1,-1).expand(1,self.tagset_size) 
                trans_score = self.transitions[next_tag].view(1,-1)

                next_tag_var = forward_var + trans_score + emit_score

                alpha_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alpha_t).view(1,-1)
        
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self,sentence):
        '''
        获得feats也就是发射得分（emissions score）
        sentence:id化的序列,也就是单词索引的张量
        '''
        # 隐藏层初始化
        self.hidden = self.init_hidden()
        # 词嵌入
        embeds = self.word_embeds(sentence).view(len(sentence),1,-1)
        # lstm计算
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 形状匹配(seq_len,hidden_dim)
        lstm_out = lstm_out.view(len(sentence),self.hidden_dim)
        # 线性映射到标签,hidden2tag形状(seq_len,tagset_size)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        '''
        计算真实标签的路径函数
        feats为发射矩阵(emissions scores),形状为(seq_len,tagset_size)
        tag为标签的索引张量,形状为(seq_len,),作用是记录句子中的真实标签索引
        return
        score: 总路径得分
        '''
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]],dtype=torch.long),tags])
        for i, feat in enumerate(feats):
            # 累加从真实标签i到真实标签i+1的概率,在加当前词在当前真实标签tags[i+1]的发射分数
            score += self.transitions[tags[i + 1],tags[i]]+ feat[tags[i + 1]]
        #最后加上stop的分数
        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score 
    
    def _viterbi_decode(self, feats):
        """
        实现CRF的维特比（Viterbi）解码算法，用于在推理时找到得分最高的标签路径。
        Args:
            feats (torch.Tensor): 发射矩阵（emission scores），形状为 (seq_len, tagset_size)。
        Returns:
            tuple: (path_score, best_path)
                   path_score (torch.Tensor): 最优路径的总得分。
                   best_path (list): 最优标签路径的索引列表。
        """
        backpoint = []

        init_vvars = torch.full((1,self.tagset_size),-10000.0)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            #获取每一步的最佳id与最佳id的得分，将得分加入到viterbivars_t中
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            # 更新当前的最优路径分数
            forward_var = (torch.cat(viterbivars_t) + feat).view(1,-1)
            # 储存当前指针的后向指针列表
            backpoint.append(bptrs_t)
        
        # 指针到序列末尾，加入stop步的转移概率
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 找到最终得分最高的路径的最后一个标签
        best_tag_id = argmax(terminal_var)
        # 最优路径的得分
        path_score = terminal_var[0][best_tag_id]

        # 从最后一个标签开始，回溯标签
        best_path = [best_tag_id]
        # 从后向前遍历
        for bptrs_t in reversed(backpoint):
            # 根据当前标签找到其他最优的前驱标签
            best_tag_id = bptrs_t[best_tag_id]
            # 根据最优标签添加最优路径
            best_path.append(best_tag_id)
        
        # 移除start标签
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        # 转为正向标签
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        '''
        计算负对数似然损失函数
        '''
        # emissions score
        feats = self._get_lstm_features(sentence)
        # 归一化因子
        forward_score = self._forward_alg(feats)
        # 真实标签序列的路径得分
        gold_score = self._score_sentence(feats, tags)
        # loss
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

# ----数据与模型训练----

training_data = [('the wall street journal reported today that apple corporation made money'.split(),
                  'B I I I O O O B I O O'.split()),
                  ('georgia tech is a university in georgia'.split(),
                   'B I O O O O B'.split()
                )]

# 给sentence中的每一个单词赋予索引
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {'B': 0,'I': 1,'O': 2,START_TAG:3,STOP_TAG:4}

model = BiLSTM_CRF(len(word_to_ix),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(),lr = 0.01,weight_decay=1e-4)  # type: ignore

for epoch in range(300):
    for sentence, tag in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence,word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tag],dtype=torch.long)

        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0],word_to_ix)
    print(model(precheck_sent))
