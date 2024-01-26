import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel
import pickle
import numpy as np
<<<<<<< HEAD
import copy
import re
=======
>>>>>>> wonhee


# 범주형 -> embedding -> linear
class ModelBase(nn.Module):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags
        self.args = args

        ## CATEGORICALS
        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        dim_cat, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(
            3, intd
        )  # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # 추가된 범주형 feature 있으면 embedding 만들기
        self.new_embeddings = []
        if len(self.args.new_cat_feats) > 0:
            for n_cat in self.args.n_cat_feats:
                self.new_embeddings.append(
                    nn.Embedding(n_cat + 1, intd).to(self.args.device)
                )

        # Concatentaed Embedding Linear Projection
<<<<<<< HEAD
<<<<<<< HEAD
        self.comb_proj = nn.Linear(intd * (len(args.cat_feats) - 1), dim_cat)
=======
        if self.args.graph_embed:
            self.comb_proj_dim = intd * (len(self.args.cat_feats) - 1)+64 # graph
        else:
            self.comb_proj_dim = intd * (len(self.args.cat_feats) - 1) # new
        
        print(self.comb_proj_dim, dim_cat)
        self.comb_proj = nn.Linear(self.comb_proj_dim, dim_cat)
>>>>>>> wooksbaby
=======
        if self.args.graph_embed:
            self.comb_proj_dim = intd * (len(self.args.cat_feats) - 2) + 64  # graph
        else:
            self.comb_proj_dim = intd * (len(self.args.cat_feats) - 1)  # new

        print(self.comb_proj_dim, dim_cat)
        self.comb_proj = nn.Linear(self.comb_proj_dim, dim_cat)
>>>>>>> wonhee
        self.layer_norm_cat = nn.LayerNorm(dim_cat)

        ## NUMERICALS
        # linear: 수치형 변수 두 개 이상 -> linear -> layer_norm
        dim_num = 0
        if len(self.args.num_feats) > 1:
            dim_num = dim_cat
            self.comb_nums = nn.Linear(len(self.args.num_feats), dim_num).to(
                self.args.device
            )  # 수치형 추상화
            self.layer_norm_num = nn.LayerNorm(dim_num)

        # Fully connected layer: output layer
        self.fc = nn.Linear(hidden_dim, 1)

    # def forward(self, test, question, tag, correct, mask, interaction):
    def forward(self, data):
        interaction, test, question, tag, correct, mask = (
            data["interaction"],
            data["test"],
            data["question"],
            data["tag"],
            data["correct"],
            data["mask"],
        )

        batch_size = interaction.size(0)
<<<<<<< HEAD
<<<<<<< HEAD
        # Embedding
=======
        
        ## 1) 범주형변수 Embedding
>>>>>>> wooksbaby
=======

        ## 1) 범주형변수 Embedding
>>>>>>> wonhee
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_tag,
            ],
            dim=2,
        )

        # 새로운 범주형 변수의 embedding을 concatenate
        if len(self.new_embeddings) > 0:
            for i, new_embedding in enumerate(self.new_embeddings):
                cat_feat = data[f"new_cat_feats_{i}"]
                temp = new_embedding(cat_feat.int())
                embed = torch.cat([embed, temp], dim=2)
<<<<<<< HEAD
=======
        
        # graph embedding
        if self.args.graph_embed:
            embed_graph = data['embed_graph']
            embed = torch.cat([embed, embed_graph], dim=2)
        # else: # just embedding
        embed = torch.cat([embed, embed_question], dim=2)
>>>>>>> wooksbaby

        # graph embedding
        if self.args.graph_embed:
            embed_graph = data["embed_graph"]
            embed = torch.cat([embed, embed_graph], dim=2)
        else:  # just embedding
            embed = torch.cat([embed, embed_question], dim=2)

        X = self.comb_proj(embed)  # embedding linear projection
        X = self.layer_norm_cat(X)

<<<<<<< HEAD
<<<<<<< HEAD
        # 수치형 변수
=======

        ## 2) 수치형 변수
>>>>>>> wooksbaby
=======
        ## 2) 수치형 변수
>>>>>>> wonhee
        if len(self.args.num_feats) > 1:
            num_feat = data["num_feats_0"].reshape(batch_size, -1, 1)
            for i in range(1, len(self.args.num_feats)):
                tmp = data[f"num_feats_{i}"].reshape(batch_size, -1, 1)
                num_feat = torch.cat([num_feat, tmp], dim=2)

            X_num = self.comb_nums(
                num_feat
            )  # [batch_size, seq_len, len(num_feats)] -> [b,s,hd]
            X_num = self.layer_norm_num(X_num)

            X = torch.cat([X, X_num], dim=2)

        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs,
    ):
        super().__init__(args, hidden_dim, n_layers, n_tests, n_questions, n_tags)

        self.args = args

        if len(self.args.new_num_feats) > 1:
            self.lstm = nn.LSTM(
                self.hidden_dim * 2, self.hidden_dim, self.n_layers, batch_first=True
            )
        else:
            self.lstm = nn.LSTM(
                self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
            )

    # def forward(self, test, question, tag, correct, mask, interaction):
    def forward(self, data):
        X, batch_size = super().forward(data=data)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs,
    ):
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(
            test=test,
            question=question,
            tag=tag,
            correct=correct,
            mask=mask,
            interaction=interaction,
        )

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs,
    ):
<<<<<<< HEAD
<<<<<<< HEAD
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
=======
        super().__init__(args, hidden_dim, n_layers, n_tests, n_questions, n_tags)
>>>>>>> wooksbaby
=======
        super().__init__(args, hidden_dim, n_layers, n_tests, n_questions, n_tags)
>>>>>>> wonhee
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

<<<<<<< HEAD
<<<<<<< HEAD
    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(
            test=test,
            question=question,
            tag=tag,
            correct=correct,
            mask=mask,
            interaction=interaction,
        )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
=======
    def forward(self, data): #, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(data=data)

        # X, batch_size = super().forward(
        #     test=test,
        #     question=question,
        #     tag=tag,
        #     correct=correct,
        #     mask=mask,
        #     interaction=interaction,
        # )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=data['mask'])
>>>>>>> wooksbaby
=======
    def forward(self, data):  # , test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(data=data)

        # X, batch_size = super().forward(
        #     test=test,
        #     question=question,
        #     tag=tag,
        #     correct=correct,
        #     mask=mask,
        #     interaction=interaction,
        # )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=data["mask"])
>>>>>>> wonhee
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


<<<<<<< HEAD

=======
>>>>>>> wonhee
class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
<<<<<<< HEAD
=======

>>>>>>> wonhee
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

<<<<<<< HEAD
    def forward(self,ffn_in):
=======
    def forward(self, ffn_in):
>>>>>>> wonhee
        x = self.layer1(ffn_in)
        x = nn.ReLU()(x)
        x = self.layer2(x)
        return x


<<<<<<< HEAD

class LastQuery(ModelBase):
    def __init__(self,
                 args,
                hidden_dim: int = 64,
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                **kwargs):
        
=======
class LastQuery(ModelBase):
    def __init__(
        self,
        args,
        hidden_dim: int = 64,
        n_layers: int = 1,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs,
    ):
>>>>>>> wonhee
        super().__init__(args, hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.args = args
        self.device = self.args.device

        if self.args.num_feats:
<<<<<<< HEAD
            self.hidden_dim = 2*self.args.hidden_dim
        else:
            self.hidden_dim = self.args.hidden_dim

=======
            self.hidden_dim = 2 * self.args.hidden_dim
        else:
            self.hidden_dim = self.args.hidden_dim

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        # self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        # self.embedding_test = nn.Embedding(self.args.n_tests + 1, self.hidden_dim//3)
        # self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        # self.embedding_tag = nn.Embedding(self.args.n_tags + 1, self.hidden_dim//3)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

>>>>>>> wonhee
        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # Encoder
<<<<<<< HEAD
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
=======
        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=self.args.n_heads
        )
        self.mask = None  # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
>>>>>>> wonhee
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
<<<<<<< HEAD
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # self.gru = nn.GRU(
        #     self.hidden_dim,
        #     self.hidden_dim,
        #     self.args.n_layers,
        #     batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

               # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0
        print('----------------- initialization ... -------------------') 
        # regular expression으로 layer 파악
        for name, param in self.named_parameters():

            print(name)
            if re.match(r'^embedding*', name): #embedding: normal dist init
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*|.*layer_norm.*', name): #layernorm or batchnorm 통과
                continue
            elif re.match(r'.*query.weight*|.*key.weight*|.*value.weight*', name):
                nn.init.xavier_uniform_(param)
            elif re.match(r'.*weight*', name): # weight: normal dist init
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}
        print(f'------------------- scaling... --------------------')
        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param
            elif re.match(r'.*ffn.*weight$|.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r".*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        print('[loading]')
        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            print(name)
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)

=======
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()
>>>>>>> wonhee

    def get_mask(self, seq_len, index, batch_size):
        """
        batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다

        참고로 (batch_size*self.args.n_heads, seq_len, seq_len) 가 아니라
              (batch_size*self.args.n_heads,       1, seq_len) 로 하는 이유는

        last query라 output의 seq부분의 사이즈가 1이기 때문이다
        """
        # [[1], -> [1, 2, 3]
        #  [2],
        #  [3]]
        index = index.view(-1)

        # last query의 index에 해당하는 upper triangular mask의 row를 사용한다
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))
        mask = mask[index]

        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
<<<<<<< HEAD
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))
=======
        mask = mask.repeat(1, self.args.n_heads).view(
            batch_size * self.args.n_heads, -1, seq_len
        )
        return mask.masked_fill(mask == 1, float("-inf"))
>>>>>>> wonhee

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
<<<<<<< HEAD
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.hidden_dim)
=======
        h = torch.zeros(self.args.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.args.n_layers, batch_size, self.hidden_dim)
>>>>>>> wonhee
        c = c.to(self.device)

        return (h, c)

<<<<<<< HEAD

    def forward(self, data):
        # test, question, tag, _, mask, interaction, index = input
        batch_size = data['interaction'].size(0)
        seq_len = data['interaction'].size(1)
=======
    def forward(self, data):
        # test, question, tag, _, mask, interaction, index = input
        batch_size = data["interaction"].size(0)
        seq_len = data["interaction"].size(1)
>>>>>>> wonhee

        embed, batch_size = super().forward(data=data)
        # 신나는 embedding
        # embed_interaction = self.embedding_interaction(interaction)
        # embed_test = self.embedding_test(test)
        # embed_question = self.embedding_question(question)
        # embed_tag = self.embedding_tag(tag)

        # embed = torch.cat([embed_interaction,
        #                    embed_test,
        #                    embed_question,
        #                    embed_tag,], 2)

        # embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
<<<<<<< HEAD
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)

        # 이 3D gathering은 머리가 아픕니다. 잠시 머리를 식히고 옵니다.
        # q = torch.gather(q, 1, index.repeat(1, self.hidden_dim).unsqueeze(1)) # 마지막 쿼리 빼고 다 가리기
=======
        q = self.query(embed)[:, -1:, :]

        # 이 3D gathering은 머리가 아픕니다. 잠시 머리를 식히고 옵니다.
        # q = torch.gather(q, 1, index.repeat(1, self.hidden_dim).unsqueeze(1)) # 마지막 쿼리 빼고 다 가리기
        q = q.permute(1, 0, 2)
>>>>>>> wonhee

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        # self.mask = self.get_mask(seq_len, index, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)
<<<<<<< HEAD
        # out, hidden = self.gru(out, hidden[0])

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        preds = self.fc(out).view(batch_size, -1)

        # preds = self.activation(out).view(batch_size, -1)

        return preds
    

    
class LastQuery_encoder(nn.Module):
    def __init__(self,
                 args,
                hidden_dim: int = 64,
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                **kwargs):
        
        super().__init__()

        self.args = args

        if self.args.num_feats:
            self.hidden_dim = 2*self.args.hidden_dim
        else:
            self.hidden_dim = self.args.hidden_dim

        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
    

    def forward(self, embed):

        q = self.query(embed)[:,-1,:].permute(1,0,2)

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        # self.mask = self.get_mask(seq_len, index, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        return out


class LastQuery2(ModelBase):
    def __init__(self,
                 args,
                hidden_dim: int = 64,
                n_layers: int = 2,
                n_tests: int = 1538,
                n_questions: int = 9455,
                n_tags: int = 913,
                **kwargs):
        
        super().__init__(args, hidden_dim, n_layers, n_tests, n_questions, n_tags)
        self.args = args
        self.device = self.args.device

        if self.args.num_feats:
            self.hidden_dim = 2*self.args.hidden_dim
        else:
            self.hidden_dim = self.args.hidden_dim

        # Embedding

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        # Encoder            
        self.encoder = nn.Sequential(*[copy.deepcopy(LastQuery_encoder(args=self.args)) for _ in range(self.args.n_layers)])

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            1,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)


    def get_mask(self, seq_len, index, batch_size):
        """
        batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다

        참고로 (batch_size*self.args.n_heads, seq_len, seq_len) 가 아니라
              (batch_size*self.args.n_heads,       1, seq_len) 로 하는 이유는

        last query라 output의 seq부분의 사이즈가 1이기 때문이다
        """
        # [[1], -> [1, 2, 3]
        #  [2],
        #  [3]]
        index = index.view(-1)

        # last query의 index에 해당하는 upper triangular mask의 row를 사용한다
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))
        mask = mask[index]

        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, data):
        # test, question, tag, _, mask, interaction, index = input
        batch_size = data['interaction'].size(0)
        seq_len = data['interaction'].size(1)

        embed, batch_size = super().forward(data=data)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        out = self.encoder(embed)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)

        # preds = self.activation(out).view(batch_size, -1)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(nn.Module):

    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.

        ### Embedding
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//3)*3, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)

        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)


        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers,
            num_decoder_layers=self.args.n_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag,], 2)

        embed_enc = self.enc_comb_proj(embed_enc)

        # DECODER
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed_interaction = self.embedding_interaction(interaction)

        embed_dec = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_interaction], 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)

        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)

        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)


        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)

        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)

        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds
=======

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        # preds = self.activation(out).view(batch_size, -1)

        return out.view(batch_size, -1)
>>>>>>> wonhee
