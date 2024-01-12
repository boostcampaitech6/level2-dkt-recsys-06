import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


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
        self.comb_proj = nn.Linear(intd * (len(args.cat_feats) - 1), dim_cat)
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
        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
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

        X = self.comb_proj(embed)  # embedding linear projection
        X = self.layer_norm_cat(X)

        # 수치형 변수
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
        self.lstm = nn.LSTM(
            self.hidden_dim * 2, self.hidden_dim, self.n_layers, batch_first=True
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
        super().__init__(hidden_dim, n_layers, n_tests, n_questions, n_tags)
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
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
