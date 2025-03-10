import torch
import torch.nn as nn
import math

class EnableStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, dropout, num_layers=2):
        super(EnableStateModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout= dropout,batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_activities)

        
    def forward(self, x, prefix_mask=None):
        if prefix_mask is not None:
            seq_lengths = prefix_mask.sum(dim=1)
            assert (seq_lengths > 0).all(), "存在长度为 0 的序列！"
            packed = nn.utils.rnn.pack_padded_sequence(x, seq_lengths.cpu().to(torch.int64), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            idx = (seq_lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, lstm_out.size(2))
            h_t  = lstm_out.gather(1, idx).squeeze(1)
        else:
            _, (h_n, c_n) = self.lstm(x)
            h_t  = h_n[-1]

        # 获取最后一个时间步的输出
        lstm_out = self.ln(h_t)
        lstm_out = self.dropout(lstm_out)
        # 全连接层输出启用状态
        out = self.fc(lstm_out)
        return out

class PositionalEncoding(nn.Module):
    """Inject sequence information in the prefix or suffix embeddings 
    before feeding them to the stack of encoders or decoders respectively. 

    Predominantly based on the PositionalEncoding module defined in 
    https://github.com/pytorch/examples/tree/master/word_language_model. 
    This reimplemetation, in contrast to the original one, caters for 
    adding sequence information in input embeddings where the batch 
    dimension comes first (``batch_first=True`). 

    Parameters
    ----------
    d_model : int
        The embedding dimension adopted by the associated Transformer. 
    dropout : float
        Dropout value. Dropout is applied over the sum of the input 
        embeddings and the positional encoding vectors. 
    max_len : int
        the max length of the incoming sequence. By default 10000. 
    """


    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Check if d_model is an integer and is even
        assert isinstance(d_model, int), "d_model must be an integer"
        assert d_model % 2 == 0, "d_model must be an even number"
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe) # shape (max_len, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Sequence of prefix event tokens or suffix event tokens fed 
            to the positional encoding module. Shape 
            (batch_size, window_size, d_model).

        Returns
        -------
        Updated sequence tensor of the same shape, with sequence 
        information injected into it, and dropout applied. 
        """
        x = x + self.pe[:x.size(1), :] # (batch_size, window_size, d_model)
        return self.dropout(x)
    
class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, embedding_size, dropout, threhold, num_layers=2):
        super(PredictionModel, self).__init__()
        self.threhold = threhold
        self.W_1 = nn.Linear(input_size, hidden_size, bias=False)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.W_2 = nn.Linear(embedding_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)
        # self.fc_1 = nn.Linear(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_activities)
        self.relu = nn.ReLU()
    
    def forward(self, x, enable_states, activity_embeddings, prefix_mask, pooling='weighted_mean'):
        # (batch_size, num_activities)
        candidate_mask = torch.sigmoid(enable_states) >=self.threhold

        if pooling == 'max':
            batch_candidate_embeddings = activity_embeddings.unsqueeze(0).expand(candidate_mask.size(0), -1, -1)  # (batch_size, num_activities, embedding_dim)
            masked_embeddings = batch_candidate_embeddings.clone()
            masked_embeddings[~candidate_mask.unsqueeze(-1).expand_as(batch_candidate_embeddings)] = float('-inf')
            global_candidate_embedding, _ = masked_embeddings.max(dim=1) # (batch_size, embedding_dim)

            all_invalid = candidate_mask.sum(dim=1) == 0
            if all_invalid.any():
                default_embedding = activity_embeddings.mean(dim=0)  # (embedding_dim,)
                global_candidate_embedding[all_invalid] = default_embedding.unsqueeze(0).expand(all_invalid.sum(), -1)
        elif pooling =='mean':
            weighted_candidate = torch.matmul(candidate_mask.float(), activity_embeddings)
            candidate_counts = candidate_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            global_candidate_embedding = weighted_candidate / candidate_counts  # (batch_size, embedding_dim)
        elif pooling =='weighted_mean':
            candidate_weights = torch.sigmoid(enable_states) * candidate_mask.float()
            weighted_candidate = torch.matmul(candidate_weights, activity_embeddings)
            candidate_counts = candidate_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            global_candidate_embedding = weighted_candidate / candidate_counts
        else:
            raise ValueError("Invalid pooling method. Choose 'max' or 'mean'.")

        # (batch_size, seq_len, hidden_size)
        x = self.W_1(x)
        x = self.positional_encoding(x* math.sqrt(x.shape[-1]))
        prefix_encoded = self.transformer_encoder(x,)
        prefix_encoded = self.ln(prefix_encoded)
        # (batch_size, hidden_size)
        global_candidate_embedding = self.W_2(global_candidate_embedding)
        # (batch_size, seq_len)
        scores = torch.einsum('bh,bth->bt', global_candidate_embedding, prefix_encoded) / torch.sqrt(torch.tensor([global_candidate_embedding.shape[-1]], dtype=torch.float32,device=x.device))
        extended_mask = ~prefix_mask.bool()  # (B, seq_len)，True 表示需要屏蔽的位置
        scores = scores.masked_fill(extended_mask, float('-1e4'))
        alpha = torch.softmax(scores, dim=-1)
        # 计算候选活动的上下文表示：对前缀原始编码（prefix_encoded）加权求和
        candidate_ctx = torch.einsum('bt,bth->bh', alpha, prefix_encoded) # (batch_size, hidden_size)
        
        # all_activities_output = self.dropout(self.relu(self.fc_1(candidate_ctx)))
        all_activities_output = self.fc(candidate_ctx)
        all_activities_output = all_activities_output.masked_fill(~candidate_mask, float('-1e4'))
        return all_activities_output  # 直接返回 logits


class EnableStateFilterModel(nn.Module):
    def __init__(self, activity_num, dimension, hidden_size_1, hidden_size_2, add_attr_num, dropout, threhold=0.5):
        super(EnableStateFilterModel, self).__init__()
        self.activity_num = activity_num
        self.add_attr_num = add_attr_num
        activity_dim = min(600, round(1.6 * activity_num**0.56))
        self.activity_embedding = nn.Embedding(activity_num + 1, activity_dim, padding_idx=0)
        self.add_attr_embeddings = nn.ModuleList()
        embed_dimension = activity_dim
        for attr_num in add_attr_num:
            if attr_num > 0:
                cat_dim = min(600, round(1.6 * attr_num**0.56))
                embedding = nn.Embedding(attr_num + 1, cat_dim, padding_idx=0)
                embed_dimension += cat_dim
                self.add_attr_embeddings.append(embedding)
            else:
                self.add_attr_embeddings.append(None)  # For numeric features

        numeric_dimension = add_attr_num.count(0) + 4  # 数值特征数量（额外特征中的数值特征 + 时间特征）
        self.input_feature_size = embed_dimension + numeric_dimension
        self.stage1 = EnableStateModel(activity_dim, hidden_size_1, activity_num, dropout)
        self.stage2 = PredictionModel(self.input_feature_size, hidden_size_2, activity_num, activity_dim, dropout, threhold)

    def get_input_feature(self, batch_data):
        # batch_data:(B, dim, max_len)
        activity_feature = batch_data[:, 0, :].long()
        unseen_idx = activity_feature == (self.activity_num + 1)
        if unseen_idx.any():
            activity_feature = torch.clamp(activity_feature, min=0, max=self.activity_num)
            activity_embeddings = self.activity_embedding(activity_feature)
            unseen_activity_embedding = torch.mean(self.activity_embedding.weight[1:],dim=0)
            activity_embeddings = torch.where(unseen_idx.unsqueeze(-1), 
                                  unseen_activity_embedding.unsqueeze(0).expand_as(activity_embeddings),
                                  activity_embeddings)
        else:
            activity_embeddings = self.activity_embedding(activity_feature)

        time_feature = torch.permute(batch_data[:, 1:5, :], (0, 2, 1))

        category_embeddings = [activity_embeddings]
        numeric_embeddings = [time_feature]
        for i, attr_num in enumerate(self.add_attr_num):
            add_feature = batch_data[:, 5 + i, :]
            if attr_num > 0:
                catogery_feature = add_feature.long()
                unseen_idx = catogery_feature == (attr_num + 1)
                if unseen_idx.any():
                    catogery_feature = torch.clamp(catogery_feature, min=0, max=attr_num)
                    category_embedding = self.add_attr_embeddings[i](catogery_feature)
                    unseen_embedding = torch.mean(self.add_attr_embeddings[i].weight[1:],dim=0)
                    category_embedding = torch.where(unseen_idx.unsqueeze(-1), 
                                            unseen_embedding.unsqueeze(0).expand_as(category_embedding), 
                                            category_embedding
                                        )
                    category_embeddings.append(category_embedding)
                else:
                    category_embeddings.append(self.add_attr_embeddings[i](catogery_feature))
            else:
                numeric_embeddings.append(torch.unsqueeze(add_feature, dim=2))

        category_embeddings = torch.cat(category_embeddings, dim=-1)
        numeric_embeddings = torch.cat(numeric_embeddings, dim=-1)
        base_feature = torch.cat((activity_embeddings, time_feature), dim=-1)
        input_feature = torch.cat((category_embeddings, numeric_embeddings), dim=-1)
        return activity_embeddings, input_feature

    def forward(self, batch_data):
        activity_seq = batch_data[:, 0, :].long()
        mask = activity_seq != 0
        # input_feature:(B, max_len, dim_new)
        base_feature, input_feature = self.get_input_feature(batch_data)
        assert input_feature.shape[2] == self.input_feature_size, "incorrect feature size"
        # enable_state:(B, activity_num)
        enable_state = self.stage1(base_feature, mask)
        prediction = self.stage2(input_feature, enable_state, self.activity_embedding.weight[1:], mask)

        p_enable = torch.sigmoid(enable_state)
        candidate_probs = torch.softmax(prediction, dim=-1)
        log_p_enable = torch.log(p_enable + 1e-8)
        log_candidate = torch.log(candidate_probs + 1e-8)
        if self.stage2.threhold == 0: # 第二阶段
            return enable_state, prediction
        prediction = log_p_enable + log_candidate
        return enable_state, prediction