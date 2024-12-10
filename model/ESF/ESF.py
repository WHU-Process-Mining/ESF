import torch
import torch.nn as nn

class EnableStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, dropout, num_layers=2):
        super(EnableStateModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_activities)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # RNN处理事件序列
        rnn_out, _ = self.rnn(x)
        # 获取最后一个时间步的输出
        rnn_out = self.ln(rnn_out)
        rnn_out = self.dropout(rnn_out[:, -1, :])
        # 全连接层输出启用状态
        out = self.fc(rnn_out)
        out = self.sigmoid(out)
        return out

class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, dropout, threshold, num_layers=1):
        super(PredictionModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_activities)
        self.relu = nn.ReLU()
        self.threshold = threshold
    
    def forward(self, x, first_stage_scores):
        rnn_out, _ = self.rnn(x)
        rnn_out = self.ln((rnn_out))
        final_hidden_state = self.dropout(rnn_out[:, -1, :])
        all_activities_output = self.relu(self.fc(final_hidden_state))
    
        # 创建启用活动的掩码
        activity_mask = (first_stage_scores >= self.threshold)
        
        # 对 logits 应用掩码
        masked_logits = all_activities_output.masked_fill(~activity_mask, -1e4)
        
        return masked_logits  # 直接返回 logits


class EnableStateFilterModel(nn.Module):
    def __init__(self, activity_num, dimension, hidden_size_1, hidden_size_2, add_attr_num, threshold, dropout):
        super(EnableStateFilterModel, self).__init__()
        self.activity_num = activity_num
        self.dimension = dimension
        self.add_attr_num = add_attr_num
        self.activity_embedding = nn.Embedding(activity_num + 1, dimension, padding_idx=0)
        self.add_attr_embeddings = nn.ModuleList()
        for attr_num in add_attr_num:
            if attr_num > 0:
                embedding = nn.Embedding(attr_num + 1, dimension, padding_idx=0)
                self.add_attr_embeddings.append(embedding)
            else:
                self.add_attr_embeddings.append(None)  # For numeric features

        # input feature size
        embed_dimension = (1 + sum(1 for num in add_attr_num if num > 0)) * dimension  # 活动特征和分类特征的嵌入维度总和
        numeric_dimension = add_attr_num.count(0) + 4  # 数值特征数量（额外特征中的数值特征 + 时间特征）
        self.input_feature_size = embed_dimension + numeric_dimension
        self.stage1 = EnableStateModel(dimension, hidden_size_1, activity_num, dropout)
        self.stage2 = PredictionModel(self.input_feature_size, hidden_size_2, activity_num, dropout, threshold)

    def get_input_feature(self, batch_data):
        # batch_data:(B, dim, max_len)
        activity_feature = batch_data[:, 0, :].long()
        activity_embeddings = self.activity_embedding(activity_feature)
        time_feature = torch.permute(batch_data[:, 1:5, :], (0, 2, 1))

        category_embeddings = [activity_embeddings]
        numeric_embeddings = [time_feature]
        for i, attr_num in enumerate(self.add_attr_num):
            add_feature = batch_data[:, 5 + i, :]
            if attr_num > 0:
                category_embeddings.append(self.add_attr_embeddings[i](add_feature.long()))
            else:
                numeric_embeddings.append(torch.unsqueeze(add_feature, dim=2))

        category_embeddings = torch.cat(category_embeddings, dim=-1)
        numeric_embeddings = torch.cat(numeric_embeddings, dim=-1)
        base_feature = torch.cat((activity_embeddings, time_feature), dim=-1)
        input_feature = torch.cat((category_embeddings, numeric_embeddings), dim=-1)
        return activity_embeddings, input_feature

    def forward(self, batch_data):
        # input_feature:(B, max_len, dim_new)
        base_feature, input_feature = self.get_input_feature(batch_data)
        assert input_feature.shape[2] == self.input_feature_size, "incorrect feature size"
        # enable_state:(B, activity_num)
        enable_state = self.stage1(base_feature)
        prediction = self.stage2(input_feature, enable_state)
        return enable_state, prediction