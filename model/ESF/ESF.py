import torch
import torch.nn as nn

class EnableStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, dropout, num_layers=2):
        super(EnableStateModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_activities)

        
    def forward(self, x):
        # RNN处理事件序列
        rnn_out, _ = self.rnn(x)
        # 获取最后一个时间步的输出
        rnn_out = self.ln(rnn_out)
        rnn_out = self.dropout(rnn_out[:, -1, :])
        # 全连接层输出启用状态
        out = self.fc(rnn_out)
        out = torch.softmax(out, dim=-1)
        return out

class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_activities, dropout, num_layers=2):
        super(PredictionModel, self).__init__()
        self.cross_feature = nn.Linear(input_size*2, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc_1 = nn.Linear(hidden_size+num_activities, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, num_activities)
        self.relu = nn.ReLU()
    
    def forward(self, x, enable_states):
        input_feature = self.cross_feature(torch.cat([x, x], dim=-1))
        rnn_out, _ = self.rnn(input_feature)
        rnn_out = self.ln((rnn_out))
        final_hidden_state = self.dropout(rnn_out[:, -1, :])

        second_stage_input = torch.cat([final_hidden_state, enable_states], dim=-1)
        all_activities_output = self.relu(self.fc_1(second_stage_input))
        all_activities_output = self.fc_2(all_activities_output)
        
        return all_activities_output  # 直接返回 logits


class EnableStateFilterModel(nn.Module):
    def __init__(self, activity_num, dimension, hidden_size_1, hidden_size_2, add_attr_num, dropout):
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
        self.stage2 = PredictionModel(self.input_feature_size, hidden_size_2, activity_num, dropout)

    def get_input_feature(self, batch_data):
        # batch_data:(B, dim, max_len)
        activity_feature = batch_data[:, 0, :].long()
        unseen_idx = activity_feature == (self.activity_num + 1)
        if unseen_idx.any():
            activity_feature = torch.clamp(activity_feature, min=0, max=self.activity_num)
            activity_embeddings = self.activity_embedding(activity_feature)
            unseen_activity_embedding = torch.mean(self.activity_embedding.weight[1:],dim=0)
            activity_embeddings[unseen_idx] = unseen_activity_embedding
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
                    category_embedding[unseen_idx] = unseen_embedding
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
        # input_feature:(B, max_len, dim_new)
        base_feature, input_feature = self.get_input_feature(batch_data)
        assert input_feature.shape[2] == self.input_feature_size, "incorrect feature size"
        # enable_state:(B, activity_num)
        enable_state = self.stage1(base_feature)
        prediction = self.stage2(input_feature, enable_state)
        return enable_state, prediction