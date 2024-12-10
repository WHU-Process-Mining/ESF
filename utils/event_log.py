import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

def get_time_feature(time_seq):
    case_interval = [i[-1]-i[0]  for i in time_seq]
    event_interval = [[i[j] - i[j-1] for j in range(1, len(i))]  for i in time_seq]
    max_case_interval = max([86400 * i.days + i.seconds  for i in case_interval])
    min_case_interval = min([86400 * i.days + i.seconds  for i in case_interval])
    max_event_interval = max([max(86400 * j.days + j.seconds for j in i)  for i in event_interval])
    min_event_interval = min([min(86400 * j.days + j.seconds for j in i)  for i in event_interval])

    return {'max_case_interval': max_case_interval,
            'min_case_interval': min_case_interval,
            'max_event_interval': max_event_interval,
            'min_event_interval': min_event_interval}

def get_category_feature(category_set):
    category_list = [str(x) if isinstance(x, float) and math.isnan(x) else x for x in category_set]
    category_list.sort()
    mapping = dict(zip(category_list, range(
            1, len(category_list) + 1)))
    return {'is_numeric':False,
            'mapping':mapping}

def get_numerical_feature(numerical_list):
    min_value = min(numerical_list)
    max_value = max(numerical_list)
    return {'is_numeric':True,
            'min': min_value,
            'max': max_value}

def split_valid_data(train_input, valid_ratio):
        input_data, future_activity = train_input
        valid_n = int(valid_ratio * len(input_data))

        train_input, val_input, train_future_activity, val_future_activity = train_test_split(
            input_data, future_activity, test_size=valid_n, shuffle=True
        )
        return [train_input, train_future_activity], [val_input, val_future_activity]

def split_valid_df(df, valid_ratio):
        all_cases = np.unique(df['case:concept:name'])
        valid_n = int(valid_ratio * len(all_cases))
        train_df_inx, val_df_ind = train_test_split(
            all_cases, test_size=valid_n, shuffle=True
        )
        train_df = df[df['case:concept:name'].isin(train_df_inx)]
        val_df = df[df['case:concept:name'].isin(val_df_ind)]
        return train_df, val_df


class EventLogData():
    def __init__(self, df, is_multi_attr = False):
        self.all_activities = np.unique(df['concept:name'])
        self.is_multi_attr = is_multi_attr
        self.feature_dict = {}
        if self.is_multi_attr:
            self.add_attr_names = []
            self.add_attr_num = []
        raw_data = self._generate_data_raw(df)
        case_num = len(np.unique(df['case:concept:name']))
        self.feature_dict['max_len'] = max([len(i[0])-1 for i in raw_data])
        self.feature_dict['activity'] = dict(zip(self.all_activities, range(1, len(self.all_activities) + 1)))
        self.feature_dict['time'] = get_time_feature([i[1] for i in raw_data])

        if self.is_multi_attr:
            attr_names = list(set(df.columns) - set(['case:concept:name', 'concept:name', 'time:timestamp']))
            attr_names.sort()  # sort the columns to ensure the order
            for index, attr_name in enumerate(attr_names):
                attr_seq = [i[index+2] for i in raw_data]
                attr_set = set([item for sublist in attr_seq for item in sublist])
                if len(attr_set) > 1:
                    if is_numeric_dtype(df[attr_name]):
                        self.add_attr_names.append(attr_name)
                        self.feature_dict[attr_name] = get_numerical_feature(attr_set)
                        self.add_attr_num.append(0)
                    else:
                        if len(attr_set)<min(case_num/2,100):
                            self.add_attr_names.append(attr_name)
                            self.feature_dict[attr_name] = get_category_feature(attr_set)
                            self.add_attr_num.append(len(attr_set))


    def _generate_data_raw(self, df):
        all_cases = np.unique(df['case:concept:name'])
        raw_data = [] # [[activity_seq][time_seq]...[]] 
        for case_id in all_cases:
            case_row = df[df['case:concept:name'] == case_id]

            activity_seq = case_row['concept:name'].to_list()
            time_seq = case_row['time:timestamp'].to_list()
            time_seq = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), time_seq))
            data_list = [activity_seq, time_seq]
            if self.is_multi_attr:
                attr_names = list(set(df.columns) - set(['case:concept:name', 'concept:name', 'time:timestamp']))
                attr_names.sort()  # sort the columns to ensure the order
                for attr_name in attr_names:
                    attr_seq = case_row[attr_name].to_list()
                    data_list.append(attr_seq)
            raw_data.append(data_list)
        return raw_data

    def generate_data_for_input(self, df, future_wz=1):
        all_cases = np.unique(df['case:concept:name'])
        input_data_list = [] # [[[activity_seq][time_seq]...[]]]
        future_activity_list = [] # [[activity_seq]]
        for case_id in all_cases:
            case_row = df[df['case:concept:name'] == case_id]

            activity_seq = case_row['concept:name'].to_list()
            activity_seq = list(map(lambda x: self.feature_dict['activity'][x], activity_seq))
            time_seq = case_row['time:timestamp'].to_list()
            time_seq = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), time_seq))
            if len(activity_seq) <2:
                raise ValueError("Invalid sequence length < 2")
            # pre-process
            feature_list = [activity_seq, time_seq]
            if self.is_multi_attr:
                for attr_name in self.add_attr_names:
                    attr_seq = case_row[attr_name].to_list()
                    if self.feature_dict[attr_name]['is_numeric']:
                        min_value = self.feature_dict[attr_name]['min']
                        max_value = self.feature_dict[attr_name]['max']
                        feature_seq = [(i-min_value)/(max_value-min_value) for i in attr_seq]
                    else:
                        mapping = self.feature_dict[attr_name]['mapping']  # category attr dict
                        feature_seq = [mapping[str(item)]if isinstance(item, float) and math.isnan(item) else mapping[item] for item in attr_seq]
                    feature_list.append(feature_seq)

            for i in range(1, len(activity_seq)):
                prefix_feature = []
                for feature in feature_list:
                    prefix_feature.append(feature[:i])
                input_data_list.append(prefix_feature)
                if (i+future_wz) >= len(activity_seq):
                    future_activity_list.append(activity_seq[i:])
                else:
                    future_activity_list.append(activity_seq[i:i+future_wz])
                
        return [input_data_list, future_activity_list]