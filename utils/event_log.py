import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
import pandas as pd

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
    mean_value = np.mean(numerical_list)
    return {'is_numeric':True,
            'min': min_value,
            'max': max_value,
            'mean': mean_value}

def split_valid_data(train_input, valid_ratio):
        input_data, future_activity = train_input
        valid_n = int(valid_ratio * len(input_data))

        train_input, val_input, train_future_activity, val_future_activity = train_test_split(
            input_data, future_activity, test_size=valid_n, shuffle=True
        )
        return [train_input, train_future_activity], [val_input, val_future_activity]

def split_valid_df(df, valid_ratio):
    case_start_df = df.pivot_table(values='time:timestamp', index='case:concept:name', aggfunc='min').reset_index().sort_values(by='time:timestamp', ascending=True).reset_index(drop=True)
    ordered_id_list = list(case_start_df['case:concept:name'])

    # Get first validation case index
    first_val_case_id = int(len(ordered_id_list)*(1-valid_ratio))

    # Get lists of case ids to be assigned to val and train set
    val_case_ids = ordered_id_list[first_val_case_id:]
    train_case_ids = ordered_id_list[:first_val_case_id]

    # Final train-val split 
    train_set = df[df['case:concept:name'].isin(train_case_ids)].copy().reset_index(drop=True)
    val_set = df[df['case:concept:name'].isin(val_case_ids)].copy().reset_index(drop=True)

    return train_set, val_set


class EventLogData():
    def __init__(self, df, is_multi_attr = False):
        self.all_activities = np.unique(df['concept:name'])
        self.feature_dict = {}
        self.feature_dict['activity'] = dict(zip(self.all_activities, range(1, len(self.all_activities) + 1)))
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        time_list = df.groupby('case:concept:name')['time:timestamp'].apply(list).tolist()
        self.feature_dict['max_len'] = max([len(i) for i in time_list])-1 # max len of the prefix
        self.feature_dict['time'] = get_time_feature(time_list)
        self.is_multi_attr = is_multi_attr
        if self.is_multi_attr:
            self.add_attr_names = []
            self.add_attr_num = []
            self._get_att_feature(df)
        


    def _get_att_feature(self, df):
        # Get all the attributes except the case_id, activity and time
        case_num = len(np.unique(df['case:concept:name']))
        attr_names = list(set(df.columns) - set(['case:concept:name', 'concept:name', 'time:timestamp']))
        attr_names.sort()  # sort the columns to ensure the order
        # Transform every cat col in category dtype and impute MVs with dedicated level( Nan for numerical)
        for attr_name in attr_names:
            if (len(set(df[attr_name]))>1) and (df[attr_name].isnull().sum() < (0.5*len(df[attr_name]))): # ignore the attr with too many MVs
                if is_numeric_dtype(df[attr_name]):
                    self.add_attr_names.append(attr_name)
                    self.feature_dict[attr_name] = get_numerical_feature(df[attr_name].tolist())
                    df[attr_name] = df[attr_name].fillna(df[attr_name].mean())
                    self.add_attr_num.append(0)
                else:
                    if len(set(df[attr_name]))<min(case_num/2,100):
                        self.add_attr_names.append(attr_name)
                        df[attr_name]=  df[attr_name].fillna('Missing_MVs')
                        self.feature_dict[attr_name] = get_category_feature(set(df[attr_name]))
                        self.add_attr_num.append(len(set(df[attr_name])))


    def generate_data_for_input(self, df, future_wz=1):
        all_cases = np.unique(df['case:concept:name'])
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%Y-%m-%d %H:%M:%S', utc=True)
        input_data_list = [] # [[[activity_seq][time_seq]...[]]]
        future_activity_list = [] # [[activity_seq]]
        max_len = self.feature_dict['max_len']
        for case_id in all_cases:
            case_row = df[df['case:concept:name'] == case_id]
            case_row = case_row.sort_values(by=['time:timestamp'])
            if len(case_row)<=max_len:
                # Get the activity sequence and time sequence
                activity_map = self.feature_dict['activity']
                activity_seq = case_row['concept:name'].to_list()
                activity_seq = [activity_map.get(activity, len(activity_map) + 1) for activity in activity_seq]
                time_seq = case_row['time:timestamp'].to_list()
                if len(activity_seq) <2:
                    raise ValueError("Invalid sequence length < 2")
                feature_list = [activity_seq, time_seq]
                # add the additional attributes list
                if self.is_multi_attr:
                    for attr_name in self.add_attr_names:
                        if self.feature_dict[attr_name]['is_numeric']:
                            min_value = self.feature_dict[attr_name]['min']
                            max_value = self.feature_dict[attr_name]['max']
                            mean_value = self.feature_dict[attr_name]['mean']
                            attr_seq=  df[attr_name].fillna(mean_value).to_list()
                            feature_seq = [(i-min_value)/(max_value-min_value) for i in attr_seq]
                        else:
                            attr_seq=  df[attr_name].fillna('Missing_MVs').to_list()
                            mapping = self.feature_dict[attr_name]['mapping']  # category attr dict
                            feature_seq = [mapping.get(str(item), len(mapping) + 1) if isinstance(item, float) and math.isnan(item) else mapping.get(item, len(mapping) + 1) for item in attr_seq]
                        feature_list.append(feature_seq)

                is_valids = case_row['predictable'].to_list()
                for i in range(1, len(activity_seq)):
                    prefix_feature = []
                    if is_valids[i] == 1:
                        for feature in feature_list:
                            prefix_feature.append(feature[:i])
                        input_data_list.append(prefix_feature)
                        if (i+future_wz) >= len(activity_seq):
                            future_activity_list.append(activity_seq[i:])
                        else:
                            future_activity_list.append(activity_seq[i:i+future_wz])
                
        return [input_data_list, future_activity_list]