from dataset.ap_dataset import APDataset
from utils.util import get_w_list_right_padding, get_time_feature
import numpy as np

class ESFDataset(APDataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(self, data_list, prediction_list, max_len, time_feature, activity_num, trace_dict):
        super(ESFDataset, self).__init__(data_list)

        self.max_len = max_len
        self.future_activity = prediction_list
        assert len(data_list) == len(prediction_list), "inconsistent number of samples"
        # self.future_wz = max([len(i) for i in prediction_list])
        self.time_feature = time_feature
        self.trace_dict = trace_dict
        self.activity_num = activity_num
        

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: (history sequence, next activity)
        """
        activity_seq, time_seq = self.data_list[idx][0], self.data_list[idx][1]
        time_case, time_event, time_day, time_week = get_time_feature(time_seq, self.time_feature)
        history_acyicity_seq = np.array(get_w_list_right_padding(activity_seq, self.max_len))
        
        history_time_case_seq = (np.array(get_w_list_right_padding(np.array(time_case), self.max_len)))
        history_time_event_seq = (np.array(get_w_list_right_padding(np.array(time_event), self.max_len)))
        history_time_day_seq = np.array(get_w_list_right_padding(time_day, self.max_len))
        history_time_week_seq = np.array(get_w_list_right_padding(time_week, self.max_len))

        history_seq = [history_acyicity_seq, history_time_case_seq, history_time_event_seq, history_time_day_seq, history_time_week_seq]
        if len(self.data_list[idx]) > 2:
            for add_feature_seq in self.data_list[idx][2:]:
                history_seq.append(np.array(get_w_list_right_padding(add_feature_seq, self.max_len)))
        history_seq = np.array(history_seq, dtype=np.float32)
        suffix_dict = self.trace_dict.get(tuple(activity_seq), None)
        candidates_freq_array = np.zeros((self.activity_num), dtype=np.float32)
        if suffix_dict:
            ids = np.array(list(suffix_dict.keys()))-1
            values = np.array(list(suffix_dict.values()))
            candidates_freq_array[ids] = values/np.sum(values)
        return history_seq,  np.array(self.future_activity[idx]), candidates_freq_array