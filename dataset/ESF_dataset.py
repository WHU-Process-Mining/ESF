from dataset.ap_dataset import APDataset
from utils.util import get_w_list, get_time_feature
import numpy as np

class ESFDataset(APDataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(self, data_list, prediction_list, max_len, time_feature):
        super(ESFDataset, self).__init__(data_list)

        self.max_len = max_len
        self.future_activity = prediction_list
        assert len(data_list) == len(prediction_list), "inconsistent number of samples"
        self.future_wz = max([len(i) for i in prediction_list])
        self.max_case_interval = time_feature['max_case_interval']
        self.min_case_interval= time_feature['min_case_interval']
        self.max_event_interval = time_feature['max_event_interval']
        self.min_event_interval = time_feature['min_event_interval']
        

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: (history sequence, next activity)
        """
        activity_seq, time_seq = self.data_list[idx][0], self.data_list[idx][1]
        time_case, time_event, time_day, time_week = get_time_feature(time_seq)
        history_acyicity_seq = np.array(get_w_list(activity_seq, self.max_len))
        
        history_time_case_seq = np.array(time_case)/self.max_case_interval
        history_time_case_seq = (np.array(get_w_list(np.array(time_case)/self.max_case_interval, 
                                                     self.max_len)))
        history_time_event_seq = (np.array(get_w_list(np.array(time_event)/self.max_event_interval, 
                                                      self.max_len)))
        history_time_day_seq = np.array(get_w_list(time_day, self.max_len))
        history_time_week_seq = np.array(get_w_list(time_week, self.max_len))

        history_seq = [history_acyicity_seq, history_time_case_seq, history_time_event_seq, history_time_day_seq, history_time_week_seq]
        if len(self.data_list[idx]) > 2:
            for add_feature_seq in self.data_list[idx][2:]:
                history_seq.append(np.array(get_w_list(add_feature_seq, self.max_len)))
        history_seq = np.array(history_seq, dtype=np.float32)
        return history_seq,  np.array(get_w_list(self.future_activity[idx], self.future_wz))