import sys
sys.path.append('/home/inspur/zhengchao/ESF')

import torch
import os
import pandas as pd
from utils.event_log import EventLogData
from configs.config import load_config_data
from execute.ESF.train import test_model
from dataset.ESF_dataset import ESFDataset
from utils.metric import EvaluationMetric

if __name__ == "__main__":
    
    cfg_model = load_config_data("configs/ESF_Model.yaml")
    dataset_cfg = cfg_model['data_parameters']
    model_cfg = cfg_model['model_parameters']

    data_path = '{}/{}/process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results_kfold_{}/{}/{}/'.format(dataset_cfg['k_fold_num'], model_cfg['model_name'], dataset_cfg['dataset'])

    os.makedirs(save_folder + f'/result', exist_ok=True)
    os.makedirs(f'{save_folder}/best_model', exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for idx in range(dataset_cfg['k_fold_num']):
        train_file_name = data_path + '/kfoldcv_' + str(idx) + '_train.csv'   
        test_file_name = data_path + '/kfoldcv_' + str(idx) + '_test.csv'
        
        train_df = pd.read_csv(train_file_name)
        test_df = pd.read_csv(test_file_name)

        event_log = EventLogData(pd.concat([train_df, test_df]), is_multi_attr=True)
        test_data = event_log.generate_data_for_input(test_df)
        
        model_cfg['activity_num'] = len(event_log.all_activities)
        model_cfg['add_attr_num'] = event_log.add_attr_num
        max_len = event_log.feature_dict['max_len']
        test_dataset = ESFDataset(test_data[0], test_data[1], max_len, event_log.feature_dict['time'])

        with open(f'{save_folder}/model/best_model_kfd{idx}.pth', 'rb') as fin:
            best_model = torch.load(fin).to(device)

        true_list, predictions_list, length_list = test_model(test_dataset, best_model, model_cfg, device)
        evaluator = EvaluationMetric(save_folder+"/result/k_fold_"+str(idx)+"_next_activity.csv", max_len)
        evaluator.prefix_metric_calculate(true_list, predictions_list, length_list)