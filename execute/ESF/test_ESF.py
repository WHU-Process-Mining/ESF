import sys
sys.path.append('/home/inspur/zhengchao/ESF')

import torch
import os
import re
import pandas as pd
from utils.event_log import EventLogData
from configs.config import load_config_data
from execute.ESF.train import test_model, analyse_suffix_variant
from dataset.ESF_dataset import ESFDataset
from utils.metric import EvaluationMetric
from model.ESF.ESF import EnableStateFilterModel

if __name__ == "__main__":
    
   cfg_model = load_config_data("configs/ESF_Model.yaml")
   dataset_cfg = cfg_model['data_parameters']
   model_cfg = cfg_model['model_parameters']

   data_path = '{}/{}/time-process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
   save_folder = 'results/{}/{}'.format(model_cfg['model_name'], dataset_cfg['dataset'])

   os.makedirs(save_folder + f'/result', exist_ok=True)
   os.makedirs(save_folder + f'/best_model', exist_ok=True)
   
   device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
   
   
   train_file_name = data_path + 'train.csv'   
   test_file_name = data_path + 'test.csv'
   

   train_df = pd.read_csv(train_file_name)
   test_df = pd.read_csv(test_file_name)

   event_log = EventLogData(train_df, is_multi_attr=True)
   train_data = event_log.generate_data_for_input(train_df)
   trace_dict = analyse_suffix_variant(train_data[0], train_data[1])

   model_cfg['activity_num'] = len(event_log.all_activities)
   model_cfg['add_attr_num'] = event_log.add_attr_num
   max_len = event_log.feature_dict['max_len']

   test_data = event_log.generate_data_for_input(test_df)
   test_dataset = ESFDataset(test_data[0], test_data[1], max_len, event_log.feature_dict['time'], model_cfg['activity_num'], trace_dict)
   
   with open(f'{save_folder}/model/best_model.txt', 'r') as fin:
      hyperparameters_str = fin.readlines()[2]
   
   hyperparameters_str = re.search(r"Best hyperparameters:\{(.*?)\}", hyperparameters_str, re.S).group(1)
   hyperparameters = eval(f"{{{hyperparameters_str}}}")
   
   model = EnableStateFilterModel(
         activity_num=model_cfg['activity_num'],
         hidden_size_1=hyperparameters['hidden_size_1'],
         hidden_size_2=hyperparameters['hidden_size_2'],
         add_attr_num = model_cfg['add_attr_num'],
         dropout=model_cfg['dropout'],
         threhold=hyperparameters['threhold'],
         ).to(device)
   
   with open(f'{save_folder}/model/best_model.pth', 'rb') as fin:
      best_model_dict = torch.load(fin)

   model.load_state_dict(best_model_dict)

   true_list, predictions_list, var_num_list, _ = test_model(test_dataset, model, model_cfg, device)
   evaluator = EvaluationMetric(save_folder+"/result/next_activity.csv", max_len)
   evaluator.prefix_metric_calculate(true_list, predictions_list, var_num_list)