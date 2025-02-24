import sys
sys.path.append('/home/inspur/zhengchao/ESF')

import torch
import shutil
import os
import re
import pandas as pd
from utils.event_log import EventLogData
from configs.config import load_config_data
from execute.ESF.train import test_model
from dataset.ESF_dataset import ESFDataset
from utils.metric import EvaluationMetric
from model.ESF.ESF import EnableStateFilterModel

def analyse_suffix_variant(input_data):
    trace_dict = {}
    history_list, future_activity_list = input_data
    for i, prefix in enumerate(history_list):
        activity_prefix = prefix[0]
        next_activity = future_activity_list[i][0]
        if tuple(activity_prefix) not in trace_dict:
            trace_dict[tuple(activity_prefix)] = {}
        if next_activity not in trace_dict[tuple(activity_prefix)]:
            trace_dict[tuple(activity_prefix)][next_activity] =1
        else:
            trace_dict[tuple(activity_prefix)][next_activity] +=1
    
    print("Sample number:{}".format(len(history_list)))
    print("Prefix number:{}".format(len(trace_dict)))
    
    prefix_suffix_dict = {}
    for prefix, next_activity_var in trace_dict.items():
        suf_var_num = len(next_activity_var)
        prefix_suffix_dict[prefix] = suf_var_num

    return prefix_suffix_dict

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
    prefix_suffix_dict= analyse_suffix_variant(event_log.generate_data_for_input(train_df))

    test_data = event_log.generate_data_for_input(test_df)

    prefix_suf_variant_list = []
    prefix_list = []
    suffix_list = []
    for prefix, suffix in zip(test_data[0], test_data[1]):
        activity_prefix = prefix[0]
        next_activity = suffix[0]
        if next_activity <= len(event_log.all_activities):
            prefix_suffix_num = prefix_suffix_dict.get(tuple(activity_prefix), 0)
            prefix_suf_variant_list.append(prefix_suffix_num)
            prefix_list.append(prefix)
            suffix_list.append(suffix)
    
    model_cfg['activity_num'] = len(event_log.all_activities)
    model_cfg['add_attr_num'] = event_log.add_attr_num
    max_len = event_log.feature_dict['max_len']
    test_dataset = ESFDataset(prefix_list, suffix_list, max_len, event_log.feature_dict['time'])
    with open(f'{save_folder}/model/best_model.txt', 'r') as fin:
       hyperparameters_str = fin.readlines()[2]
    
    hyperparameters_str = re.search(r"Best hyperparameters:\{(.*?)\}", hyperparameters_str, re.S).group(1)
    hyperparameters = eval(f"{{{hyperparameters_str}}}")
    
    model = EnableStateFilterModel(
            activity_num=model_cfg['activity_num'],
            dimension=hyperparameters['dimension'],
            hidden_size_1=hyperparameters['hidden_size_1'],
            hidden_size_2=hyperparameters['hidden_size_2'],
            add_attr_num = model_cfg['add_attr_num'],
            threshold=hyperparameters['threshold'],
            dropout=hyperparameters['dropout'],
            ).to(device)
    
    with open(f'{save_folder}/model/best_model.pth', 'rb') as fin:
       best_model_dict = torch.load(fin)

    model.load_state_dict(best_model_dict)

    true_list, predictions_list, length_list = test_model(test_dataset, model, model_cfg, device)
    evaluator = EvaluationMetric(save_folder+"/result/next_activity.csv", max_len)
    is_well = evaluator.prefix_metric_calculate(true_list, predictions_list, prefix_suf_variant_list)
    if is_well:
        # 复制文件
        shutil.copyfile(f'{save_folder}/model/best_model.pth', 
                        f'{save_folder}/best_model/best_model.pth')
        shutil.copyfile(f'{save_folder}/model/best_model.txt',
                        f'{save_folder}/best_model/best_model.txt')