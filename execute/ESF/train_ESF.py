import sys
sys.path.append('/home/inspur/zhengchao/ESF')

import torch
import os
import pandas as pd
from model.ESF.ESF import EnableStateFilterModel
from configs.config import load_config_data
from dataset.ESF_dataset import ESFDataset
from execute.ESF.train import setup_seed, train_model, analyse_suffix_variant
from utils.event_log import EventLogData, split_valid_df
from utils.util import generate_graph

if __name__ == "__main__":
    
    # load the model config
    cfg_model_train = load_config_data("configs/ESF_Model.yaml")
    
    setup_seed(cfg_model_train['seed'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_cfg = cfg_model_train['data_parameters']
    model_cfg = cfg_model_train['model_parameters']

    data_path = '{}/{}/time-process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results/{}/{}/'.format(model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(save_folder + f'/model', exist_ok=True)
    os.makedirs(save_folder + f'/curves', exist_ok=True)
        
    print("************* Training***************")

    train_file_name = data_path + 'train.csv'   
    test_file_name = data_path + 'test.csv'
    
    train_df = pd.read_csv(train_file_name)
    test_df = pd.read_csv(test_file_name)

    event_log = EventLogData(train_df, is_multi_attr=True)
    max_len = event_log.feature_dict['max_len']
    # spilit the dataset
    train_df, val_df = split_valid_df(train_df, dataset_cfg['valid_ratio'])
    train_data = event_log.generate_data_for_input(train_df)
    val_data = event_log.generate_data_for_input(val_df)
    
    model_cfg['activity_num'] = len(event_log.all_activities)
    model_cfg['add_attr_num'] = event_log.add_attr_num
    trace_dict = analyse_suffix_variant(train_data[0]+val_data[0], train_data[1]+val_data[1])
    train_dataset = ESFDataset(train_data[0], train_data[1], max_len, event_log.feature_dict['time'], model_cfg['activity_num'], trace_dict)
    val_dataset = ESFDataset(val_data[0], val_data[1], max_len, event_log.feature_dict['time'], model_cfg['activity_num'], trace_dict)

    model = EnableStateFilterModel(
            activity_num=model_cfg['activity_num'],
            dimension=model_cfg['dimension'],
            hidden_size_1=model_cfg['hidden_size'],
            hidden_size_2=model_cfg['hidden_size'],
            add_attr_num = model_cfg['add_attr_num'],
            dropout=model_cfg['dropout'],
            ).to(device)
    
    best_model_dict, best_val_accurace, train_loss_plt, train_accuracy_plt, val_accuracy_plt = train_model(train_dataset, val_dataset, model, model_cfg, device)

    # print the loss and accurace curve
    generate_graph(save_folder + f'/curves/curve.jpg', train_loss_plt, train_accuracy_plt, val_accuracy_plt)
    with open( f'{save_folder}/model/best_model.pth', 'wb') as fout:
        torch.save(best_model_dict, fout)