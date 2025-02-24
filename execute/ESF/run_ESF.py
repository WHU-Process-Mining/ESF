import sys
sys.path.append('/home/inspur/zhengchao/ESF')

import gc
import os
import time
import torch
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from dataset.ESF_dataset import ESFDataset
from model.ESF.ESF import EnableStateFilterModel
from utils.event_log import EventLogData, split_valid_df
from optuna.visualization import plot_optimization_history
from execute.ESF.train import setup_seed, train_model
from configs.config import load_config_data

def ESF_parameters(trial, cfg):
    # define the parameter search space
    model_parameters = {}
    
    model_parameters['dimension'] = trial.suggest_categorical('dimension', [16, 32, 64, 128])
    model_parameters['hidden_size_1'] = trial.suggest_categorical('hidden_size_1', [64, 128, 256, 512])
    model_parameters['hidden_size_2'] = trial.suggest_categorical('hidden_size_2', [64, 128, 256, 512])
    # model_parameters['threshold'] = trial.suggest_categorical('threshold', [0.1, 0.2, 0.3, 0.4, 0.5])
    model_parameters['threshold'] = trial.suggest_float('threshold', 0, 1)
    model_parameters['dropout'] = trial.suggest_float('dropout', 0, 1)
    model_parameters['alpha'] = trial.suggest_float('alpha', 1e1, 1e5, log=True)
    
    model_parameters['learning_rate'] = cfg['learning_rate']
    model_parameters['num_epochs'] = cfg['num_epochs']
    model_parameters['batch_size'] = cfg['batch_size']
    model_parameters['activity_num'] = cfg['activity_num']
    model_parameters['max_patience_num'] = cfg['max_patience_num']
    model_parameters['add_attr_num'] = cfg['add_attr_num']
    return model_parameters


def objective(trial, cfg_parameters, train_dataset, val_dataset, save_folder): 
    model_parameters = ESF_parameters(trial, cfg_parameters)
    device = 'cuda:'+ cfg_parameters['device_id'] if torch.cuda.is_available() else 'cpu'
    model = EnableStateFilterModel(
                activity_num=model_parameters['activity_num'],
                dimension=model_parameters['dimension'],
                hidden_size_1=model_parameters['hidden_size_1'],
                hidden_size_2=model_parameters['hidden_size_2'],
                add_attr_num=model_parameters['add_attr_num'],
                threshold=model_parameters['threshold'],
                dropout=model_parameters['dropout']).to(device)
    
    start_time = time.time()
    
    best_model, best_val_accurace, _, _, _ = train_model(train_dataset, val_dataset, model, model_parameters, device, trial)
    current_best = trial.study.best_value if trial.number > 0 else 0
    if best_val_accurace > current_best:
        with open( f'{save_folder}/model/best_model.pth', 'wb') as fout:
            torch.save(best_model, fout)

    duartime = time.time() - start_time
   
    record_file = open(f'{save_folder}/optimize/opt_history.txt', 'a')
    record_file.write(f"\n{trial.number},{best_val_accurace},{model_parameters['dimension']},{model_parameters['hidden_size_1']},{model_parameters['hidden_size_2']},{model_parameters['threshold']},{model_parameters['dropout']},{model_parameters['alpha']},{duartime}")
    record_file.close()
    
    return best_val_accurace

if __name__ == "__main__":
    
    
    cfg_model_train = load_config_data("configs/ESF_Model.yaml")
    # Fixed random number seed
    setup_seed(cfg_model_train['seed']) 
    
    dataset_cfg = cfg_model_train['data_parameters']
    model_cfg = cfg_model_train['model_parameters']
    model_cfg['device_id'] = '0'
    
    data_path = '{}/{}/time-process/'.format(dataset_cfg['data_path'], dataset_cfg['dataset'])
    save_folder = 'results/{}/{}/'.format(model_cfg['model_name'], dataset_cfg['dataset'])
    
    os.makedirs(f'{save_folder}/optimize', exist_ok=True)
    os.makedirs(f'{save_folder}/model', exist_ok=True)
    
    
    # record optimization
    record_file = open(f'{save_folder}/optimize/opt_history.txt', 'w')
    record_file.write("tid,score,dimension,hidden_size_1,hidden_size_2,threshold,dropout,alpha,duartime")
    record_file.close()
    
    train_file_name = data_path + 'train.csv'   
    test_file_name = data_path + 'test.csv'
    
    train_df = pd.read_csv(train_file_name)
    test_df = pd.read_csv(test_file_name)

    event_log = EventLogData(train_df, is_multi_attr=True)
    # spilit the dataset
    train_df, val_df = split_valid_df(train_df, dataset_cfg['valid_ratio'])
    train_data = event_log.generate_data_for_input(train_df, future_wz=model_cfg['future_window_size'])
    val_data = event_log.generate_data_for_input(val_df, future_wz=model_cfg['future_window_size'])
    
    model_cfg['activity_num'] = len(event_log.all_activities)
    model_cfg['add_attr_num'] = event_log.add_attr_num
    max_len = event_log.feature_dict['max_len']
    train_dataset = ESFDataset(train_data[0], train_data[1], max_len, event_log.feature_dict['time'])
    val_dataset = ESFDataset(val_data[0], val_data[1], max_len, event_log.feature_dict['time'])

    print(f"seed: {cfg_model_train['seed']}")
    print(f"dataset: {dataset_cfg['dataset']}, train size: {len(train_dataset)}, valid size:{len(val_dataset)}")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg_model_train['seed'])) # fixed parameter
    study.optimize(lambda trial: objective(trial, model_cfg, train_dataset, val_dataset, save_folder), n_trials=30, gc_after_trial=True, callbacks=[lambda study, trial:gc.collect()])
    
    # record optimization history
    history = optuna.visualization.plot_optimization_history(study)
    plot_optimization_history(study).write_image(f"{save_folder}/optimize/opt_history.png")
    
    outfile = open(f'{save_folder}/model/best_model.txt', 'w')
    best_params = study.best_params
    best_accurace = study.best_value

    print("Best hyperparameters:", best_params)
    print("Best accurace:", best_accurace)
    
    outfile.write('Random Seed:' + str(cfg_model_train['seed']))
    outfile.write('\nBest trail:' + str(study.best_trial.number))
    outfile.write('\nBest hyperparameters:' + str(best_params))
    outfile.write('\nBest accurace:' + str(best_accurace))
    outfile.close()