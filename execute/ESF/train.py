import torch
import random
import optuna
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from utils.metric import metric_calculate
from model.loss import ESFLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False # 
    torch.backends.cudnn.deterministic = True

def analyse_suffix_variant(prefix_list, next_activity_list):
    trace_dict = {}

    for prefix, next_activity in zip(prefix_list, next_activity_list):
        activity_prefix = prefix[0]
        if tuple(activity_prefix) not in trace_dict:
            trace_dict[tuple(activity_prefix)] = {}
        if next_activity not in trace_dict[tuple(activity_prefix)]:
            trace_dict[tuple(activity_prefix)][next_activity] =1
        else:
            trace_dict[tuple(activity_prefix)][next_activity] +=1
    
    print("Sample number:{}".format(len(prefix_list)))
    print("Prefix number:{}".format(len(trace_dict)))

    return trace_dict

# Test the test data(val data)
def test_model(test_dataset, model, model_parameters, device):
    predictions_list = []
    true_list = []
    var_num_list = []
    test_loss = 0
    criterion = ESFLoss(alpha=model_parameters['alpha'])
    test_dataloader = DataLoader(test_dataset, batch_size=model_parameters['batch_size'], shuffle=False)
    with torch.no_grad():
        model.eval()
        for seq, targets,candidates_freq in test_dataloader:
            batch_data = seq.to(device)
            logits = model(batch_data)
            _, _, total_loss =  criterion(logits, targets.to(device), candidates_freq.to(device))
            true_list.extend(targets.cpu().numpy().tolist())
            predictions_list.extend((torch.argmax(logits[1], dim=1).cpu().numpy()+1).tolist())
            var_num = torch.sum(candidates_freq > 0, dim=1)
            var_num_list.extend(var_num.cpu().numpy().tolist())
            test_loss += total_loss.item()
    
    return true_list, predictions_list, var_num_list, test_loss

def train_model(train_dataset, val_dataset, model, model_parameters, device, trial=None):
    print("************* Training Model ***************")
    
    train_dataloader = DataLoader(train_dataset, batch_size=model_parameters['batch_size'], shuffle=True)
    criterion = ESFLoss(alpha=model_parameters['alpha'])
    
    train_loss_plt = []
    train_accuracy_plt = []
    val_accuracy_plt = []
    
    best_val_accuracy = 0
    patience_count = 0
    max_patience_num = model_parameters['max_patience_num']
        
    optimizer = optim.AdamW(model.parameters(), lr=model_parameters['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    # Train Model
    for epoch in range(model_parameters['num_epochs']):
        model.train()
        predictions_list = [] 
        true_list = []
        training_loss = 0
        training_stg1_loss = 0
        training_stg2_loss = 0
        num_train = 0
        
        for seq, targets, candidates_freq in train_dataloader:
            batch_data = seq.to(device)
            candidates_freq_data = candidates_freq.to(device)
            logits = model(batch_data)
            stg_1_loss, stg_2_loss, total_loss =  criterion(logits, targets.to(device), candidates_freq_data)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            true_list.extend(targets.cpu().numpy().tolist())
            predictions_list.extend((torch.argmax(logits[1], dim=1).cpu().numpy()+1).tolist())
            
            num_train += 1
            training_loss += total_loss.item()
            training_stg1_loss += stg_1_loss.item()
            training_stg2_loss += stg_2_loss.item()
        
        train_loss_plt.append(training_loss/num_train)
        train_accurace,  _, _, train_fscore= metric_calculate(true_list, predictions_list)
        train_accuracy_plt.append(train_accurace)

        
        # test the accurace in val dataset
        val_truth_list, val_prediction_list, _, val_loss = test_model(val_dataset, model, model_parameters, device)
        val_accurace,  _, _, val_fscore= metric_calculate(val_truth_list, val_prediction_list)
        val_accuracy_plt.append(val_accurace)
        print(f"epoch: {epoch}, train_total_loss:{training_loss/num_train}, train_stage1_loss:{training_stg1_loss/num_train}, train_stage2_loss:{training_stg2_loss/num_train}, train_accurace:{train_accurace}, val_accurace:{val_accurace}, train_fscore:{train_fscore}, val_fscore:{val_fscore}")
        
        scheduler.step(val_loss)
        # Early Stop
        
        if epoch == 0 or val_accurace > best_val_accuracy:
            best_val_accuracy =  val_accurace
            patience_count = 0
            best_model_dict = deepcopy(model.state_dict())
        else:
            patience_count += 1
        
        if patience_count == max_patience_num:
            break
            
        if trial:
            # Report intermediate objective value.
            trial.report(best_val_accuracy, epoch)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
        
    print(f"best val_accuracy:{best_val_accuracy} ")
    return best_model_dict, best_val_accuracy, train_loss_plt, train_accuracy_plt, val_accuracy_plt