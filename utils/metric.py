from sklearn import metrics 
import numpy as np
import pandas as pd
import os

def metric_calculate(truth_list, prediction_list):
    accuracy = metrics.accuracy_score(truth_list, prediction_list)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    truth_list, prediction_list, average="macro", zero_division=0)
    return accuracy, precision, recall, fscore

class EvaluationMetric():
    def __init__(self, save_file, max_case_length):
        self.save_file = save_file
        self.max_case_length = max_case_length
    
    def prefix_metric_calculate(self, truth_list, prediction_list, prefix_suf_variant_list):
#       Evaluate over all the prefixes (k) and save the results
        k, size, accuracies,fscores, precisions, recalls = [],[],[],[],[],[]
        idxs = []
        
        prefix_suffix_predict = {}
        prefix_suffix_label = {}

        for i in range(len(truth_list)):
            prefix_suffix_num = prefix_suf_variant_list[i]
            if prefix_suffix_num not in prefix_suffix_predict:
                prefix_suffix_predict[prefix_suffix_num] = [prediction_list[i]]
                prefix_suffix_label[prefix_suffix_num] = [truth_list[i]]
            else:
                prefix_suffix_predict[prefix_suffix_num].append(prediction_list[i])
                prefix_suffix_label[prefix_suffix_num].append(truth_list[i])
        
        for prefix_suffix_num, predictions in prefix_suffix_predict.items():
            grounds = prefix_suffix_label[prefix_suffix_num]
            accuracy, precision, recall, fscore = metric_calculate(grounds, predictions)
            k.append(prefix_suffix_num)
            size.append(len(predictions))
            accuracies.append(accuracy)
            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            print("prefix_var_num:{}, involved sample size:{}, accuracy:{}".format(prefix_suffix_num, len(predictions), accuracy))
        
        accuracy, precision, recall, fscore = metric_calculate(truth_list, prediction_list)
        
        k.append(max(prefix_suffix_predict.keys())+1)
        size.append(len(truth_list))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        
        print('Average accuracy across all prefixes:', accuracy)
        print('Average precision across all prefixes:', precision)
        print('Average recall across all prefixes:', recall)   
        print('Average f-score across all prefixes:', fscore) 
        
        if os.path.exists(self.save_file):
            df = pd.read_csv(self.save_file)
            old_acc = df.iloc[-1]['accuracy']
        else:
            old_acc = 0.0
        
        if accuracy>old_acc:
            results_df = pd.DataFrame({"suffix_var_num":k, "sample size":size, "accuracy":accuracies, 
                "precision":precisions, "recall":recalls,  "fscore": fscores,})
            results_df.sort_values(by="suffix_var_num", inplace=True)
            results_df.to_csv(self.save_file, index=False)
            return True
        else:
            return False