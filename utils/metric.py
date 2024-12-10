from sklearn import metrics 
import numpy as np
import pandas as pd
import os

def metric_calculate(truth_list, prediction_list):
    accuracy = metrics.accuracy_score(truth_list, prediction_list)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    truth_list, prediction_list, average="weighted", zero_division=0)
    return accuracy, precision, recall, fscore

class EvaluationMetric():
    def __init__(self, save_file, max_case_length):
        self.save_file = save_file
        self.max_case_length = max_case_length
    
    def prefix_metric_calculate(self, truth_list, prediction_list, length_list):
#       Evaluate over all the prefixes (k) and save the results
        k, size, accuracies,fscores, precisions, recalls = [],[],[],[],[],[]
        idxs = []
        
        for i in range(1,self.max_case_length+1):
            idx = np.where(np.array(length_list) == i)[0]
            if len(idx) > 0:
                accuracy, precision, recall, fscore = metric_calculate(np.array(truth_list)[idx], np.array(prediction_list)[idx])
                k.append(i)
                size.append(len(idx))
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)
                print("prefix size:{}, involved sample size:{}".format(i, len(idx)))
                idxs.extend(idx)
        
        accuracy, precision, recall, fscore = metric_calculate(np.array(truth_list)[idxs], np.array(prediction_list)[idxs])
        
        k.append(self.max_case_length+1)
        size.append(len(idxs))
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        
        print('Average accuracy across all prefixes:', accuracy)
        print('Average precision across all prefixes:', precision)
        print('Average recall across all prefixes:', recall)   
        print('Average f-score across all prefixes:', fscore)

        results_df = pd.DataFrame({"k":k, "sample size":size, "accuracy":accuracies, 
                "precision":precisions, "recall":recalls,  "fscore": fscores,})
        results_df.to_csv(self.save_file, index=False)