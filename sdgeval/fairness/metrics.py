import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score
from sdgeval.utils.utils import evaluate_multilabel_classifier
from sklearn.metrics import f1_score

#TODO: Can add support for multiclass that is not binary later. Would need to look into prior research on this.
#TODO: For multiclass, follow the FairLearn approach and treat it as a composite of binary classification problems for now

def calculate_tpr_fpr(TP, FP, FN, TN):    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    TNR = TN / (TN + FP)
    return TPR, FPR, FNR, TNR

def group_metric(lst, v_group):
    result = sum([abs(v - v_group) for v in lst])
    return [result for i in range(len(lst))]

def groupwise_multiclass_metrics(group):
    y_true, y_pred = list(group['Ground']), list(group['Predicted'])

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    tp, fp = cm[0][0], cm[0][1]
    fn, tn = cm[1][0], cm[1][1]
    #print(cm)
    #tn = cm.diagonal().sum()
    #fp = cm.sum(axis=0) - cm.diagonal() 
    #fn = cm.sum(axis=1) - cm.diagonal()
    #tp = cm.sum() - tn - fp - fn

    result = {'Accuracy': accuracy, 'Confusion Matrix': cm,
              'Group Type': group.name, 'Num of Samples': len(group),}
                
    result['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    result['f1_micro'] = f1_score(y_true, y_pred, average='micro')

    result['TN'], result['FP'], result['FN'], result['TP'] = tn, fp.tolist(), fn.tolist(), tp
    return result

def groupwise_multilabel_metrics(group, mlb):
    
    try:
        y_true = [ast.literal_eval(item) for item in list(group['Ground'])]
        y_pred = [ast.literal_eval(item) for item in list(group['Predicted'])]
    except:
        y_true = list(group['Ground'])
        y_pred = list(group['Predicted'])

    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.fit_transform(y_pred)

    result = evaluate_multilabel_classifier(y_pred, y_true)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0].sum()
    fp = mcm[:, 0, 1].sum()
    fn = mcm[:, 1, 0].sum()
    tp = mcm[:, 1, 1].sum()
    result['Group Type'] = group.name  # Add ethnicity to the result dictionary
    result['Num of Samples'] = len(group)
    result['TN'], result['FP'], result['FN'], result['TP'] = tn, fp, fn, tp
    return result

def calculate_performance_metrics(df, subgroup_type, num_classes = 2, problem_type = "multilabel"):
    if(problem_type == "multilabel"):
        mlb = MultiLabelBinarizer(classes=range(num_classes))
        results = df.groupby(subgroup_type).apply(lambda group: groupwise_multilabel_metrics(group, mlb)).tolist()
        results_df = pd.DataFrame(results)
    else:
        results = df.groupby(subgroup_type).apply(lambda group: groupwise_multiclass_metrics(group)).tolist()
        results_df = pd.DataFrame(results)

    return results_df    

def calculate_fairness_metrics(df):

    #Read the subset results for each model type, then calculate the fairness results and save them to a new file for each subgroup
    true_negs, true_pos = [int(i) for i in df['TN'].tolist()], [int(i) for i in df['TP'].tolist()]
    false_negs, false_pos = [int(i) for i in df['FN'].tolist()], [int(i) for i in df['FP'].tolist()]
    f1_macro, f1_micro = [float(i) for i in df['f1_macro'].tolist()], [float(i) for i in df['f1_micro'].tolist()]

    tprs, fprs, fnrs, tnrs = [], [], [], []
    TPR_tot, FPR_tot, FNR_tot, TNR_tot = calculate_tpr_fpr(sum(true_pos), sum(false_pos), sum(false_negs), sum(true_negs))
    
    for i in range(len(df)):
        tpr, fpr, fnr, tnr = calculate_tpr_fpr(true_pos[i], false_pos[i], false_negs[i], true_negs[i])
        tprs.append(tpr)
        fprs.append(fpr)
        fnrs.append(fnr)
        tnrs.append(tnr)
    
    fned = group_metric(fnrs, FNR_tot)
    fped = group_metric(fprs, FPR_tot)
    tped = group_metric(tprs, TPR_tot)
    tned = group_metric(tnrs, TNR_tot)
    f1_macro_diff = [max(f1_macro)-min(f1_macro)]*len(df)
    f1_micro_diff = [max(f1_micro)-min(f1_micro)]*len(df)
    eq_ods = [max([max(tprs) - min(tprs), max(fprs) - min(fprs)])] * len(df)

    result_df = pd.DataFrame({})
    result_df['Group Type'] = df['Group Type']
    result_df['FNED'], result_df['FPED'], result_df['TPED'], result_df['TNED'] = fned, fped, tped, tned
    result_df['FNR'], result_df['TNR'], result_df['TPR'], result_df['FPR'] = fnrs, tnrs, tprs, fprs
    result_df['F1-micro-diff'], result_df['F1-macro-diff'] = f1_micro_diff, f1_macro_diff
    result_df['Equalized Odds'] = eq_ods
    
    return result_df

def analyze_group_fairness_performance(df, problem_type, num_classes):

    performance_df = calculate_performance_metrics(df, num_classes = num_classes, subgroup_type = "Subgroup", problem_type = problem_type)
    fairness_metric_df = calculate_fairness_metrics(performance_df)
    
    return performance_df, fairness_metric_df

#df = pd.read_csv("predictions.csv")
#p_df, f_df = analyze_group_fairness_performance(df)
#print(p_df)
#print(f_df)
