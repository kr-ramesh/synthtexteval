import os
import pandas as pd
from tabulate import tabulate
from synthtexteval.utils.utils import evaluate_multilabel_classifier, evaluate_multiclass_classifier
from synthtexteval.eval.fairness.metrics import analyze_group_fairness_performance

def tabulate_results(csv_paths, n_labels, print_fairness=False, subgroup_type=None, problem_type = "multiclass"):
    """
    This function accepts multiple paths to CSV files, a boolean flag for printing fairness results, 
    and a subgroup type for fairness evaluation. It runs `evaluate_multilabel_classifier` to compute 
    the results, formats them using `tabulate`, and prints them.
    
    Args:
    - csv_paths (list of str): List of paths to CSV files.
    - print_fairness (bool): Flag to determine whether to print fairness results.
    - subgroup_type (str): Type of subgroup for fairness evaluation (e.g., "gender", "age").
    """
    
    all_results = []
    
    # Process each CSV file
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            if(problem_type == "multilabel"):
                ypred, ytrue = [ast.literal_eval(i) for i in df['Predicted'].tolist()], [ast.literal_eval(i) for i in df['Ground'].tolist()]
                results = evaluate_multilabel_classifier(ytrue, ypred)
            
            elif(problem_type == "multiclass"):
                ypred, ytrue = df['Predicted'].tolist(), df['Ground'].tolist()
                results = evaluate_multiclass_classifier(ytrue, ypred)
                
            if print_fairness:
                p_df, f_df = analyze_group_fairness_performance(df, problem_type = problem_type, num_classes = n_labels, subgroup_type = subgroup_type)
            
            # Add results to the all_results list
            # change all values in results to .3f 
            results = {k: f"{v:.3f}" for k, v in results.items()}
            if(print_fairness):
                f_df = f_df.to_dict(orient='records')[0]
                f_df = {k: f"{v:.2f}" for k, v in f_df.items() if k != 'Group Type'}
                results.update(f_df)
            all_results.append(results)
        else:
            print(f"Warning: {csv_path} does not exist.")
    
    # Neatly format the results with tabulate
    print("Evaluation Results:")
    formatted_results = tabulate(all_results, headers="keys", tablefmt="pretty")
    print(formatted_results)

    if(print_fairness):
        print("\nFairness Results:")
        print(tabulate(p_df, headers="keys", tablefmt="pretty"))

