import pandas as pd
import os

performance_metric_dir = '/mnt/d/Project_1_SDNet/Results/final_model_classification_loss_1_performance/inference/Performance_Metrics'
save_dir = '/mnt/d/Project_1_SDNet/Results/final_model_classification_loss_1_performance/inference/performance_summary.csv'

performance_metrics_list = ['ACC_mean.csv',
                       'AFDE_mean.csv',
                       'Fixel_Accuracy.csv',
                       'PAE_mean.csv',
                       'SSE_mean.csv']

results_df = pd.DataFrame(columns = ['Metric', 'Mean', 'Std Error'])

for performance_metric_str in performance_metrics_list:
    df = pd.read_csv(os.path.join(performance_metric_dir, performance_metric_str))
    mean_value = df['wm'].mean()
    std_error_value = df['wm'].sem()

    results_df = results_df.append({'Metric': performance_metric_str,
                                    'Mean': mean_value,
                                    'Std Error': std_error_value}, ignore_index = True)
    
    def format_mean_and_std_error(col):
        return f"{col['Mean']:.4f}±{col['Std Error']:.4f}"
    
    
results_df['Mean ± Std Error'] = results_df.apply(format_mean_and_std_error, axis = 1)
results_df.T.to_csv(save_dir, index=False, encoding = 'utf-8')