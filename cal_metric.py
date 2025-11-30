import pandas as pd
import sklearn.metrics as metrics

file1 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/201125-0658/detailed_results.csv')
file2 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/201125-1106/detailed_results.csv')
file3 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/201125-1443/detailed_results.csv')
file4 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/201125-1603/detailed_results.csv')

result = pd.concat([file1, file2, file3, file4], ignore_index=True)
result = result.dropna(subset=['expected_label', 'predicted_label'])
y_true = result['expected_label'].tolist()
y_pred = result['predicted_label'].tolist()

class_report = metrics.classification_report(y_true, y_pred, digits=4)

print("Classification Report (qwen2.5:0.5b-instruct):\n", class_report)