import pandas as pd
import sklearn.metrics as metrics

file1 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/191125-0912/detailed_results.csv')
file2 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/191125-1424/detailed_results.csv')
file3 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/191125-1632/detailed_results.csv')
file4 = pd.read_csv('/home/knhung/UDPTDLTM/ClaimCheck/reports/191125-1956/detailed_results.csv')

result = pd.concat([file1, file2, file3, file4], ignore_index=True)
result = result.dropna(subset=['label', 'numeric_verdict'])
y_true = result['label'].tolist()
y_pred = result['numeric_verdict'].tolist()

class_report = metrics.classification_report(y_true, y_pred, digits=4)

print("Classification Report (qwen2.5:0.5b):\n", class_report)