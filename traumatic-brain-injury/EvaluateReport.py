import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

filename = '/mnt/research/woldring_lab/Members/Needs/msm/traumatic-brain-injury/report.csv'
data = pd.read_csv(filename)

fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(data=data, x='Step', y='Accuracy_Train', label='Training data', ax=ax)
sns.lineplot(data=data, x='Step', y='Accuracy_Val', label='Validation data', ax=ax)
ax.set_title('Classification Accuracy, 4-mers, all data, LOOCV')
ax.set_xlabel('Model Step')
ax.set_ylabel('Mean Accuracy')
ax.legend()
 
plt.savefig('/mnt/research/woldring_lab/Members/Needs/msm/traumatic-brain-injury/AccuracyPlot.png')


fig2, ax2 = plt.subplots(figsize=(10,6))
sns.lineplot(data=data, x='Step', y='Cost_train', label='Training data', ax=ax2)
sns.lineplot(data=data, x='Step', y='Cost_Val', label='Validation data', ax=ax2)
ax2.set_title('Classification Accuracy, 4-mers, all data, LOOCV')
ax2.set_xlabel('Model Step')
ax2.set_ylabel('Mean Cost')
ax2.legend()
 
plt.savefig('/mnt/research/woldring_lab/Members/Needs/msm/traumatic-brain-injury/CostPlot.png')
