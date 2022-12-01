from sklearn import metrics
import torch
from siamese_bert_nojust import BertForSequenceClassification,text_dataset,DataLoader
import pandas as pd
import sys
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
#matplotlib
import matplotlib.pyplot as plt
def get_metrics(y_true, y_pred):
    """
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, precision, recall, f1
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

model = BertForSequenceClassification(num_labels=2)
model.load_state_dict(torch.load('checkpoints/bert_model_test_roberta_binary.pth'))
model.eval()

test_df = pd.read_csv('test.tsv', sep="\t", header=None)
test_df = test_df.fillna(0)
test = test_df.values


labels =  [test[i][1] for i in range(len(test))]
statements = [test[i][2] for i in range(len(test))]
subjects = [test[i][3] for i in range(len(test))]
speakers = [test[i][4] for i in range(len(test))]
jobs =  [test[i][5] for i in range(len(test))]
states = [test[i][6] for i in range(len(test))]
affiliations = [test[i][7] for i in range(len(test))]
credits = [test[i][8:13] for i in range(len(test))]
contexts = [test[i][13] for i in range(len(test))]
# print(labels[0])
# print(speakers[0])

num_labels = 2
if num_labels == 6:

    def to_onehot(a):
        a_cat = [0]*len(a)
        for i in range(len(a)):
            if a[i]=='true':
                a_cat[i] = [1,0,0,0,0,0]
            elif a[i]=='mostly-true':
                a_cat[i] = [0,1,0,0,0,0]
            elif a[i]=='half-true':
                a_cat[i] = [0,0,1,0,0,0]
            elif a[i]=='barely-true':
                a_cat[i] = [0,0,0,1,0,0]
            elif a[i]=='false':
                a_cat[i] = [0,0,0,0,1,0]
            elif a[i]=='pants-fire':
                a_cat[i] = [0,0,0,0,0,1]
            else:
                print('Incorrect label')
        return a_cat

elif num_labels == 2:

    def to_onehot(a):
        a_cat = [0]*len(a)
        for i in range(len(a)):
            if a[i]=='true':
                a_cat[i] = [1,0]
            elif a[i]=='mostly-true':
                a_cat[i] = [1,0]
            elif a[i]=='half-true':
                a_cat[i] = [1,0]
            elif a[i]=='barely-true':
                a_cat[i] = [0,1]
            elif a[i]=='false':
                a_cat[i] = [0,1]
            elif a[i]=='pants-fire':
                a_cat[i] = [0,1]
            else:
                print('Incorrect label')
        return a_cat

else:

    print('Invalid number of labels. The number of labels should be either 2 or 6')

    sys.exit()


labels_onehot = to_onehot(labels)
metadata = [0]*len(test)
credit_score = [0]*len(test)
for i in range(len(test)):
    subject = subjects[i]
    if subject == 0:
        subject = 'None'

    speaker = speakers[i]
    if speaker == 0:
        speaker = 'None'

    job = jobs[i]
    if job == 0:
        job = 'None'

    state = states[i]
    if state == 0:
        state = 'None'

    affiliation = affiliations[i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts[i]
    if context == 0 :
        context = 'None'
    
    meta = subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context

    metadata[i] = meta




for i in range(len(test)):
    credit = credits[i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score[i] = [score for i in range(1536)]

X_test = statements
y_test = labels_onehot
X_test_credit = credit_score
X_test_meta = metadata
test_lists = [X_test, X_test_meta, X_test_credit, y_test]
test_dataset = text_dataset(x_y_list = test_lists)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
dataset_size = len(X_test)


#


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
y_true = []
y_pred = []
for inputs,label in tqdm(test_dataloader):
    inputs1 = inputs[0] # News statement input
    # inputs2 = inputs[1] # Justification input
    inputs3 = inputs[1] # Meta data input
    inputs4 = inputs[2] # Credit scores input

    inputs1 = inputs1.to(device)
    # inputs2 = inputs2.to(device)
    inputs3 = inputs3.to(device)
    inputs4 = inputs4.to(device)
    label = label.to(device)
    outputs = model(inputs1, inputs3, inputs4)

    outputs = F.softmax(outputs,dim=1)
    _, preds = torch.max(outputs, 1)
    y_true.append(label.cpu().numpy())
    y_pred.append(preds.cpu().numpy())

y_true = np.concatenate(y_true, axis=0)
y_pred = np.concatenate(y_pred, axis=0)
confusion_matrix = metrics.confusion_matrix(y_true.argmax(axis=1), y_pred)
_,ax = plt.subplots(figsize=(10,10))
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center',size=20)
plt.xlabel('Predicted label',size=20)
plt.ylabel('True label',size=20)
plt.xticks([0,1],['True','False'],size=20)
plt.yticks([0,1],['True','False'],size=20)
plt.title('Confusion matrix',size=20)

plt.savefig('confusion_matrix.png',dpi=1200)
plt.show()


