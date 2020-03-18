import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from helper import *
from deepFM_model import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

train_file = '/home/xiao/DC竞赛/员工离职预测数据/train.csv'
test_file = '/home/xiao/DC竞赛/员工离职预测数据/test_noLabel.csv'
train_X = pd.read_csv(train_file)
test_X = pd.read_csv(test_file)
test_ID=np.array(test_X['ID'])
num_train, num_test = train_X.shape[0], test_X.shape[0]
train_Y = np.array(train_X['Label'])
train_X = train_X.drop('Label', axis=1)
X = pd.concat((train_X, test_X))
X = X.drop('ID', axis=1)
X['Age-Cut'] = X['Age'].apply(cut_age)
X = X.drop('Age', axis=1)
X['Department_cut'] = X['Department'].apply(cut_department).astype('int')
X = X.drop('Department', axis=1)
X = X.drop('EmployeeNumber', axis=1)
X['Dis_Home<10'] = X['DistanceFromHome'].apply(cut_dist).astype('int64')
X = X.drop('DistanceFromHome', axis=1)
X['Education'] = X['Education'].apply(cut_Edu)
X['EducationField'] = X['EducationField'].apply(cut_field).astype(int)
X['JobInvolvement'] = X['JobInvolvement'].apply(cut_jobInvolve)
X['JobLevel'] = X['JobLevel'].apply(cut_jobLevel)
X['JobRole'] = X['JobRole'].apply(cut_JobRole)
X['MonthlyIncome'] = X['MonthlyIncome'].apply(cut_monthlyIncome)
X = X.drop('NumCompaniesWorked', axis=1)
X = X.drop('Over18', axis=1)
X['PercentSalaryHike'] = X['PercentSalaryHike'].apply(cut_percentSalaryHike)
X['RelationshipSatisfaction'] = X['RelationshipSatisfaction'].apply(cut_relationshipSatisfaction)
X = X.drop('StandardHours', axis=1)
X['TotalWorkingYears'] = X['TotalWorkingYears'].apply(cut_totalWorkingYears)
X['TrainingTimesLastYear'] = X['TrainingTimesLastYear'].apply(cut_TrainingTimesLastYear)
X['WorkLifeBalance'] = X['WorkLifeBalance'].apply(cut_WorkLifeBalance)
X['YearsAtCompany'] = X['YearsAtCompany'].apply(cut_YearsAtCompany)
X['YearsInCurrentRole'] = X['YearsInCurrentRole'].apply(cut_YearsInCurrentRole)
X['YearsSinceLastPromotion'] = X['YearsSinceLastPromotion'].apply(cut_YearsSinceLastPromotion)
X['YearsWithCurrManager'] = X['YearsWithCurrManager'].apply(cut_YearsWithCurrManager)
for col in X.columns:
    enc = LabelEncoder()
    #print(X[col].unique().size)
    X[col]=enc.fit_transform(X[col])
X, field_size, feat_sizes = find_deep_params(np.array(X))
deep_model:nn.Module=DeepFM(field_size=field_size,feature_sizes=feat_sizes,
                  embedding_size=4)
cri=nn.BCELoss(reduction='sum')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deep_model=deep_model.to(device)
lr=3e-3
num_epoch=100
opt = torch.optim.Adam(lr=lr, params=deep_model.parameters())
num_eval=num_train//10
train_x,eval_x,test_x=X[:(num_train-num_eval)],X[(num_train-num_eval):num_eval],X[num_train:]
train_y,eval_y=train_Y[:(num_train-num_eval)],train_Y[(num_train-num_eval):num_eval]
train_x,test_x=torch.Tensor(train_x).long(),torch.Tensor(test_x).long()
train_y,eval_y=torch.Tensor(train_y),torch.Tensor(eval_y)
data_set=Data.TensorDataset(train_x,train_y)
data_loader=Data.DataLoader(dataset=data_set,batch_size=128,shuffle=True,num_workers=4)
tal_loses=[]
precises=[]
for epoch in range(num_epoch):
    tal_los=0
    for step,(batch_x,batch_y) in enumerate(data_loader):
        opt.zero_grad()
        batch_x,batch_y=batch_x.to(device),batch_y.to(device)
        outputs=deep_model(batch_x)
        loss=cri(outputs,batch_y)
        loss.backward()
        opt.step()
        tal_los+=loss.item()
    print('epoch:{},loss:{}'.format(epoch,tal_los))
    tal_loses.append(tal_los)
    if epoch%3==0:
        with torch.no_grad():
            pred_text = deep_model(train_x)
            #roc_score = roc_auc_score(train_y, pred_text)
            batch_res = (outputs > 0.16).float()
           # print(batch_res)
            precise=torch.sum(batch_y==batch_res).item()/batch_y.shape[0]
            precises.append(precise)
            if precise>0.88:
                break
fig=plt.figure()
ax1,ax2=fig.subplots(1,2,sharey=False)
ax1.plot(tal_loses,color='r',ls='--')
ax1.scatter(x=list(range(len(tal_loses))),y=tal_loses,color='b')
ax2.plot(precises, color='g', ls='--')
ax2.scatter(x=list(range(len(precises))), y=precises, color='r', marker='o')
plt.show()
deep_model.eval()
with torch.no_grad():
    outputs=deep_model(test_x)
print(outputs.tolist())
outputs=np.array(outputs)
print('训练集中正项比例:{}'.format(sum(train_y)/len(train_y)))
for val in [0.15,0.16,0.17,0.2,0.25]:
    res=(outputs>val).astype(int)
    print('当阀值为{}时，正项比例{}'.format(val,sum(res)/num_test))
    df=pd.DataFrame({'ID':test_ID,'Label':res})
    df.to_csv('{}res.csv'.format(val),index=False)
