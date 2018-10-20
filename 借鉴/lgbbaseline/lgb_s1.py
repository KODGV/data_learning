import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 基础配置信息
path = '../data/'
n_splits = 10
seed = 42

# lgb 参数
params={
    "learning_rate":0.1,
    "lambda_l1":0.1,
    "lambda_l2":0.2,
    "max_depth":4,
    "objective":"multiclass",
    "num_class":3,
    "silent":True,
}





input_dir="G:\DianXin\input\\"
laji_dir="G:\DianXin\input\\"
f=open(laji_dir+"SY1_CON.pkl",'rb')
train_data=pickle.load(f)
df_train = train_data[train_data['current_service']!=-1]
df_train=df_train.sample(frac=1)
df_test  = train_data[train_data['current_service']==-1]
df_test = df_test.drop(columns = ['current_service'], axis = 1)
X_train = df_train.drop(columns = ['current_service'], axis = 1)
y_train = df_train['current_service']
X_test=df_test
print('Input Matrix Dimension:  ', X_train.shape)
print('Output Vector Dimension: ', y_train.shape)
print('Test Data Dimension:     ', X_test.shape)

#预处理
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
le_sex.fit(y_train)
y=le_sex.transform(y_train)


X=np.abs(X_train.values)
X_test=np.abs(X_test.values)
print(y)

# 采取k折模型方案
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True

xx_score = []
cv_pred = []

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
for index,(train_index,test_index) in enumerate(skf.split(X,y)):

    X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf=lgb.train(params,train_data,num_boost_round=100000,valid_sets=[validation_data],early_stopping_rounds=50,feval=f1_score_vali,verbose_eval=1)

    xx_pred = clf.predict(X_valid,num_iteration=clf.best_iteration)

    xx_pred = [np.argmax(x) for x in xx_pred]

    xx_score.append(f1_score(y_valid,xx_pred,average='weighted'))

    y_test = clf.predict(X_test,num_iteration=clf.best_iteration)

    y_test = [np.argmax(x) for x in y_test]

    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

# 投票
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

# 保存结果
# 保存结果
df_test = pd.DataFrame()
submit=le_sex.inverse_transform(submit)
df_test['predict'] = submit
df_test.to_csv("SY1.csv")

print(xx_score,np.mean(xx_score))
