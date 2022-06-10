def random_forest(x_train,y_train,x_test,y_test):
    random_forest=RandomForestClassifier(n_estimators=100,random_state=99,max_depth=10)
    random_forest.fit(x_train, y_train)
    importances = pd.DataFrame({'feature': x_train.columns, 'importance': random_forest.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    # print("random forest importance: \n",importances)
    y_pred=random_forest.predict(x_test)
    acc=random_forest.score(x_train, y_train)
    acc_test=random_forest.score(x_test,y_test)
    print('random forest test: ', acc_test)
    print('random forest train: ', acc)
    print()
    return y_pred


def decision_tree(x_train,y_train,x_test,y_test):
    decision_tree=DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    importances=pd.DataFrame({'feature':x_train.columns,'importance':decision_tree.feature_importances_})
    importances=importances.sort_values('importance',ascending=False).set_index('feature')
    # print('decision tree importance: \n',importances)
    y_pred=decision_tree.predict(x_test)
    acc=decision_tree.score(x_test, y_test)
    acc_train=decision_tree.score(x_train, y_train)
    # print(np.mean(score))
    print('decision tree test: ', acc)
    print('decision tree train: ',acc_train)
    print()
    return y_pred


def sgd(x_train,y_train,x_test,y_test):
    sgd=SGDClassifier()
    sgd.fit(x_train,y_train)
    y_pred=sgd.predict(x_test)
    acc_train=sgd.score(x_train, y_train)
    acc_test=sgd.score(x_test,y_test)
    print('sgd test: ',acc_test)
    print('sgd train: ',acc_train)
    print()
    return y_pred


def svc(x_train,y_train,x_test,y_test):
    svc=SVC()
    svc.fit(x_train, y_train)
    y_pred=svc.predict(x_test)
    acc_train=svc.score(x_train, y_train)
    acc_test=svc.score(x_test, y_test)
    print('svc test: ',acc_test)
    print('svc train: ',acc_train)
    print()
    return y_pred


def knn(x_train,y_train,x_test,y_test):
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    yk_pred=knn.predict(x_test)
    acc_train=knn.score(x_train,y_train)
    acc_test=knn.score(x_test,y_test)
    print('knn test: ',acc_test)
    print('knn train: ',acc_train)
    print()
    return yk_pred

def linear_svc(x_train,y_train,x_test,y_test):
    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    yl_pred = linear_svc.predict(x_test)
    acc_train =linear_svc.score(x_train, y_train)
    acc_test =linear_svc.score(x_test, y_test)
    print('linear svc test: ',acc_test)
    print('linear svc train: ',acc_train)
    print()
    return yl_pred


def perceptron(x_train,y_train,x_test,y_test):
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    yp_pred = perceptron.predict(x_test)
    acc_train = perceptron.score(x_train, y_train)
    acc_test = perceptron.score(x_test, y_test)
    print('perceptron test: ',acc_test)
    print('perceptron train: ',acc_train)
    print()
    return yp_pred


def naivebayes(x_train,y_train,x_test,y_test):
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    yb_pred = gaussian.predict(x_test)
    acc_train =gaussian.score(x_train, y_train)
    acc_test =gaussian.score(x_test, y_test)
    print('naive test: ',acc_test)
    print('naive train: ',acc_train)
    print()
    return yb_pred

def logic_regression(x_train,y_train,x_test,y_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    yreg_pred = logreg.predict(x_test)
    acc_train = logreg.score(x_train, y_train)
    acc_test = logreg.score(x_test, y_test)

    print('logic reg test: ',acc_test)
    print('logic reg train: ',acc_train)
    print()
    return yreg_pred


def diff():
    print("\nDiff with")

    temp = y_test - yd_pred
    print('with dec :', temp[temp != 0].count() / temp.count())

    temp = y_test - yr_pred
    print('with ran :', temp[temp != 0].count() / temp.count())

    temp = y_test - ys_pred
    print('with sgd :', temp[temp != 0].count() / temp.count())

    temp = y_test - ysvc_pred
    print('with svc :', temp[temp != 0].count() / temp.count())

    temp = y_test - yk_pred
    print('with knn :', temp[temp != 0].count() / temp.count())

    temp = y_test - yl_pred
    print('with linear svc :', temp[temp != 0].count() / temp.count())

    temp = y_test - yp_pred
    print('with perceptron :', temp[temp != 0].count() / temp.count())

    temp = y_test - yb_pred
    print('with naive bayes :', temp[temp != 0].count() / temp.count())

    temp = y_test - yreg_pred
    print('with log regression :', temp[temp != 0].count() / temp.count())


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def save(predict,file_name):
    output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predict})
    output.to_csv(file_name+".csv",index=False)


def chart(feature):
    survivied=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survivied,dead])
    df.index=['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    plt.suptitle(feature)
    plt.show()


def chart_continus(feature):
    facet=sns.FacetGrid(train,hue='Survived',aspect=4)
    facet.map(sns.kdeplot,feature,shade=True)
    facet.set(xlim=(0,train[feature].max()))
    facet.add_legend()
    plt.suptitle(feature)
    plt.show()


import pandas as pd
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
yrand=pd.read_csv('result random.csv')
combine=[train,test]
# print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by=['Survived'],ascending=False))
# chart('Sex')
train=train.drop(["Ticket"],axis=1)
test=test.drop(["Ticket"],axis=1)
# chart('Pclass')
# print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False))
combine=[train,test]
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
# print(train['Title'].value_counts())

for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Countess','Capt','Col',\
                                               'Don','Dr','Major','Rev','Sir','Jonkheer','Dona','Master'],'Rare')

    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')

# chart('Title')
# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False))

    # dataset['Title'] = dataset['Title'].replace(['Col','Dr', 'Major','Master'], 'Half')
    # dataset['Title'] = dataset['Title'].replace(['Capt','Don','Rev','Jonkheer'], 'Zero')
    # dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Sir','Mlle','Ms','Mme'], 'One')

title_mapping={"Mr":0,"Miss":1,"Mrs":2,'Rare':3}
# 11
# title_mapping={"Zero":0,"Half":1,"One":2,'Mr':3,'Mrs':4,"Miss":5}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(dataset['Title'].dropna().mode()[0])
# print(train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=False))

train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
combine=[train,test]
for dataset in combine:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)

# print(train[['Age','Survived']].groupby(['Age'],as_index=False).mean().sort_values(by='Age',ascending=False))

guess_ages=np.zeros((2,3))

for dataset in combine:
    for i in range(0,2):
        for j in  range(0,3):
            guess_df=dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()

            age_guess=guess_df.mean()

            guess_ages[i,j]=int(age_guess)
    for i in range(0,2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull())&(dataset['Sex']==i)&(dataset['Pclass']==j+1),'Age']=guess_ages[i,j]

    dataset['Age']=dataset['Age'].astype(int)


for dataset in combine:
    dataset.loc[dataset['Age']<=20,'Age']=0
    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=30),'Age']=1
    dataset.loc[(dataset['Age']>30)&(dataset['Age']<=45),'Age']=2
    dataset.loc[(dataset['Age']>45)&(dataset['Age']<=65),'Age']=3
    dataset.loc[dataset['Age']>65,'Age']=4

# print(train[['Age','Survived']].groupby(['Age'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# chart('Age')
for dataset in combine:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
    dataset['FamilySize']=dataset['FamilySize'].astype(int)
# print(train['FamilySize'].value_counts())
# print(train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False))
for dataset in combine:
    dataset.loc[dataset['FamilySize'].isin([8,11]),'FamilySize']=0
    dataset.loc[dataset['FamilySize'].isin([5,6]),'FamilySize']=-1
    dataset.loc[dataset['FamilySize'].isin([1,7]),'FamilySize']=-2
    dataset.loc[dataset['FamilySize'].isin([2,3]),'FamilySize']=-3
    dataset.loc[dataset['FamilySize']==4,'FamilySize']=-4
    dataset['FamilySize']=dataset['FamilySize'].astype(int)

for dataset in combine:
    dataset['FamilySize']=abs(dataset['FamilySize'])
# print(train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# chart('FamilySize')
# print(train['FamilySize'].value_counts())
train=train.drop(['Parch','SibSp'],axis=1)#'FamilySize'
test=test.drop(['Parch','SibSp'],axis=1)#'FamilySize'
combine=[train,test]


for dataset in combine:
    freq_port=dataset['Embarked'].dropna().mode()[0]
    dataset['Embarked']=dataset['Embarked'].fillna(freq_port)

# print(train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# chart('Embarked')
for dataset in combine:
    dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

guss_fare=np.zeros((3,6))
for dataset in combine:
    for i in range(0,3):
        for j in range(0,6):
            datafram=dataset[(dataset['Pclass']==i+1) & (dataset['Age']==j)]['Fare'].dropna()
            if(datafram.empty):
                guss_fare[i,j]=int(dataset[dataset['Pclass']==i+1]['Fare'].mean())
            else:
                guss_fare[i,j] = int(datafram.mean())

    for i in range(0,3):
        for j in range(0,6):
            dataset.loc[(dataset.Fare.isnull()) & (dataset['Pclass'] == i+1) & (dataset['Age'] == j), 'Fare'] =guss_fare[i,j]


for dataset in combine:
    dataset.loc[dataset['Fare']<=15,'Fare']=0
    dataset.loc[(dataset['Fare']>15)&(dataset['Fare']<=25),'Fare']=1
    dataset.loc[(dataset['Fare']>25)&(dataset['Fare']<=75),'Fare']=2
    dataset.loc[(dataset['Fare']>75)&(dataset['Fare']<=100),'Fare']=3
    dataset.loc[dataset['Fare']>100,'Fare']=4
    dataset['Fare']=dataset['Fare'].astype(int)

# print(train[['Fare','Survived']].groupby(['Fare'],as_index=False).mean().sort_values(by='Survived',ascending=False))
# chart('Fare')
x_train=train.drop(['Survived','Cabin'],axis=1)
y_train=train['Survived']
x_test=test.drop(['PassengerId','Cabin'],axis=1)

y_test=pd.read_csv('result f.csv')
y_test=y_test['Survived']




yd_pred=decision_tree(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yd_pred})
output.to_csv("result dec.csv",index=False)




yr_pred=random_forest(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yr_pred})
output.to_csv("result random.csv",index=False)


ys_pred=sgd(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ys_pred})
output.to_csv("result sgd.csv",index=False)



ysvc_pred=svc(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ysvc_pred})
output.to_csv("result svc.csv",index=False)



yk_pred=knn(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yk_pred})
output.to_csv("result knn.csv",index=False)

yl_pred=linear_svc(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yl_pred})
output.to_csv("result linear svc.csv",index=False)

yp_pred=perceptron(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yp_pred})
output.to_csv("result perceptron.csv",index=False)

yb_pred=naivebayes(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yb_pred})
output.to_csv("result naive bayes.csv",index=False)

yreg_pred=logic_regression(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yreg_pred})
output.to_csv("result log regression.csv",index=False)


# diff()
