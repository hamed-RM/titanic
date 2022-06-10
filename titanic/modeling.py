
def random_forest(x_train,y_train,x_test):
    random_forest=RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train)
    importances = pd.DataFrame({'feature': x_train.columns, 'importance': random_forest.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    print("rand",importances)
    y_pred=random_forest.predict(x_test)
    acc=random_forest.score(x_train, y_train)
    print(acc)
    return y_pred


def decision_tree(x_train,y_train,x_test,y_test):
    decision_tree=DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    importances=pd.DataFrame({'feature':x_train.columns,'importance':decision_tree.feature_importances_})
    importances=importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances)
    y_pred=decision_tree.predict(x_test)
    acc=decision_tree.score(x_test, y_test)
    acc_train=decision_tree.score(x_train, y_train)
    score=cross_val_score(decision_tree,x_train,y_train,cv=k_fold,n_jobs=1,scoring='accuracy')
    # print(np.mean(score))
    print('test : ', acc)
    print('train : ',acc_train)
    return y_pred


def sgd(x_train,y_train,x_test):
    sgd=SGDClassifier()
    sgd.fit(x_train,y_train)
    y_pred=sgd.predict(x_test)
    print(sgd.score(x_train, y_train))
    return y_pred


def svc(x_train,y_train,x_test):
    svc=SVC()
    svc.fit(x_train, y_train)
    y_pred=svc.predict(x_test)
    acc=svc.score(x_train, y_train)
    print(acc)
    return y_pred


def knn(x_train,y_train,x_test):
    knn=KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    yk_pred=knn.predict(x_test)
    acc=knn.score(x_train,y_train)
    print(acc)
    return yk_pred

def linear_svc(x_train,y_train,x_test):
    linear_svc = LinearSVC()
    linear_svc.fit(x_train, y_train)
    yl_pred = linear_svc.predict(x_test)
    acc =linear_svc.score(x_train, y_train)
    print(acc)
    return yl_pred


def perceptron(x_train,y_train,x_test):
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train)
    yp_pred = perceptron.predict(x_test)
    acc = perceptron.score(x_train, y_train)
    print(acc)
    return yp_pred


def naivebayes(x_train,y_train,x_test):
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    yb_pred = gaussian.predict(x_test)
    acc =gaussian.score(x_train, y_train)
    print(acc)
    return yb_pred

def logic_regression(x_train,y_train,x_test):
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    yreg_pred = logreg.predict(x_test)
    acc = logreg.score(x_train, y_train)
    print(acc)
    return yreg_pred

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



yd_pred=decision_tree(x_train,y_train,x_test,y_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yd_pred})
output.to_csv("result dec.csv",index=False)




yr_pred=random_forest(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yr_pred})
output.to_csv("result random.csv",index=False)


ys_pred=sgd(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ys_pred})
output.to_csv("result sgd.csv",index=False)



ysvc_pred=svc(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':ysvc_pred})
output.to_csv("result svc.csv",index=False)



yk_pred=knn(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yk_pred})
output.to_csv("result knn.csv",index=False)

yl_pred=linear_svc(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yl_pred})
output.to_csv("result linear svc.csv",index=False)

yp_pred=perceptron(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yp_pred})
output.to_csv("result perceptron.csv",index=False)

yb_pred=naivebayes(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yb_pred})
output.to_csv("result naive bayes.csv",index=False)

yreg_pred=logic_regression(x_train,y_train,x_test)
output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':yreg_pred})
output.to_csv("result log regression.csv",index=False)



print("\nDiff with")

temp=y_test-yd_pred
print('with dec :',temp[temp!=0].count()/temp.count())

temp=y_test-yr_pred
print('with ran :',temp[temp!=0].count()/temp.count())

temp=y_test-ys_pred
print('with sgd :',temp[temp!=0].count()/temp.count())

temp=y_test-ysvc_pred
print('with svc :',temp[temp!=0].count()/temp.count())

temp=y_test-yk_pred
print('with knn :',temp[temp!=0].count()/temp.count())

temp=y_test-yl_pred
print('with linear svc :',temp[temp!=0].count()/temp.count())

temp=y_test-yp_pred
print('with perceptron :',temp[temp!=0].count()/temp.count())

temp=y_test-yb_pred
print('with naive bayes :',temp[temp!=0].count()/temp.count())

temp=y_test-yreg_pred
print('with log regression :',temp[temp!=0].count()/temp.count())