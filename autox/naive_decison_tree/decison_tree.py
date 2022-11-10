from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt # data visualization
from sklearn import tree
import pandas as pd 
def train(X_train,y_train,X_test,y_test):
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
        # ,class_weight='balanced')
        # fit the model
        # class_weight='balanced'
        clf_gini.fit(X_train, y_train)
        y_pred_gini = clf_gini.predict(X_test)
        print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
        y_pred_train_gini = clf_gini.predict(X_train)
        print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
        print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))
        print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
        print(classification_report(y_test, y_pred_gini))
        plt.figure(figsize=(12,8))
        tree.plot_tree(clf_gini.fit(X_train, y_train))
        return clf_gini


def fit_and_train(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    # encode variables with ordinal encoding
    cols=X.columns.to_list()
    encoder = ce.OrdinalEncoder(cols=cols)
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)    
    clf_gini=train(X_train,y_train,X_test,y_test)
    return clf_gini
        
if __name__ == '__main__':
    df=pd.read_csv("/Users/liqiqi/Desktop/CDSS/胸痛cdss/CDSS_code/我们的数据/交付2021上半年.csv")
    y = df['门诊诊断_诊断主动脉夹层']+df['病案首页_诊断主动脉夹层']
    # 交付2021上半年.csv
    df_1 = df.drop(['姓名', '医保卡'], axis=1)
    for col in df_1.columns.to_list():
        if "诊断" in col :
            print(col)
            df_1.drop(col,axis=1,inplace=True)
    # df_1.drop('肺栓塞需复审=1；主动脉综合征需复审=2；其余=空白',axis=1,inplace=True)
    # df_1.drop('主动脉夹层',axis=1,inplace=True)
    # df_1.drop('dig',axis=1,inplace=True)
    X=df_1
    clf_gini=fit_and_train(X,y)
    import graphviz
    dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=X.columns, proportion=False)
    graph = graphviz.Source(dot_data) 
    print(graph)
