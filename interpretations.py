import argparse
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from xgboost import plot_importance
import os
import shap
import warnings
import csv

def evaluate_auc(clf,x_tr,x_te,y_tr,y_te,labels,weird_probas=None):
  preds_te = clf.predict_proba(x_te)
  preds_tr = clf.predict_proba(x_tr)
  list_auc_train= list()
  list_auc_test = list()
  mean_auc_train = 0
  mean_auc_test = 0
  for i in range(len(labels)):
    if weird_probas==True:
      p_tr = preds_tr[:,i]
      p_te = preds_te[:,i]
    else:
      p_tr = preds_tr[i][:,1]
      p_te = preds_te[i][:,1]
    auc_train = roc_auc_score(y_tr[labels[i]],p_tr)
    list_auc_train.append(auc_train)
    auc_test = roc_auc_score(y_te[labels[i]],p_te)
    list_auc_test.append(auc_test)
    mean_auc_train += auc_train
    mean_auc_test += auc_test
  return float(mean_auc_test/len(labels))

def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='jamendo',
                        choices=['jamendo', 'gtzan']
                        )
    parser.add_argument(
    '--interpretation', type=str, default='xgboost',
                    choices=['xgboost', 'ablation','shap','permutation']
                    )
    
    args = parser.parse_args()
    
    all = ['Melody','Articulation','Rhythm Complexity','Rhythm Stability', 'Dissonance', 'Atonality', 'Mode', 
    'Dominants', 'Subdominants', 'sub-sub', 'sub-dom', 'dom-sub', 'dom-tonic', 'glob-sub',  'glob-dom', 
    'sub-sub-dom', 'sub-dom-sub', 'dom-sub-dom', 'sub-dom-tonic', 'dom-tonic-sub', 
    'dom-sub-sub', 'sub-sub-sub', 'glob-sub-glob','glob-dom-tonic', 'glob-sub-sub', 'dom-dom', 'glob-glob',  'dom-dom-sub', 'glob-glob-dom', 'glob-dom-glob', 
    'glob-glob-sub',  'dom-dom-tonic', 'glob-sub-dom',  'dom-tonic-dom',  'glob-dom-sub', 'sub-dom-dom',  'dom-dom-dom','glob-dom-dom', 'glob-glob-glob',  'Danceability','Loudness','Chords-Changes-Rate','Dynamic-Complexity','Zerocrossingrate','Chords-Number-Rate'
    ,'Pitch-Salience','Spectral-Centroid','Spectral-Complexity','Spectral-Decrease','Spectral-Energyband-High',
    'Spectral-Energyband-Low','Spectral-Energyband-Middle-High','Spectral-Energyband-Middle-Low','Spectral-Entropy','Spectral-Flux','Spectral-Rolloff','Spectral-Spread','Onset-Rate','Length','BPM',
    'Beats-Loud']
    warnings.filterwarnings('ignore')
    if(args.interpretation == 'ablation'):
        #Midlevel (7)
        midlevel = ['Melody','Articulation','Rhythm Complexity','Rhythm Stability', 'Dissonance', 'Atonality', 'Mode']

        #Harmonic (32)
        harmonic = ['Dominants', 'Subdominants','sub-sub', 'sub-dom', 'dom-sub', 'dom-tonic', 'glob-sub', 'glob-dom', 'dom-dom', 'glob-glob','sub-sub-dom', 'sub-dom-sub', 'dom-sub-dom', 'sub-dom-tonic', 'dom-tonic-sub', 'dom-sub-sub', 'sub-sub-sub', 
        'glob-sub-glob', 'glob-dom-tonic', 'glob-sub-sub', 'dom-dom-sub', 'glob-glob-dom', 'glob-dom-glob', 'glob-glob-sub', 
        'dom-dom-tonic', 'glob-sub-dom', 'dom-tonic-dom', 'glob-dom-sub', 'sub-dom-dom', 'dom-dom-dom', 'glob-dom-dom', 'glob-glob-glob']

        # Signal only (21)
        signal = ['Danceability','Loudness','Chords-Changes-Rate','Dynamic-Complexity','Zerocrossingrate','Chords-Number-Rate'
        ,'Pitch-Salience','Spectral-Centroid','Spectral-Complexity','Spectral-Decrease','Spectral-Energyband-High',
        'Spectral-Energyband-Low','Spectral-Energyband-Middle-High','Spectral-Energyband-Middle-Low','Spectral-Entropy','Spectral-Flux',
        'Spectral-Rolloff','Spectral-Spread','Onset-Rate','Length','BPM','Beats-Loud']

        if(args.dataset == "jamendo"):
            pth_train = './jamendo/perceptual_features/train.csv'
            pth_test = './jamendo/perceptual_features/test.csv'
            pth_val = './jamendo/perceptual_features/validation.csv'

            df_train = pd.read_csv(pth_train)
            df_test = pd.read_csv(pth_test)
            df_val = pd.read_csv(pth_val)

            signal.append('Vocal-Instrumental')


            # Names
            names = ["Mid-level", "Harmonic", "Signal","Roc-Auc"]
            datas = list()
            for a in range(2):
                features = midlevel+harmonic+signal
                drop = ['Track']
                if(a==0):
                    features1 = [x for x in features if x not in midlevel]
                    drop1 = drop + midlevel
                else:
                    features1 = features
                    drop1 = drop
                for b in range(2):
                    if(b==0):
                        features2 = [x for x in features1 if x not in harmonic]
                        drop2 = drop1 + harmonic
                    else:
                        drop2 = drop1
                        features2 = features1
                    for c in range(2):
                        if (a==0 and b==0 and c==0):
                            datas.append([0,0,0,0])
                        else:
                            if(c==0):
                                features3 = [x for x in features2 if x not in signal]
                                drop3 = drop2 + signal
                            else:
                                drop3=drop2
                                features3 = features2

                            x_tr = df_train[features3]
                            y_tr = df_train.drop(drop3+features3,axis=1)
                            x_te = df_test[features3]
                            y_te = df_test.drop(drop3+features3,axis=1)
                            x_va = df_val[features3]
                            y_va = df_val.drop(drop3+features3,axis=1)
                            labels = list(y_tr.columns)

                            x_tr = pd.concat([x_tr,x_va])
                            y_tr = pd.concat([y_tr,y_va])
                            scaler = StandardScaler()
                            scaler.fit(x_tr)
                            x_tr=scaler.transform(x_tr)
                            x_te=scaler.transform(x_te)

                            xgb_estimator = xgb.XGBClassifier(max_depth=3,learning_rate = 0.1,objective='binary:logistic',eval_metric='auc', n_estimators = 70, gamma=7.5628225927223830,min_child_weight=6)
                            multilabel_model = MultiOutputClassifier(xgb_estimator)
                            multilabel_model.fit(x_tr, y_tr,verbose=True)
                            value = evaluate_auc(multilabel_model,x_tr,x_te,y_tr,y_te,labels)

                            datas.append([a,b,c,value])


            with open('./ablation.csv', 'w', encoding='UTF-8', newline='') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(names)

                # write multiple rows
                writer.writerows(datas)

        if(args.dataset == "gtzan"):
            pth = './gtzan/perceptual_features.csv'
            df = pd.read_csv(pth).drop(['Track'],axis=1)
            df.rename(columns = {'dom':'Dominants','sub':'Subdominants'}, inplace = True)
            x = df[all]
            y = df.drop(all,axis=1)
            labels = list(y.columns)

            # Names
            names = ["Mid-level", "Harmonic", "Signal","Accuracy"]
            datas = list()
            for a in range(2):
                features = midlevel+harmonic+signal
                drop = ['Track']
                if(a==0):
                    features1 = [x for x in features if x not in midlevel]
                    drop1 = drop + midlevel
                else:
                    features1 = features
                    drop1 = drop
                for b in range(2):
                    if(b==0):
                        features2 = [x for x in features1 if x not in harmonic]
                        drop2 = drop1 + harmonic
                    else:
                        drop2 = drop1
                        features2 = features1
                    for c in range(2):
                        if (a==0 and b==0 and c==0):
                            datas.append([0,0,0,0])
                            print(a,b,c)
                        else:
                            if(c==0):
                                features3 = [x for x in features2 if x not in signal]
                                drop3 = drop2 + signal
                            else:
                                drop3=drop2
                                features3 = features2

                            x_tr,x_te,y_tr,y_te = train_test_split(x,y,train_size=0.8, random_state=4)
                            x_tr = x_tr[features3]
                            x_te = x_te[features3]
                            labels = list(y_tr.columns)

                            scaler = StandardScaler()
                            scaler.fit(x_tr)
                            x_tr=scaler.transform(x_tr)
                            x_te=scaler.transform(x_te)

                            y_tra= list()
                            y_tr = y_tr.values.tolist()
                            for i in y_tr:
                                for index,j in enumerate(i):
                                    if j == 1:
                                        y_tra.append(index)
                                        break

                            y_te = y_te.values.tolist()
                            y_tes = list()
                            for i in y_te:
                                for index,j in enumerate(i):
                                    if j == 1:
                                        y_tes.append(index)
                                        break

                            xgb_estimator = xgb.XGBClassifier(max_depth=2,eta= 0.3,objective='multi:softmax',num_class=10,importance_type='weight')
                            xgb_estimator.fit(x_tr, y_tra,verbose=True)
                            value = xgb_estimator.score(x_te,y_tes)
                            datas.append([a,b,c,value])

            with open('./ablation.csv', 'w', encoding='UTF-8', newline='') as f:
                writer = csv.writer(f)

                # write the header
                writer.writerow(names)

                # write multiple rows
                writer.writerows(datas)

    else:
        if(args.dataset == "jamendo"):
            pth_train = './jamendo/perceptual_features/train.csv'
            pth_test = './jamendo/perceptual_features/test.csv'
            pth_val = './jamendo/perceptual_features/validation.csv'

            all.append('Vocal-Instrumental')

            df_train = pd.read_csv(pth_train).drop(['Track'],axis=1)
            df_test = pd.read_csv(pth_test).drop(['Track'],axis=1)
            df_val = pd.read_csv(pth_val).drop(['Track'],axis=1)

            x_tr = df_train[all]
            y_tr = df_train.drop(all,axis=1)
            x_te = df_test[all]
            y_te = df_test.drop(all,axis=1)
            x_va = df_val[all]
            y_va = df_val.drop(all,axis=1)
            labels = list(y_tr.columns)

            x_tr = pd.concat([x_tr,x_va])
            y_tr = pd.concat([y_tr,y_va])
    
            scaler = StandardScaler()
            scaler.fit(x_tr)
            x_tr=scaler.transform(x_tr)
            x_te=scaler.transform(x_te)

            xgb_estimator = xgb.XGBClassifier(max_depth=3,learning_rate = 0.1,objective='binary:logistic',eval_metric='auc', n_estimators = 70, gamma=7.5628225927223830,min_child_weight=6)

            multilabel_model = MultiOutputClassifier(xgb_estimator)

            multilabel_model.fit(x_tr, y_tr,verbose=True)


        elif(args.dataset == "gtzan"):

            pth = './gtzan/perceptual_features.csv'
            df = pd.read_csv(pth).drop(['Track'],axis=1)
            df.rename(columns = {'dom':'Dominants','sub':'Subdominants'}, inplace = True)

            x = df[all]
            y = df.drop(all,axis=1)
            labels = list(y.columns)

            x_tr,x_te,y_tr,y_te = train_test_split(x,y,train_size=0.8, random_state=4)

            scaler = StandardScaler()
            scaler.fit(x_tr)
            x_tr=scaler.transform(x_tr)
            x_te=scaler.transform(x_te)

            xgb_estimator = xgb.XGBClassifier(max_depth=2,learning_rate = 0.3)

            multilabel_model = MultiOutputClassifier(xgb_estimator)

            multilabel_model.fit(x_tr, y_tr,verbose=True)

        if(args.interpretation == 'xgboost'):

            directory = "xgboost feature importance"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.clf()
            for i in range(len(labels)):
                xg = multilabel_model.estimators_[i]
                xg.get_booster().feature_names = all
                print(labels[i])
                plt.rcParams['font.size'] = 10
                plot_importance(xg,title=labels[i], max_num_features=5)
                plt.xlabel("Xgboost Feature Importance")
                plt.savefig('./'+directory + '/'+ labels[i]+'.png', bbox_inches='tight', dpi=300)
                plt.close()


        if(args.interpretation == 'shap'):
            directory = "shap"
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(len(labels)):

                xg = multilabel_model.estimators_[i]
                xg.get_booster().feature_names = all
                explainer = shap.TreeExplainer(xg)
                shap_values = explainer.shap_values(x_te)
                print(labels[i])

                shap.summary_plot(shap_values,x_te,feature_names= all,max_display = 5,show=False)
                plt.savefig('./'+directory + '/'+ labels[i]+'.png', bbox_inches='tight', dpi=300)
                plt.close()

        if(args.interpretation == 'permutation'):
            perm_importance = permutation_importance(multilabel_model, x_tr, y_tr)

            res = dict(zip(all, perm_importance.importances_mean))
            new_dic = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

            keys = list(new_dic.keys())
            values = list(new_dic.values())
            fig, axes = plt.subplots(1,figsize=(15,15))
            axes.bar(keys, values, color ='maroon', width = 0.4)
            y_pos = range(len(all))
            axes.set_xticklabels(keys, rotation=270)
            plt.savefig('permutation_feature_importance.png')




if __name__ == "__main__":
    main()