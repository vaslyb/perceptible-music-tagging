import argparse
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def evaluate_auc(clf,x_te,x_tr,y_tr,y_te,labels,weird_probas=None):
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

  print("MEAN")
  print("AUC train : "+str(mean_auc_train/len(labels)))
  print("AUC test : "+str(mean_auc_test/len(labels)))


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='jamendo',
                        choices=['jamendo', 'gtzan','magnatagatune'],
                        )

    args = parser.parse_args()
    music_tags = [
        "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", 
        "rock", "fast", "piano", "ambient", "beat", "violin", "vocal", "synth", 
        "female", "indian", "opera", "male", "singing", "vocals", "no vocals", 
        "harpsichord", "loud", "quiet", "flute", "woman", "male vocal", "no vocal", 
        "pop", "soft", "sitar", "solo", "man", "classic", "choir", "voice", 
        "new age", "dance", "male voice", "female vocal", "beats", "harp", "cello", 
        "no voice", "weird", "country", "metal", "female voice", "choral"
    ]


    all = ['Melody','Articulation','Rhythm Complexity','Rhythm Stability', 'Dissonance', 'Atonality', 'Mode', 
    'Dominants', 'Subdominants', 'sub-sub', 'sub-dom', 'dom-sub', 'dom-tonic', 'glob-sub',  'glob-dom', 
    'sub-sub-dom', 'sub-dom-sub', 'dom-sub-dom', 'sub-dom-tonic', 'dom-tonic-sub', 
    'dom-sub-sub', 'sub-sub-sub', 'glob-sub-glob','glob-dom-tonic', 'glob-sub-sub', 'dom-dom', 'glob-glob',  'dom-dom-sub', 'glob-glob-dom', 'glob-dom-glob', 
    'glob-glob-sub',  'dom-dom-tonic', 'glob-sub-dom',  'dom-tonic-dom',  'glob-dom-sub', 'sub-dom-dom',  'dom-dom-dom','glob-dom-dom', 'glob-glob-glob',  'Danceability','Loudness','Chords-Changes-Rate','Dynamic-Complexity','Zerocrossingrate','Chords-Number-Rate'
    ,'Pitch-Salience','Spectral-Centroid','Spectral-Complexity','Spectral-Decrease','Spectral-Energyband-High',
    'Spectral-Energyband-Low','Spectral-Energyband-Middle-High','Spectral-Energyband-Middle-Low','Spectral-Entropy','Spectral-Flux','Spectral-Rolloff','Spectral-Spread','Onset-Rate','Length','BPM',
    'Beats-Loud']

    if(args.dataset == "jamendo" or args.dataset == "magnatagatune"):
        if (args.dataset == "jamendo"):
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
        else:
          all.remove('Length')
          pth = './magnatagatune/perceptual_features.csv'
          df = pd.read_csv(pth).drop(['Track'],axis=1)
          df.rename(columns = {'dom':'Dominants','sub':'Subdominants'}, inplace = True)
          x = df[all]
          y = df.drop(all,axis=1)
          y = y[music_tags]
          labels = list(y.columns)
          x_tr,x_te,y_tr,y_te = train_test_split(x,y,train_size=0.8, random_state=4)
 
          scaler = StandardScaler()
          scaler.fit(x_tr)
          x_tr=scaler.transform(x_tr)
          x_te=scaler.transform(x_te)

        xgb_estimator = xgb.XGBClassifier(max_depth=3,learning_rate = 0.1,objective='binary:logistic',eval_metric='auc', n_estimators = 70, gamma=7.5628225927223830,min_child_weight=6)

        multilabel_model = MultiOutputClassifier(xgb_estimator)
        multilabel_model.fit(x_tr, y_tr,verbose=True)

        evaluate_auc(multilabel_model,x_te,x_tr,y_tr,y_te,labels)


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
        print("MEAN")
        print("Accuracy train : ",xgb_estimator.score(x_tr,y_tra))
        print("Accuracy test : ",xgb_estimator.score(x_te,y_tes))


if __name__ == "__main__":
    main()