import pandas as pd,numpy as np,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
raw=pd.read_csv('Fraude.csv').drop(columns=['Unnamed: 0','isFlaggedFraud'],errors='ignore')
def f(x):
 x=x.copy();x['hourOfDay']=(x.step-1)%24;x['day']=(x.step-1)//24+1;x['isNight']=x.hourOfDay.between(0,5).astype('int8');x['logAmount']=np.log1p(x.amount);x['origBalanceDelta']=x.newbalanceOrig-x.oldbalanceOrg;x['destBalanceDelta']=x.newbalanceDest-x.oldbalanceDest;eo=np.where(x.type.eq('CASH_IN'),x.oldbalanceOrg+x.amount,x.oldbalanceOrg-x.amount);ed=np.where(x.type.eq('CASH_IN'),x.oldbalanceDest-x.amount,x.oldbalanceDest+x.amount);x['origBalanceError']=x.newbalanceOrig-eo;x['destBalanceError']=x.newbalanceDest-ed
 for a,b in [('origZeroBefore','oldbalanceOrg'),('origZeroAfter','newbalanceOrig'),('destZeroBefore','oldbalanceDest'),('destZeroAfter','newbalanceDest')]:x[a]=x[b].eq(0).astype('int8')
 x['destPrefix']=x.nameDest.str[0];x['amountToOrigBalance']=x.amount/(x.oldbalanceOrg.abs()+1);x['amountToDestBalance']=x.amount/(x.oldbalanceDest.abs()+1);return x
x=f(raw);tr=x[x.step<=405].copy();va=x[x.step.between(406,548)].copy();om=tr.nameOrig.value_counts(normalize=True);dm=tr.nameDest.value_counts(normalize=True)
def ids(d):d=d.copy();d['origFrequency']=d.nameOrig.map(om).fillna(0);d['destFrequency']=d.nameDest.map(dm).fillna(0);return d.drop(columns=['nameOrig','nameDest'])
tr,va=ids(tr),ids(va);features=[c for c in tr if c!='isFraud'];cat=['type','destPrefix'];num=[c for c in features if c not in cat];pre=ColumnTransformer([('num',RobustScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)]);pos=tr[tr.isFraud.eq(1)];neg=tr[tr.isFraud.eq(0)].sample(len(pos)*8,random_state=42);fit=pd.concat([pos,neg]);model=Pipeline([('pre',pre),('model',RandomForestClassifier(n_estimators=80,max_depth=12,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1))]);model.fit(fit[features],fit.isFraud);s=model.predict_proba(va[features])[:,1];p,r,t=precision_recall_curve(va.isFraud,s);threshold=float(t[np.nanargmax((2*p*r/(p+r+1e-12))[:-1])]);joblib.dump({'model':model,'threshold':threshold,'features':features,'origin_frequency':om.to_dict(),'destination_frequency':dm.to_dict(),'best_model':'Random Forest'},'fraud_model.joblib',compress=3);print('Modelo generado:',threshold)
