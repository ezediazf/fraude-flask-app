import os, joblib, numpy as np, pandas as pd
from flask import Flask, render_template, request, jsonify
app=Flask(__name__)
MODEL=os.path.join(os.path.dirname(__file__),'fraud_model.joblib')
ART=joblib.load(MODEL) if os.path.exists(MODEL) else None

def prepare(d):
    x=d.copy();x['hourOfDay']=(x.step-1)%24;x['day']=(x.step-1)//24+1;x['isNight']=x.hourOfDay.between(0,5).astype('int8');x['logAmount']=np.log1p(x.amount)
    x['origBalanceDelta']=x.newbalanceOrig-x.oldbalanceOrg;x['destBalanceDelta']=x.newbalanceDest-x.oldbalanceDest
    eo=np.where(x.type.eq('CASH_IN'),x.oldbalanceOrg+x.amount,x.oldbalanceOrg-x.amount);ed=np.where(x.type.eq('CASH_IN'),x.oldbalanceDest-x.amount,x.oldbalanceDest+x.amount)
    x['origBalanceError']=x.newbalanceOrig-eo;x['destBalanceError']=x.newbalanceDest-ed
    for a,b in [('origZeroBefore','oldbalanceOrg'),('origZeroAfter','newbalanceOrig'),('destZeroBefore','oldbalanceDest'),('destZeroAfter','newbalanceDest')]:x[a]=x[b].eq(0).astype('int8')
    x['destPrefix']=x.nameDest.str[0];x['amountToOrigBalance']=x.amount/(x.oldbalanceOrg.abs()+1);x['amountToDestBalance']=x.amount/(x.oldbalanceDest.abs()+1)
    x['origFrequency']=x.nameOrig.map(ART['origin_frequency']).fillna(0);x['destFrequency']=x.nameDest.map(ART['destination_frequency']).fillna(0)
    return x.drop(columns=['nameOrig','nameDest'])

def predict(payload):
    if ART is None: raise RuntimeError('Falta fraud_model.joblib. Copie Fraude.csv y ejecute python train_model.py')
    x=prepare(pd.DataFrame([payload]))[ART['features']];p=float(ART['model'].predict_proba(x)[:,1][0])
    return {'clasificacion':'ALERTA' if p>=ART['threshold'] else 'SIN ALERTA','probabilidad_fraude':p,'umbral':ART['threshold'],'modelo':ART['best_model']}
@app.route('/',methods=['GET','POST'])
def home():
    result=None
    if request.method=='POST':
        try:
            ks=['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest'];q={k:(request.form[k] if k in ['type','nameOrig','nameDest'] else float(request.form[k])) for k in ks};q['step']=int(q['step']);result=predict(q)
        except Exception as e: result={'error':str(e)}
    return render_template('index.html',result=result,ready=ART is not None)
@app.post('/api/predict')
def api_predict():
    try:return jsonify(predict(request.get_json(force=True)))
    except Exception as e:return jsonify({'error':str(e)}),503 if ART is None else 400
@app.get('/health')
def health():return ({'status':'ok','model':ART['best_model']} if ART else {'status':'model_missing'}),(200 if ART else 503)
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',5000)))
