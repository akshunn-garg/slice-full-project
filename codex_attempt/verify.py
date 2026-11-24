"""Independent post-training checks; reads frozen predictions and never tunes a model."""
from pathlib import Path
import json,sys,socket
import numpy as np
import pandas as pd
import joblib, requests
from predict import ROOT, extract, predict, FAMILIES, EXCLUDE

def main():
 a=json.loads((ROOT/'reports/leakage_audit.json').read_text())
 assert all(v==0 for v in a['overlap'].values())
 d=pd.read_pickle(ROOT/'private/prepared.pkl'); manifest=pd.read_csv(ROOT/'private/split_manifest.csv')
 assert len(d)==len(manifest) and d.email_id.is_unique
 for name in ['train','tune','calibration','threshold','test']:
  assert d.loc[d.partition==name,'label'].nunique()==2
 art=joblib.load(ROOT/'models/model.joblib')
 assert set(art['features'])==set(FAMILIES)
 for fam,cols in art['features'].items():
  assert all(c.startswith(fam+'__') and c not in EXCLUDE for c in cols)
 assert all(c.startswith('SRC__') for c in art['extra_columns'])
 sample=d[d.partition=='test'].iloc[:10].copy()
 expected=predict(sample,art)
 # Label, provenance, future-lookup columns must have no influence on inference.
 changed=sample.copy(); changed['label']=1-changed.label
 for c in list(changed.columns):
  if c.startswith('prov__'): changed[c]='INJECTED'
 changed['STS__new_sender_domain']=999
 changed['RCS__redirect_hop_label']=999
 changed['INFRA__DNS_TTL_avg']=999
 np.testing.assert_allclose(predict(changed,art).final_score,expected.final_score,rtol=0,atol=0)
 kept=[]
 def deny(*args,**kwargs):
  kept.append(True); raise AssertionError('Network operation attempted')
 socket.create_connection=deny;socket.getaddrinfo=deny;requests.sessions.Session.request=deny
 root=ROOT.parent/'phish_dataset_v2'
 for folder in ['in_legit','in_phish']:
  raw=next((root/'private/v4'/folder).glob('*.eml')).read_bytes()
  frame,n=extract(raw); result=predict(frame,art)
  assert np.isfinite(result.final_score).all() and n>0
 # A linkless email is explicitly outside the intended scope.
 frame,n=extract(b'From: Example <a@example.org>\r\nTo: b@example.net\r\nSubject: plain\r\n\r\nHello.')
 assert n==0
 assert not kept
 results={'cross_partition_overlap_checks':len(a['overlap']),'forbidden_features_excluded':True,
  'label_provenance_future_columns_have_zero_inference_effect':True,'network_denied_raw_email_inference':True,'linkless_scope_check':True,
  'model_reload_prediction_parity':'covered in latency benchmark','status':'PASS'}
 (ROOT/'reports/verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 print(json.dumps(results,indent=2))
if __name__=='__main__': main()
