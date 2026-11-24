"""Measure real raw-input latency on a frozen model; never change models or thresholds."""
import os,time,json,sys,csv,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from predict import ROOT,extract,predict

def distribution(v):
 return {'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'min_ms':float(np.min(v)),'max_ms':float(np.max(v))}

def main():
 source=ROOT.parent/'phish_dataset_v2'; d=pd.read_pickle(ROOT/'private/prepared.pkl')
 test=d[d.partition=='test'].copy(); model=joblib.load(ROOT/'models/model.joblib')
 mapping={x['email_id']:Path(x['source_file'].replace('\\','/')).name for x in csv.DictReader(open(source/'private/v4/id_map_v4.csv',encoding='utf-8'))}
 available={}
 for key in test.email_id:
  if key in mapping:
   for folder in ['inbox_eml','pot_raw']:
    p=source/'private'/folder/mapping[key]
    if p.exists(): available[key]=p; break
 chosen=test[test.email_id.isin(available)].sample(min(50,len(available)),random_state=20261115)
 feature_ms=[]; inference_ms=[]; total_ms=[]; io_ms=[]; deltas=[]; diff={}
 import socket,requests
 calls=[]
 def denied(*a,**k): calls.append(True); raise AssertionError('Unexpected network request')
 socket.create_connection=denied;socket.getaddrinfo=denied;requests.sessions.Session.request=denied
 with threadpool_limits(limits=4):
  predict(chosen.iloc[:1],model)
  for _,row in chosen.iterrows():
   t=time.perf_counter(); raw=available[row.email_id].read_bytes(); read=time.perf_counter()
   frame,n=extract(raw); mid=time.perf_counter(); out=predict(frame,model); end=time.perf_counter()
   feature_ms.append((mid-read)*1000); inference_ms.append((end-mid)*1000);total_ms.append((end-t)*1000);io_ms.append((read-t)*1000)
   ref=predict(pd.DataFrame([row]),model).final_score.iloc[0]
   deltas.append(abs(float(ref)-float(out.final_score.iloc[0])))
   for c in frame:
    a,b=frame.iloc[0][c],row[c]
    same=a==b if isinstance(a,str) else np.isclose(float(a),float(b),rtol=1e-10,atol=1e-10,equal_nan=True)
    if not same: diff[c]=diff.get(c,0)+1
 assert not calls
 # New interpreter per call: imports + artifact loading + one original raw-email prediction.
 cold=[]
 for key in chosen.email_id.iloc[:3]:
  t=time.perf_counter()
  p=subprocess.run([sys.executable,'-B',str(ROOT/'predict.py'),str(available[key])],capture_output=True,text=True,check=True)
  cold.append((time.perf_counter()-t)*1000)
 result={'raw_input_count':len(chosen),'original_raw_feature_extraction':distribution(feature_ms),'original_raw_model_inference':distribution(inference_ms),
  'original_raw_read_to_verdict':distribution(total_ms),'raw_file_read':distribution(io_ms),'fresh_process_read_to_verdict':distribution(cold),
  'normalized_vs_original_max_score_difference':max(deltas,default=0),'normalized_vs_original_differing_features':diff,
  'network_attempts':len(calls),'logical_cpus':os.cpu_count(),'threads_during_warm_benchmark':4,
  'notes':'Original inbox/phishing-pot .eml inputs, not pre-normalized copies; warm measurements include disk read, header normalization, static feature extraction and inference. Fresh-process measurements include interpreter startup/imports and model loading, with OS file cache warm; n=3.'}
 (ROOT/'reports/raw_latency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
