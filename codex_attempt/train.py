"""Rebuild SLiCE from raw mail under a frozen, offline, purged protocol."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys, json, csv, re, hashlib, time, platform, logging, warnings
from email import policy
from email.parser import BytesParser
from itertools import combinations
import numpy as np
import pandas as pd
import joblib
from bs4 import BeautifulSoup
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, confusion_matrix
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from predict import ROOT, FAMILIES, EXCLUDE, extract, numeric, rule_risk, logit, combine, predict
SOURCE=ROOT.parent/'phish_dataset_v2'
SEED=20261115
SPLITS=['train','tune','calibration','threshold','test']
PROTOCOL={
 'seed':SEED,'input':'existing v4 cohort plus the three existing training additions; all features re-extracted offline from raw normalized mail',
 'chronology':'main-cohort time quantiles 55/70/80/85%; extras restricted to <= original train cutoff; all group components retained only in their latest partition',
 'groups':'transitive closure of sender/link-organization template, normalized body fingerprint, Message-ID, and SHA256; conflicting exact duplicates excluded',
 'base_selection':'nested grouped 3-fold OOF on TRAIN; RF, LightGBM, logistic regression fixed candidates; no tuning on test',
 'link_selection':'fixed LightGBM and logistic candidates, selected on TUNE only using TRAIN nested-OOF base scores',
 'calibration':'class-balanced Platt on CALIBRATION only; final fixed STS weight 3',
 'threshold':'max recall subject to empirical FPR <=1% on THRESHOLD only',
 'evaluation':'TEST is evaluated once after model and threshold are saved; 95% cluster-bootstrap intervals',
 'excluded':'all RCS, all INFRA, redirect results, DKIM-derived link features, sender age; no label/provenance feature inputs',
 'limitations':['Historical cohort already used during earlier project development; this is a protocol-validation rerun, not a pristine external holdout.',
 'Source-label confounding remains: one main legitimate mailbox and one main phishing honeypot.',
 'Existing cohort eligibility was influenced by earlier live-processing success and feed screening; offline re-extraction cannot undo that selection.',
 'Static shortener/brand lists are frozen current code, not historically versioned threat intelligence.',
 'Content fingerprinting and template grouping reduce repetition but do not prove every near-duplicate or semantic campaign has been found.',
 'Authentication verdicts are trusted receiver-provided arrival headers; arbitrary uploaded header authenticity is not verified.']}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def write_json(path,obj): path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')

def resolve(s):
 p=Path(str(s).replace('\\','/'))
 if p.exists(): return p
 p=SOURCE/p
 if p.exists(): return p
 raise FileNotFoundError(str(s))

def data_and_paths():
 logging.info('Loading original cohort (no existing predictions or models)')
 d=pd.read_excel(SOURCE/'slice_dataset_v4.xlsx',sheet_name='Dataset')
 main_count=len(d)
 paths={}
 for x in csv.DictReader(open(SOURCE/'private/v4/id_map_v4.csv',encoding='utf-8')):
  paths[x['email_id']]=resolve(x['source_file'])
 cutoff=pd.to_datetime(d.loc[d.prov__time_split=='train','prov__arrival_time']).max()
 boundaries=pd.to_datetime(d.prov__arrival_time).quantile([.55,.70,.80,.85]).tolist()
 inputs=[SOURCE/'slice_dataset_v4.xlsx']
 for fn,checkpoint in [('private/nazario_train/train_rows.csv','private/nazario_train/checkpoint.jsonl'),
  ('private/v4/extra_legit_rows.csv','private/v4/extra_legit_checkpoint.jsonl'),
  ('private/v4/unused_train_rows.csv','private/v4/unused_train_checkpoint.jsonl')]:
  p=SOURCE/fn; e=pd.read_csv(p); inputs.append(p)
  assert (pd.to_datetime(e.prov__arrival_time)<=cutoff).all(), 'Extra training rows cross original cutoff'
  wanted={re.search(r'(?:email|phish)_\d+$',str(k)).group():str(k) for k in e.email_id}
  with open(SOURCE/checkpoint,encoding='utf-8') as f:
   for line in f:
    rec=json.loads(line)
    if rec['email_id'] in wanted: paths[wanted[rec['email_id']]]=resolve(rec['path'])
  d=pd.concat([d,e],ignore_index=True)
 assert d.email_id.is_unique
 assert all(str(x) in paths for x in d.email_id)
 d['arrival']=pd.to_datetime(d.prov__arrival_time)
 assert d.arrival.notna().all()
 d['partition']=np.array(SPLITS)[np.searchsorted(np.array(boundaries,dtype='datetime64[ns]'),d.arrival.to_numpy(),side='left')]
 # Additions are admitted only to TRAIN, and must predate its new cutoff as well.
 extra=d.index>=main_count
 d.loc[extra & (d.arrival>boundaries[0]),'partition']='excluded_extra_after_new_train_cutoff'
 return d,paths,inputs,boundaries

def fingerprints(raw):
 msg=BytesParser(policy=policy.default).parsebytes(raw)
 parts=[]
 for p in msg.walk():
  if p.get_content_maintype()=='text' and p.get_content_disposition()!='attachment':
   b=p.get_payload(decode=True)
   if b:
    try: txt=b.decode(p.get_content_charset() or 'utf-8',errors='replace')
    except LookupError: txt=b.decode('utf-8',errors='replace')
    if p.get_content_subtype()=='html':
     soup=BeautifulSoup(txt,'html.parser')
     for n in soup(['script','style']): n.decompose()
     txt=soup.get_text(' ',strip=True)
    parts.append(txt)
 body=' '.join(parts).lower()
 body=re.sub(r'[\w.+-]+@[\w.-]+',' EMAIL ',body)
 body=re.sub(r'https?://\S+',' URL ',body)
 body=re.sub(r'\b[0-9a-f]{12,}\b|\d+',' N ',body)
 body=re.sub(r'\s+',' ',body).strip()
 digest=hashlib.sha256(body.encode()).hexdigest() if len(body.split())>=30 else ''
 return {'raw_sha256':hashlib.sha256(raw).hexdigest(),'body_fingerprint':digest,'message_id':str(msg.get('Message-ID','')).strip().lower()}

def extract_record(item):
 key,path=item
 import socket,requests
 def denied(*a,**k): raise RuntimeError('NETWORK_DISABLED_IN_OFFLINE_PIPELINE')
 socket.create_connection=denied; socket.getaddrinfo=denied; requests.sessions.Session.request=denied
 raw=Path(path).read_bytes(); frame,nlinks=extract(raw)
 return {'email_id':key,**fingerprints(raw),'n_links':nlinks,'features':frame.iloc[0].to_dict()}


def prepare():
 d,paths,inputs,bounds=data_and_paths()
 cache=ROOT/'private/offline_rows.jsonl'; records={}
 if cache.exists():
  for l in cache.open(encoding='utf-8'):
   try: r=json.loads(l); records[r['email_id']]=r
   except json.JSONDecodeError: pass
 # Deny real network operations even if a helper accidentally calls one.
 import socket,requests
 def denied(*a,**k): raise RuntimeError('NETWORK_DISABLED_IN_OFFLINE_PIPELINE')
 socket.create_connection=denied; socket.getaddrinfo=denied; requests.sessions.Session.request=denied
 n=len(d); logging.info('Offline raw-email extraction and fingerprinting: %d rows, %d cached; four workers',n,len(records))
 from concurrent.futures import ProcessPoolExecutor
 pending=[(str(k),str(paths[str(k)])) for k in d.email_id if str(k) not in records]
 with cache.open('a',encoding='utf-8') as f, ProcessPoolExecutor(max_workers=4) as pool:
  for i,rec in enumerate(pool.map(extract_record,pending,chunksize=8),1):
   f.write(json.dumps(rec,default=str)+'\n'); f.flush(); records[rec['email_id']]=rec
   if i%250==0: logging.info('Extracted/audited %d/%d',len(records),n)
 # Header normalization is idempotent: canonical training files are not normalized twice.
 # Reconcile the header-only feature in older extraction-cache rows; static link fields are unchanged.
 from email.parser import BytesHeaderParser
 from predict import sts
 corrected=0
 for key in d.email_id.astype(str):
  raw=paths[key].read_bytes(); msg=BytesHeaderParser(policy=policy.default).parsebytes(raw)
  if not str(msg.get('Authentication-Results','')).startswith('slice-normalised;'):
   msg=BytesHeaderParser(policy=policy.default).parsebytes(__import__('predict').normalise(raw,['phishing@pot'])[0])
  value=sts.analyze_received_chain(msg)['unusual_received_chain']
  corrected+=records[key]['features']['STS__unusual_received_chain']!=value
  records[key]['features']['STS__unusual_received_chain']=value
 logging.info('Header normalization reconciled: %d cached header values corrected',corrected)
 # Check complete feature extraction against cached values, without computing model scores.
 chosen=d.sample(min(40,len(d)),random_state=SEED)
 for key in chosen.email_id.astype(str):
  check,_=extract(paths[key].read_bytes()); before=records[key]['features']
  for col,value in check.iloc[0].items():
   previous=before[col]
   if isinstance(value,str): assert value==previous,(key,col,value,previous)
   else: np.testing.assert_allclose(float(value),float(previous),rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=key+' '+col)
 tmp=cache.with_suffix('.reconciled.jsonl')
 with tmp.open('w',encoding='utf-8') as f:
  for key in d.email_id.astype(str): f.write(json.dumps(records[key],default=str)+'\n')
 tmp.replace(cache)
 write_json(ROOT/'reports/normalization_checks.json',{'corrected_header_values':int(corrected),'full_feature_parity_samples':len(chosen),'status':'PASS'})
 feats=pd.DataFrame([records[str(k)]['features'] for k in d.email_id])
 for c in ['raw_sha256','body_fingerprint','message_id','n_links']: d[c]=[records[str(k)][c] for k in d.email_id]
 d=pd.concat([d[['email_id','label','prov__source','prov__template','arrival','partition','raw_sha256','body_fingerprint','message_id','n_links']],feats],axis=1)
 d=d[d.n_links>0].copy()
 # Duplicate-label conflicts invalidate the affected raw-message groups.
 conflict=d.groupby('raw_sha256').label.nunique(); bad=set(conflict[conflict>1].index)
 excluded=d[d.raw_sha256.isin(bad)].assign(exclusion='conflicting_raw_labels')
 d=d[~d.raw_sha256.isin(bad)].sort_values(['arrival','email_id'])
 dup=d.duplicated('raw_sha256',keep='last')
 excluded=pd.concat([excluded,d[dup].assign(exclusion='duplicate_raw_email')]); d=d[~dup].copy().reset_index(drop=True)
 parent=list(range(len(d)))
 def find(a):
  while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
  return a
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b: parent[b]=a
 for col in ['prov__template','raw_sha256','body_fingerprint','message_id']:
  seen={}
  for i,v in enumerate(d[col].fillna('').astype(str)):
   if v and v!='nan':
    if v in seen: union(i,seen[v])
    else: seen[v]=i
 d['group']=[find(i) for i in range(len(d))]
 ranks={s:i for i,s in enumerate(SPLITS)}
 d['rank']=d.partition.map(ranks).fillna(-1).astype(int)
 latest=d.groupby('group')['rank'].transform('max')
 purge=(d['rank']!=latest)|(d['rank']<0)
 excluded=pd.concat([excluded,d[purge].assign(exclusion='group_overlap_or_extra_after_cutoff')])
 d=d[~purge].copy().reset_index(drop=True)
 audit={'boundaries':[str(b) for b in bounds],'retained_rows':len(d),'excluded_rows':len(excluded),
  'counts':d.groupby(['partition','label']).size().unstack(fill_value=0).to_dict(), 'overlap':{}}
 for a,b in combinations(SPLITS,2):
  for col in ['group','prov__template','raw_sha256','body_fingerprint','message_id']:
   sa=set(d.loc[d.partition==a,col].dropna())-{'','nan'}; sb=set(d.loc[d.partition==b,col].dropna())-{'','nan'}
   val=len(sa&sb); audit['overlap'][a+'|'+b+'|'+col]=val; assert val==0
 for s in SPLITS:
  part=d[d.partition==s]
  if part.label.nunique()!=2 or part.groupby('label').size().min()<15: raise ValueError(f'Insufficient classes after purging in {s}: {part.label.value_counts().to_dict()}')
 for a,b in zip(SPLITS,SPLITS[1:]): assert d.loc[d.partition==a,'arrival'].max()<=d.loc[d.partition==b,'arrival'].min()
 d[['email_id','label','arrival','partition','group','prov__source','prov__template','raw_sha256','body_fingerprint','message_id']].to_csv(ROOT/'private/split_manifest.csv',index=False)
 excluded[['email_id','exclusion']].to_csv(ROOT/'private/exclusions.csv',index=False)
 d.to_pickle(ROOT/'private/prepared.pkl')
 write_json(ROOT/'reports/leakage_audit.json',audit)
 write_json(ROOT/'reports/input_hashes.json',{str(p.relative_to(SOURCE)):sha(p) for p in inputs})
 logging.info('Partition counts after purging:\n%s',d.groupby(['partition','label']).size().unstack(fill_value=0).to_string())
 return d

def options():
 return [('rf',lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=120,min_samples_leaf=10,n_jobs=4,random_state=SEED))),
 ('lgb',lambda:LGBMClassifier(n_estimators=160,num_leaves=15,min_child_samples=25,learning_rate=.05,n_jobs=4,verbosity=-1,random_state=SEED,deterministic=True,force_col_wise=True)),
 ('logistic',lambda:make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.1,max_iter=1500)))]

def select_base(X,y,g):
 folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(X,y,g))
 best=None; rows=[]
 for name,factory in options():
  p=np.empty(len(y))
  for a,b in folds: p[b]=factory().fit(X.iloc[a],y[a]).predict_proba(X.iloc[b])[:,1]
  ap=float(average_precision_score(y,p)); ll=float(log_loss(y,np.clip(p,1e-7,1-1e-7)))
  rows.append({'algorithm':name,'pr_auc':ap,'log_loss':ll})
  if best is None or (ap,-ll)>best[0]: best=((ap,-ll),name,factory)
 return best[1],best[2],rows

def metric(y,p,threshold):
 pred=(np.asarray(p)>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
 return {'n':len(y),'accuracy':float((tp+tn)/len(y)),'precision':float(prec),'recall':float(rec),'f1':float(2*prec*rec/max(prec+rec,1e-12)),
  'fpr':float(fp/max(fp+tn,1)),'specificity':float(tn/max(tn+fp,1)),'roc_auc':float(roc_auc_score(y,p)),
  'pr_auc':float(average_precision_score(y,p)),'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7))),
  'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':float(threshold)}

def choose_threshold(y,p):
 # Never extrapolate a zero-FP threshold from only positive scores.
 candidates=np.r_[np.nextafter(max(p),np.inf),np.unique(p)]
 valid=[]
 for t in candidates:
  flag=p>=t; fp=int(((y==0)&flag).sum()); tp=int(((y==1)&flag).sum())
  if fp/max(int((y==0).sum()),1)<=.01: valid.append((tp,-fp,float(t)))
 return max(valid)[2]

def train(d):
 tr=np.flatnonzero(d.partition=='train'); tu=np.flatnonzero(d.partition=='tune'); ca=np.flatnonzero(d.partition=='calibration'); th=np.flatnonzero(d.partition=='threshold')
 y=d.label.to_numpy(dtype=int); g=d.group.to_numpy(); bases={}; features={}; score_full={}; score_oof={}; search=[]
 for fam in FAMILIES:
  cols=[c for c in d if c.startswith(fam+'__') and c not in EXCLUDE]; features[fam]=cols
  X=numeric(d,cols); Xt=X.iloc[tr].reset_index(drop=True); yt=y[tr]; gt=g[tr]
  outer=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED+1).split(Xt,yt,gt)); oof=np.empty(len(tr))
  for j,(a,b) in enumerate(outer):
   name,factory,rows=select_base(Xt.iloc[a].reset_index(drop=True),yt[a],gt[a])
   oof[b]=factory().fit(Xt.iloc[a],yt[a]).predict_proba(Xt.iloc[b])[:,1]
   search.extend([{**r,'family':fam,'outer_fold':j} for r in rows])
   logging.info('%s nested OOF fold %d complete (%s)',fam,j,name)
  name,factory,rows=select_base(Xt,yt,gt); model=factory().fit(Xt,yt); bases[fam]=model
  score_full[fam]=model.predict_proba(X)[:,1]; score_oof[fam]=oof
  search.extend([{**r,'family':fam,'outer_fold':'full_train'} for r in rows])
  logging.info('%s full-TRAIN winner: %s',fam,name)
 pd.DataFrame(search).to_csv(ROOT/'reports/base_selection.csv',index=False)
 extra=[c for c in d if c.startswith('SRC__')]
 M=pd.DataFrame(score_full,index=d.index)
 for c in extra: M[c]=numeric(d,[c])[c]
 Mtrain=M.iloc[tr].copy()
 for fam in FAMILIES: Mtrain[fam]=score_oof[fam]
 best=None; final_search=[]
 # Same modular stacking method; monotone link-score inputs, static SRC replacing live INFRA.
 for leaves in [7,15,31]:
  model=LGBMClassifier(n_estimators=180,num_leaves=leaves,min_child_samples=25,learning_rate=.05,n_jobs=4,verbosity=-1,random_state=SEED,
   deterministic=True,force_col_wise=True,monotone_constraints=[1]*len(FAMILIES)+[0]*len(extra))
  model.fit(Mtrain,y[tr]); p=model.predict_proba(M.iloc[tu])[:,1]
  ap=float(average_precision_score(y[tu],p)); ll=float(log_loss(y[tu],np.clip(p,1e-7,1-1e-7)))
  final_search.append({'algorithm':'lightgbm','leaves':leaves,'tune_pr_auc':ap,'tune_log_loss':ll})
  if best is None or (ap,-ll)>best[0]: best=((ap,-ll),model,leaves)
 # Logistic candidate uses the same probability columns (fitted preprocessing inside TRAIN).
 model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.1,max_iter=1500))
 model.fit(Mtrain,y[tr]); p=model.predict_proba(M.iloc[tu])[:,1]
 ap=float(average_precision_score(y[tu],p)); ll=float(log_loss(y[tu],np.clip(p,1e-7,1-1e-7)))
 final_search.append({'algorithm':'logistic','leaves':None,'tune_pr_auc':ap,'tune_log_loss':ll})
 if (ap,-ll)>best[0]: best=((ap,-ll),model,None)
 pd.DataFrame(final_search).to_csv(ROOT/'reports/link_selection.csv',index=False)
 raw=best[1].predict_proba(M)[:,1]
 cal=LogisticRegression(class_weight='balanced',C=1e6,max_iter=1500).fit(logit(raw[ca]).reshape(-1,1),y[ca])
 link=cal.predict_proba(logit(raw).reshape(-1,1))[:,1]; final=combine(link,rule_risk(d))
 threshold=choose_threshold(y[th],final[th])
 artifact={'bases':bases,'features':features,'extra_columns':extra,'link_model':best[1],'calibrator':cal,'threshold':threshold,
  'protocol':PROTOCOL,'features_timing':'arrival-time-only static evidence','final_score_is_calibrated_probability':False,
  'notes':'Platt calibration covers link score; fixed STS combination is a risk score, not verified deployment probability.'}
 joblib.dump(artifact,ROOT/'models/model.joblib',compress=3)
 write_json(ROOT/'reports/threshold_selection.json',metric(y[th],final[th],threshold))
 write_json(ROOT/'reports/model_manifest.json',{'features':features,'extra_columns':extra,'threshold':threshold,'link_algorithm':type(best[1]).__name__,
  'model_sha256':sha(ROOT/'models/model.joblib'),'sts_weights':__import__('predict').RULE_WEIGHTS,'sts_combination_weight':3})
 logging.info('Model and threshold frozen. Starting the single final test evaluation.')
 return artifact

def evaluate(d,art):
 te=d[d.partition=='test'].reset_index(drop=True); y=te.label.to_numpy(dtype=int); out=predict(te,art); p=out.final_score.to_numpy()
 m=metric(y,p,art['threshold']); rng=np.random.default_rng(SEED); groups=te.group.unique(); draws=[]
 indices={k:np.flatnonzero(te.group.to_numpy()==k) for k in groups}
 for _ in range(1000):
  ix=np.concatenate([indices[k] for k in rng.choice(groups,len(groups),replace=True)])
  if len(np.unique(y[ix]))==2: draws.append(metric(y[ix],p[ix],art['threshold']))
 ci={k:[float(x) for x in np.quantile([v[k] for v in draws],[.025,.975])] for k in ['accuracy','precision','recall','f1','fpr','roc_auc','pr_auc']}
 report={'test':m,'cluster_bootstrap_95pct':ci,'test_groups':len(groups),'bootstrap_replicates':len(draws),'limitations':PROTOCOL['limitations']}
 for prevalence in [.01,.001]: report['precision_if_prevalence_'+str(prevalence)]=m['recall']*prevalence/max(m['recall']*prevalence+m['fpr']*(1-prevalence),1e-12)
 write_json(ROOT/'reports/metrics.json',report)
 pd.concat([te[['email_id','label','group']],out],axis=1).to_csv(ROOT/'private/test_predictions.csv',index=False)
 # Latency: warm single-row calls, batch throughput, and actual raw-email extraction + prediction.
 for _ in range(5): predict(te.iloc[:1],art)
 single=[]
 for i in range(100):
  frame=te.iloc[[i%len(te)]]; t=time.perf_counter(); predict(frame,art); single.append((time.perf_counter()-t)*1000)
 batch=[]
 for _ in range(5):
  t=time.perf_counter(); predict(te,art); batch.append((time.perf_counter()-t)*1000/len(te))
 # Locate test raw messages without reloading the workbook.
 rawmap={r['email_id']:resolve(r['source_file']) for r in csv.DictReader(open(SOURCE/'private/v4/id_map_v4.csv',encoding='utf-8'))}
 candidates=te[te.email_id.isin(rawmap)].sample(min(50,len(te[te.email_id.isin(rawmap)])),random_state=SEED)
 feat_times=[]; total_times=[]; parity=[]
 for _,row in candidates.iterrows():
  raw=rawmap[row.email_id].read_bytes(); t=time.perf_counter(); frame,_=extract(raw); middle=time.perf_counter(); q=predict(frame,art); end=time.perf_counter()
  feat_times.append((middle-t)*1000); total_times.append((end-t)*1000)
  reference=predict(pd.DataFrame([row]),art).final_score.iloc[0]
  parity.append(abs(float(q.final_score.iloc[0])-float(reference)))
 assert max(parity,default=0)<1e-10, 'Offline feature/inference reproducibility failed'
 t=time.perf_counter(); reloaded=joblib.load(ROOT/'models/model.joblib'); load=(time.perf_counter()-t)*1000
 np.testing.assert_allclose(predict(te,reloaded).final_score,p,atol=1e-12)
 def dist(v): return {'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))}
 latency={'single_email_warm_model':dist(single),'batch_amortized_per_email':dist(batch),'offline_feature_extraction':dist(feat_times),
  'raw_email_to_verdict_warm':dist(total_times),'reload_ms_os_cache_warm':load,'parity_max_error':max(parity,default=0),'threads':4,
  'machine':platform.platform(),'processor':platform.processor(),'python':sys.version,'timing_notes':'Raw bytes read before timer; features include normalization, parsing and temporary-file I/O; no network, no lookup cache. Warm repeated measurements on this machine, not service SLA.'}
 write_json(ROOT/'reports/latency.json',latency)
 logging.info('TEST results: %s',json.dumps(m)); logging.info('Latency: %s',json.dumps(latency))


def main():
 for name in ['models','reports','private']: (ROOT/name).mkdir(exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(ROOT/'reports/run.log',encoding='utf-8')])
 warnings.filterwarnings('ignore',message='X does not have valid feature names')
 if (ROOT/'reports/metrics.json').exists(): raise RuntimeError('Final test already evaluated. This script refuses to retune/re-evaluate it in place.')
 protocol_path=ROOT/'PROTOCOL.json'
 if protocol_path.exists(): assert json.loads(protocol_path.read_text(encoding='utf-8'))==PROTOCOL
 else: write_json(protocol_path,PROTOCOL)
 t=time.perf_counter()
 with threadpool_limits(limits=4):
  if (ROOT/'private/prepared.pkl').exists(): d=pd.read_pickle(ROOT/'private/prepared.pkl')
  else: d=prepare()
  art=train(d); evaluate(d,art)
 logging.info('Completed in %.1f seconds',time.perf_counter()-t)
if __name__=='__main__': main()
