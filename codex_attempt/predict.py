"""Frozen arrival-time-only SLiCE inference. No network lookup is executed."""
from pathlib import Path
import sys, tempfile, time, json, argparse
import numpy as np
import pandas as pd
import joblib
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from feature_modules import bas_features as bas, dms_features as dms, los_features as los, sts_features as sts
from feature_modules import source_link_infra_features as src
from build_v4.normalize import normalise
# Only discarded outputs use these functions. They must never contact the network.
bas.get_final_redirect_domain = lambda url: None
dms.get_final_redirect_domain = lambda url: None
FAMILIES = ('BAS','DMS','LOS')
EXCLUDE = {'BAS__from_domain_equals_dkim_domain','BAS__authenticated_domain_equals_destination',
 'BAS__claimed_or_authenticated_brand_equals_final_redirect','DMS__dkim_domain_equals_destination',
 'DMS__final_domain_equals_expected'}
RULE_WEIGHTS = {'spf':1.,'dkim':1.,'dmarc':1.5,'reply_to_mismatch':1.,
 'return_path_mismatch':.5,'sender_domain_entropy':.5,'unusual_received_chain':.5}

def numeric(df, columns):
    missing=set(columns)-set(df.columns)
    if missing: raise ValueError(f'Missing required features: {sorted(missing)}')
    return df[columns].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)

def rule_risk(df):
    def col(k): return df[k].astype(str).str.lower()
    spf=col('STS_meta__spf_result').map({'pass':0.,'fail':1.}).fillna(.5)
    dk=col('STS_meta__dkim_result')
    dkim=np.where(dk=='pass',np.where(df['STS__dkim_aligned']==1,0.,.5),np.where(dk.isin(['none','unknown','','nan']),.6,1.))
    dm=col('STS_meta__dmarc_result').map({'pass':0.,'fail':1.,'permerror':1.}).fillna(.5)
    ent=df['STS__sender_domain_entropy'].fillna(0)
    return (spf+dkim+1.5*dm+df['STS__from_reply_to_mismatch'].fillna(0)+.5*df['STS__from_return_path_mismatch'].fillna(0)
      +.5*np.where(ent>3.8,1.,np.where(ent>3.5,.5,0.))+.5*df['STS__unusual_received_chain'].fillna(0))/sum(RULE_WEIGHTS.values())

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def combine(link,risk): return 1/(1+np.exp(-(logit(link)+3*(np.asarray(risk)-.5))))

def predict(df, artifact):
    scores={s:artifact['bases'][s].predict_proba(numeric(df,artifact['features'][s]))[:,1] for s in FAMILIES}
    M=pd.DataFrame(scores,index=df.index)
    for c in artifact['extra_columns']: M[c]=numeric(df,[c])[c]
    raw=artifact['link_model'].predict_proba(M)[:,1]
    link=artifact['calibrator'].predict_proba(logit(raw).reshape(-1,1))[:,1]
    risk=rule_risk(df)
    final=combine(link,risk)
    out=pd.DataFrame({**{s+'_score':v for s,v in scores.items()},'link_score':link,'sts_risk':risk,'final_score':final},index=df.index)
    out['flag']=(final>=artifact['threshold']).astype(int)
    return out

def extract(raw):
    from email import policy
    from email.parser import BytesParser
    original=BytesParser(policy=policy.default).parsebytes(raw)
    already_normalized=str(original.get('Authentication-Results','')).startswith('slice-normalised;')
    norm=raw if already_normalized else normalise(raw,['phishing@pot'])[0]
    msg=original if already_normalized else BytesParser(policy=policy.default).parsebytes(norm)
    row={}
    with tempfile.TemporaryDirectory(prefix='slice_offline_') as td:
        p=Path(td)/'email.eml'; p.write_bytes(norm)
        for fam,fn in [('BAS',bas.extract_bas_features),('DMS',dms.extract_dms_features),('LOS',los.extract_los_features)]:
            features,meta=fn(p)
            row.update({fam+'__'+k:v for k,v in features.items() if fam+'__'+k not in EXCLUDE})
    dk,_,aligned=sts.extract_dkim(msg)
    row.update({'STS_meta__spf_result':sts.extract_spf(msg),'STS_meta__dkim_result':dk,'STS_meta__dmarc_result':sts.extract_dmarc(msg),
      'STS__dkim_aligned':aligned,'STS__from_reply_to_mismatch':sts.from_reply_to_mismatch(msg),
      'STS__from_return_path_mismatch':sts.from_return_path_mismatch(msg),'STS__sender_domain_entropy':sts.calculate_entropy(sts.extract_email_domain(msg.get('From','')) or ''),
      'STS__unusual_received_chain':sts.analyze_received_chain(msg)['unusual_received_chain']})
    urls=src.extract_links(src.get_html(msg)); brand=bas.extract_claimed_brand(msg)
    per=[src.extract_url_features(u,brand) for u in urls]
    for k in ['hostname_length','path_length','dot_count','hyphen_count','slash_count','query_parameter_count','digit_count','repeated_brand_terms']:
        vals=[x[k] for x in per]
        row['SRC__'+k+'_avg']=float(np.mean(vals)) if vals else 0.
        row['SRC__'+k+'_max']=max(vals) if vals else 0.
    for k in ['at_symbol','suspicious_tld']: row['SRC__'+k+'_any']=max([x[k] for x in per],default=0)
    return pd.DataFrame([row]),len(urls)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('email'); ap.add_argument('--model',default=str(ROOT/'models/model.joblib'))
    args=ap.parse_args(); t=time.perf_counter(); art=joblib.load(args.model); load=time.perf_counter()-t
    t=time.perf_counter(); frame,n=extract(Path(args.email).read_bytes()); fe=time.perf_counter()-t
    t=time.perf_counter(); result=predict(frame,art).iloc[0].to_dict(); inf=time.perf_counter()-t
    print(json.dumps({'result':result,'in_scope':n>0,'links':n,'load_ms':load*1000,'features_ms':fe*1000,'inference_ms':inf*1000},indent=2))
if __name__=='__main__': main()
