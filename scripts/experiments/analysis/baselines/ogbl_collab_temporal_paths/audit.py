"""Independent aggregate gate, calendar, and reused-case audit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from replicate import ages,make_graph

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args();d=json.loads((a.runtime/'results.json').read_text());private=json.loads((a.runtime/'private_cases.json').read_text());old=json.loads(a.old.read_text())
 outcomes=[]
 for y,r in d['years'].items():
  for name,c in r['cohorts'].items():
   v=sorted(g['younger_preference'] for g in c['groups']);n=len(v);assert n==c['usable_groups'];assert abs(sum(v)/n-c['younger_preference'])<1e-12;keep=n-int(np.ceil(.2*n));assert abs(sum(v[:keep])/keep-c['delete_best_20percent_preference'])<1e-12
  c=r['cohorts']['any_seed'];cc=r['cohorts']['consistent'];ok=c['usable_groups']>=30 and c['younger_preference']>.5 and c['delete_best_20percent_preference']>.5 and(cc['usable_groups']<10 or cc['younger_preference']>=.5);assert ok==r['replication_gate_pass'];outcomes.append(ok)
 assert all(outcomes)==d['gate_pass']
 # Two simple paths: dates use FIRST event, not latest recurrence; target-year excluded.
 es=np.array([[0,1],[1,2],[2,3],[0,1],[0,4],[4,5],[5,3],[0,6],[6,7],[7,3]]);ys=np.array([2010,2011,2012,2016,2014,2015,2016,2017,2017,2017])
 for year,expected in [(2017,[5,1]),(2016,[4])]:
  m=ys<year;adj,k,f,l=make_graph(es[m],ys[m],8);assert sorted(ages(0,3,adj,k,f,year))==sorted(expected);assert ages(0,0,adj,k,f,year)==[]
 current=private['2018']['features'];previous=old['cases'];shared=set(current)&set(previous)
 out={'complete':True,'gate_independently_recomputed':True,'calendar_first_event_and_cutoff_tests':True,'source_sha256_matches':hashlib.sha256((Path(__file__).parent/'replicate.py').read_bytes()).hexdigest()==d['source_sha256'],'old2018_example_reuse':{'current_cases':len(current),'previous_cases':len(previous),'shared_cases':len(shared),'shared_positive_cases':sum(k.startswith('p') for k in shared),'shared_negative_cases':sum(k.startswith('n') for k in shared)},'test_read_or_scored':False}
 assert out['source_sha256_matches'];(a.runtime/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
