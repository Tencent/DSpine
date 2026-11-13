"""CPU regressions for isolated, task-separated SPS budget search."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest

from scripts.sps_search import (
    aggregate, current_budget, interleaved_order, paired_decision,
    prefer_budget, replace_budget, run_search, split_tasks,
)


def document():
    return {"format":"dspine-sps-v2", "hardware":{},
            "policy":{"verify_widths":{},"token_budgets":{}},
            "mode_policies":{"greedy":{"verify_widths":{},"token_budgets":{"19":304}},
                             "sampled":{"verify_widths":{},"token_budgets":{"19":304}}}}


def measurement(budget, rates, status="PASS"):
    rounds=[{"repeat":i,"requests":128,"jobs_sha256":f"pair{i}","status":"PASS",
             "wall_seconds":128/rate,"output_tokens":128,"verify_calls":32,
             "tokens_per_second":rate} for i,rate in enumerate(rates)]
    return {**aggregate(rounds), "token_budget":budget, "status":status}


class SearchTests(unittest.TestCase):
    def test_split_is_task_disjoint_across_turns_and_sampler_families(self):
        rows=[{"dataset":"chat","task_id":task,"turn":turn,"input_ids":[task,turn],"_mode":mode}
              for task in range(10) for turn in (0,1) for mode in ("greedy","sampled")]
        tune,held,receipt=split_tasks(rows)
        self.assertTrue(receipt["task_disjoint"])
        self.assertEqual(len(tune)+len(held),40)
        self.assertFalse({r["task_id"] for r in tune}&{r["task_id"] for r in held})
        self.assertEqual(split_tasks(list(reversed(rows)))[2],receipt)

    def test_explicit_overlap_fails_before_any_measurement(self):
        a={"dataset":"chat","task_id":"a","input_ids":[1],"_mode":"greedy"}
        with self.assertRaisesRegex(ValueError,"identities overlap"):
            split_tasks([a],[{**a,"input_ids":[2]}])
        with self.assertRaisesRegex(ValueError,"tokens overlap"):
            split_tasks([a],[{**a,"task_id":"b"}])

    def test_one_entry_preserves_unknown_batches_and_other_mode(self):
        original=document();before=copy.deepcopy(original)
        changed=replace_budget(original,"greedy",7,32)
        self.assertEqual(original,before)
        self.assertEqual(changed["mode_policies"]["sampled"],before["mode_policies"]["sampled"])
        self.assertEqual(changed["mode_policies"]["greedy"]["token_budgets"],{"7":32,"19":304})
        original["policy"]["verify_widths"]={"3":8};original["mode_policies"].pop("greedy")
        self.assertEqual(current_budget(original,"greedy",3),24)
        changed=replace_budget(original,"greedy",3,16)
        self.assertEqual(current_budget(changed,"greedy",3),16)
        self.assertEqual(changed["mode_policies"]["greedy"]["verify_widths"],{})

    def test_one_percent_winner_is_tie_and_wider_candidate_wins(self):
        rows=[measurement(32,[101,101]),measurement(64,[100,100]),measurement(128,[70,70])]
        self.assertEqual(prefer_budget(rows,128)["selected_budget"],64)
        self.assertEqual(prefer_budget(rows[:2],64)["selected_budget"],64)
        self.assertFalse(paired_decision(rows[1],rows[0])["accepted"])

    def test_unstable_repeat_or_short_trial_cannot_win(self):
        baseline=measurement(64,[100,100])
        unstable=measurement(32,[130,95])
        self.assertFalse(paired_decision(baseline,unstable)["accepted"])
        short={**measurement(32,[200,200]),"status":"INSUFFICIENT"}
        self.assertEqual(prefer_budget([baseline,short],64)["selected_budget"],64)
        changed=copy.deepcopy(unstable);changed['rounds'][0]['jobs_sha256']='different'
        with self.assertRaisesRegex(ValueError,"paired complete"):
            paired_decision(baseline,changed)
        self.assertEqual(interleaved_order([1,2,3],0),[1,2,3])
        self.assertEqual(interleaved_order([1,2,3],1),[3,2,1])

    def test_full_search_can_validate_without_modifying_other_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=types.SimpleNamespace(min_requests=16,min_seconds=1,repeats=2,
                batch_sizes=[3,7],widths=[4,8,16],token_grid=[],min_gain=.03,max_new_tokens=4096)
            saved={};observed=[]
            def save(path,value):saved[str(path)]=copy.deepcopy(value)
            def budgets(sizes,width,grid):return {bs:min(bs*16,max(bs,16*round(bs*width/16))) for bs in sizes}
            def trial(doc,rows,bs,repeat,count,path):
                mode=rows[0]['_mode'];budget=current_budget(doc,mode,bs)
                if bs==3:rate={16:126,32:125,48:80}[budget]
                else:rate={32:130 if mode=='sampled' else 126,64:110 if mode=='sampled' else 125,112:80}[budget]
                observed.append((str(path),mode,bs,copy.deepcopy(doc)))
                return {'repeat':repeat,'requests':count,'jobs_sha256':f'{mode}-{bs}-{repeat}-{count}',
                        'status':'PASS','wall_seconds':2.,'output_tokens':int(2*rate),
                        'tokens_per_second':rate,'verify_calls':count}
            rows=[{'_mode':mode,'input_ids':[i]} for mode in ('greedy','sampled') for i in range(3)]
            chosen,report=run_search(args,document(),rows,rows,Path(temporary),trial,budgets,save)
            self.assertEqual(report['selection_status'],'VALIDATED')
            self.assertEqual(current_budget(chosen,'greedy',3),32)
            self.assertEqual(current_budget(chosen,'greedy',7),64)
            self.assertEqual(current_budget(chosen,'sampled',7),32)
            self.assertEqual(current_budget(chosen,'sampled',19),304)
            # During each coordinate sweep only the targeted table entry varies.
            grouped={}
            for path,mode,bs,doc in observed:
                if '/tune/' not in path:continue
                canonical=replace_budget(doc,mode,bs,bs*16)
                grouped.setdefault((mode,bs),[]).append(canonical)
            self.assertTrue(all(all(d==docs[0] for d in docs) for docs in grouped.values()))

    def test_duration_growth_reruns_every_candidate_with_same_longer_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=types.SimpleNamespace(min_requests=8,min_seconds=1,repeats=2,
                batch_sizes=[4],widths=[4,16],token_grid=[],min_gain=.03,max_new_tokens=4096)
            observed=[]
            def trial(doc,rows,bs,repeat,count,path):
                budget=current_budget(doc,'greedy',bs);rate=100 if budget==64 else 120
                observed.append((str(path),budget,count))
                wall=count*.01
                return {'repeat':repeat,'requests':count,'jobs_sha256':f'{repeat}-{count}',
                    'status':'PASS','wall_seconds':wall,'output_tokens':wall*rate,
                    'tokens_per_second':rate,'verify_calls':count}
            chosen,report=run_search(args,document(),[{'_mode':'greedy'}],[{'_mode':'greedy'}],Path(temporary),trial,
                lambda sizes,width,grid:{bs:bs*width for bs in sizes},lambda *a:None)
            first=[(b,n) for p,b,n in observed if '/tune/' in p and '/attempt0/' in p]
            grown=[(b,n) for p,b,n in observed if '/tune/' in p and '/attempt1/' in p]
            self.assertEqual({n for b,n in first},{32})
            self.assertTrue(grown and min(n for b,n in grown)>32)
            self.assertEqual({b for b,n in grown},{16,64})
            self.assertEqual(len({n for b,n in grown}),1)
            self.assertEqual(report['status'],'PASS')


if __name__=='__main__':unittest.main()
