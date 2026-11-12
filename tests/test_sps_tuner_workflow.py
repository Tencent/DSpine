"""Exercise the actual tuner CLI with a CPU-only service callback."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.sps_search import current_budget


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # CPU import only; main's hardware access is replaced below.
        except ImportError:
            raise unittest.SkipTest("the deployment package requires torch")
        path = Path(__file__).resolve().parents[1] / "scripts/tune_sps.py"
        spec = importlib.util.spec_from_file_location("sps_workflow_under_test", path)
        cls.tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tuner)

    def test_cli_validates_independent_long_holdout_and_keeps_unknown_batch(self):
        m = self.tuner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def data(tasks):
                return [{"dataset":"train","task_id":task,"input_ids":[task],
                         "sampling_params":{"temperature":temp}}
                        for task in tasks for temp in (0.0,.37)]
            tune,hold=root/"tune.jsonl",root/"holdout.jsonl"
            tune.write_text("".join(json.dumps(r)+"\n" for r in data(range(1,7))))
            hold.write_text("".join(json.dumps(r)+"\n" for r in data(range(10,13))))
            baseline=root/"baseline.json"
            baseline.write_text(json.dumps({"format":"dspine-sps-v2","hardware":{},
                "policy":{"verify_widths":{},"token_budgets":{"19":304}},"mode_policies":{}}))
            output=root/"sps.json";calls=[]
            @contextlib.contextmanager
            def server(*args,**kwargs):yield "http://cpu"
            def request(*args,**kwargs):
                return {"speculative_algorithm":"DSPINE","speculative_num_draft_tokens":16,
                    "cuda_graph_config":{"decode":{"backend":"full","bs":list(range(1,17))}}}
            def trial(command,document,rows,bs,repeat,count,maximum,folder,port,startup_timeout,request_timeout=None):
                mode=m.prefix_mode(rows[0]);budget=current_budget(document,mode,bs)
                rate={16:126,32:125,48:80}[budget] if bs==3 else {32:126,64:125,80:100,112:80}[budget]
                calls.append((mode,bs,count,maximum,request_timeout))
                return {"repeat":repeat,"requests":count,"jobs_sha256":f"{mode}-{bs}-{repeat}-{count}",
                        "status":"PASS","wall_seconds":2.,"output_tokens":int(2*rate),
                        "tokens_per_second":rate,"verify_calls":count}
            argv=[str(m.__file__),"--requests",str(tune),"--holdout-requests",str(hold),
                "--baseline-profile",str(baseline),"--output",str(output),"--batch-sizes","3","7",
                "--repeats","2","--min-seconds","1","--min-requests","4","--request-timeout","3600",
                "--","sglang","serve","--speculative-algorithm","DSPINE"]
            with patch.object(m,"hardware_identity",return_value={}),patch.object(m,"server",server),\
                    patch.object(m,"request",request),patch.object(m,"run_profile_trials",trial),\
                    patch.object(sys,"argv",argv):
                m.main()
            result=json.loads(output.read_text());report=json.loads((root/"sps.runs/RESULTS.json").read_text())
            self.assertEqual(result['tuning']['selection_status'],'VALIDATED')
            self.assertEqual(current_budget(result,'greedy',7),64)
            self.assertEqual(current_budget(result,'sampled',19),304)
            self.assertTrue(calls and all(x[3:]==(4096,3600) for x in calls))
            self.assertEqual(report['status'],'PASS')
            self.assertTrue(all(x['pass'] for x in report['neighbor_validation']+report['table_validation']))
            partition=json.loads((root/"sps.runs/INPUTS.json").read_text())
            self.assertTrue(partition['task_disjoint'] and partition['input_ids_disjoint'])


if __name__=='__main__':unittest.main()
