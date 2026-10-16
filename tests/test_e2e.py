"""End-to-end GPU tests for greedy, sampled, and mixed DSPINE requests.
Requires the installed plugin and a target/draft model pair:
    DSPINE_TARGET_MODEL=<target> DSPINE_DRAFT_MODEL=<draft> pytest tests/test_e2e.py
Skipped when either model environment variable is unset.
"""

import os
import unittest

import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TARGET_MODEL = os.environ.get("DSPINE_TARGET_MODEL")
DRAFT_MODEL = os.environ.get("DSPINE_DRAFT_MODEL")

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "List three prime numbers larger than 100 and explain how you checked them.",
]
GREEDY = {"temperature": 0.0}
SAMPLED = [
    {"temperature": 1.0},
    {"temperature": 0.7, "top_k": 20, "top_p": 0.9, "min_p": 0.05},
]


class TestDSpineServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (TARGET_MODEL and DRAFT_MODEL):
            raise unittest.SkipTest("set DSPINE_TARGET_MODEL and DSPINE_DRAFT_MODEL")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "DSPINE",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, sampling_params):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": PROMPTS,
                "sampling_params": [
                    {"max_new_tokens": 128, **params} for params in sampling_params
                ],
            },
            timeout=600,
        )
        response.raise_for_status()
        outputs = response.json()
        self.assertEqual(len(outputs), len(PROMPTS))
        return outputs

    def _assert_speculates(self, outputs):
        for output in outputs:
            self.assertGreater(len(output["text"]), 0)
        tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
        verifies = sum(o["meta_info"]["spec_verify_ct"] for o in outputs)
        self.assertGreater(verifies, 0)
        print(
            f"{tokens} tokens over {verifies} verifies: {tokens / verifies:.2f} per verify"
        )
        # Every verify commits at least one token; an accepted draft commits more.
        self.assertGreater(tokens / verifies, 1.0)

    def test_server_info(self):
        response = requests.get(self.base_url + "/server_info", timeout=10)
        response.raise_for_status()
        info = response.json()
        self.assertEqual(info["speculative_algorithm"], "DSPINE")

    def test_greedy_requests(self):
        self._assert_speculates(self._generate([GREEDY] * len(PROMPTS)))

    def test_sampled_requests(self):
        for params in SAMPLED:
            with self.subTest(**params):
                self._assert_speculates(self._generate([params] * len(PROMPTS)))

    def test_greedy_and_sampled_requests_share_a_batch(self):
        mixed = [GREEDY, SAMPLED[1], GREEDY, SAMPLED[0]]
        self._assert_speculates(self._generate(mixed))


if __name__ == "__main__":
    unittest.main()
