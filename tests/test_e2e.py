"""TP1/TP2 end-to-end tests for greedy, sampled, and mixed DSPINE requests.
Requires GPUs, the installed plugin, and a target/draft model pair:
    DSPINE_TARGET_MODEL=<target> DSPINE_DRAFT_MODEL=<draft> pytest tests/test_e2e.py
Skipped when either model environment variable is unset.
"""

import os
import unittest

import pytest
import requests
from transformers import AutoConfig

TARGET_MODEL = os.environ.get("DSPINE_TARGET_MODEL")
DRAFT_MODEL = os.environ.get("DSPINE_DRAFT_MODEL")
if not (TARGET_MODEL and DRAFT_MODEL):
    pytest.skip(
        "set DSPINE_TARGET_MODEL and DSPINE_DRAFT_MODEL", allow_module_level=True
    )

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

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
    tp_size = 1

    @classmethod
    def setUpClass(cls):
        if not (TARGET_MODEL and DRAFT_MODEL):
            raise unittest.SkipTest("set DSPINE_TARGET_MODEL and DSPINE_DRAFT_MODEL")
        cls.base_url = DEFAULT_URL_FOR_TEST
        model_config = AutoConfig.from_pretrained(TARGET_MODEL, trust_remote_code=True)
        text_config = getattr(model_config, "text_config", model_config)
        other_args = [
            "--speculative-algorithm",
            "DSPINE",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-num-draft-tokens",
            "16",
            "--tp-size",
            str(cls.tp_size),
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fa3",
            "--speculative-draft-attention-backend",
            "triton",
            "--mem-fraction-static",
            "0.7",
            "--context-length",
            "8192",
            "--max-running-requests",
            "8",
            "--cuda-graph-max-bs-decode",
            "8",
            "--min-free-slots-delay",
            "1",
            "--sampling-backend",
            "flashinfer",
            "--trust-remote-code",
        ]
        if getattr(text_config, "linear_num_value_heads", None):
            other_args.extend(
                [
                    "--linear-attn-backend",
                    "triton",
                    "--enable-linear-replayssm-spec",
                    "--mamba-ssm-dtype",
                    "float32",
                    "--mamba-radix-cache-strategy",
                    "extra_buffer",
                    "--max-mamba-cache-size",
                    "32",
                ]
            )
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
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


class TestDSpineTP2Server(TestDSpineServer):
    tp_size = 2


if __name__ == "__main__":
    unittest.main()
