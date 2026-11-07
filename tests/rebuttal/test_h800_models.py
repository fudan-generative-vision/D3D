"""Small CPU parity checks for the inference families and PARD adapter."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')

import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from inference.model.configuration_d3d import D3DConfig
from inference.model.modeling_d3d import D3DForCausalLM
from inference.rebuttal.pard import PardIndependentForD3D


class InferenceParityTests(unittest.TestCase):
    def source(self, family):
        torch.manual_seed(17)
        config_class, model_class = (LlamaConfig, LlamaForCausalLM) if family == 'llama' else (Qwen3Config, Qwen3ForCausalLM)
        config = config_class(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
        config._attn_implementation = 'sdpa'
        original = model_class(config).eval()
        adapted_config = D3DConfig(**(config.to_dict() | {'use_qk_norm': family == 'qwen3', 'head_dim': 8}))
        adapted_config._attn_implementation = 'sdpa'
        return original, adapted_config

    def test_qwen_and_llama_causal_logits_and_cached_continuation(self):
        ids = torch.tensor([[1, 5, 11, 9, 2, 4]])
        for family in ('qwen3', 'llama'):
            with self.subTest(family=family), torch.no_grad():
                source, config = self.source(family)
                adapted = D3DForCausalLM(config).eval()
                adapted.load_state_dict(source.state_dict(), strict=True)
                expected = source(ids, use_cache=False).logits
                mask = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
                actual = adapted(ids, attention_mask=mask, use_cache=False, draft_causal=True).logits
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                cache = DynamicCache()
                adapted(ids[:, :4], attention_mask=mask[:, :, :4, :4], past_key_values=cache,
                        use_cache=True, store_kv=True, draft_causal=True)
                actual = adapted(ids[:, 4:], attention_mask=mask[:, :, 4:, :], past_key_values=cache,
                                 use_cache=True, store_kv=False, draft_causal=True).logits
                torch.testing.assert_close(actual, expected[:, 4:], atol=1e-5, rtol=1e-5)

    def test_pard_packed_independent_proposals_match_causal_backbone(self):
        source, config = self.source('qwen3')
        adapted = PardIndependentForD3D(config).eval()
        adapted.load_state_dict(source.state_dict(), strict=True)
        prefix = torch.tensor([[1, 2, 3]])
        queries = torch.tensor([[4, 63, 63, 63, 5, 63, 63, 63]])
        mask = torch.zeros(8, 11, dtype=torch.bool)
        mask[:, :3] = True
        mask[:4, 3:7] = True
        mask[4:, 7:] = True
        positions = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]])
        for store_kv in (False, True):
            cache = DynamicCache()
            with torch.no_grad():
                adapted(prefix, past_key_values=cache, use_cache=True, store_kv=True)
                actual = adapted(queries, attention_mask=mask[None, None], position_ids=positions,
                                 past_key_values=cache, use_cache=True, store_kv=store_kv).logits
                for start in (0, 4):
                    expected = source(torch.cat([prefix, queries[:, start:start + 4]], dim=1)).logits[:, 3:6]
                    torch.testing.assert_close(actual[:, start + 1:start + 4], expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
