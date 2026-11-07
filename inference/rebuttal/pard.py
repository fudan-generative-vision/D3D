"""Target-independent PARD-2 proposals in the same packed execution engine.

PARD uses causal masked AR prediction. The adapter preserves its attention
pattern and shifts next-token logits into D3D's same-position interface. The
checkpoint backbone is unchanged; the train-only target projection is unused.
"""
import torch
from inference.model.modeling_d3d import D3DForCausalLM


class PardIndependentForD3D(D3DForCausalLM):
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                logits_to_keep=0, **kwargs):
        if input_ids is None:
            raise ValueError("The PARD inference adapter expects token IDs")
        n = input_ids.shape[1]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        row = torch.arange(n, device=input_ids.device)[:, None] + past
        column = torch.arange(past + n, device=input_ids.device)[None]
        causal = column <= row
        if attention_mask is None:
            attention_mask = causal[None, None]
        elif attention_mask.dtype == torch.bool:
            attention_mask = attention_mask & causal
        else:
            attention_mask = attention_mask.masked_fill(~causal, -torch.inf)
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, logits_to_keep=0, **kwargs)
        # Packed drafting also sets store_kv=True to commit the pending prefix.
        # Logit alignment must therefore be independent of the cache-write flag.
        shifted = torch.roll(output.logits, shifts=1, dims=1)
        # Each segment's first position is an anchor, replaced by the caller.
        shifted[:, 0] = 0
        output.logits = shifted
        if isinstance(logits_to_keep, int):
            if logits_to_keep:
                output.logits = output.logits[:, -logits_to_keep:]
        else:
            output.logits = output.logits[:, logits_to_keep]
        return output
