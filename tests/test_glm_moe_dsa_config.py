# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.transformer.transformer_config import MLATransformerConfig

from mcore_bridge.model.register import get_model_meta


def test_glm_moe_dsa_config_exposes_native_mla_options(monkeypatch):
    meta = get_model_meta('glm_moe_dsa')
    # Weight conversion needs process groups; configuration construction does not.
    monkeypatch.setattr(meta, 'bridge_cls', lambda config: None)
    config = meta.config_cls(
        mcore_model_type='glm_moe_dsa',
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        multi_latent_attention=True,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_head_dim=8,
        qk_pos_emb_head_dim=8,
        v_head_dim=8,
    )

    assert config.mla_proj_use_quantization is True
    assert isinstance(config, MLATransformerConfig)
