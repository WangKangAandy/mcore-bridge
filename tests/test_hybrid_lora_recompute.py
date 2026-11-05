# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU regression for frozen HybridModel inputs and full recomputation.

Use the real bridge constructor/forward, Core HybridModel.forward, mHC expansion
and Core checkpointed_forward. Replace only heavyweight Core construction with
tiny modules and accelerator RNG bookkeeping with CPU RNG bookkeeping. Pipeline
input routing is simulated; this is not a distributed or Flash-kernel test.
"""
import pytest
import torch
from contextlib import contextmanager, nullcontext
from megatron.core.models.hybrid.hybrid_model import HybridModel as CoreModel
from megatron.core.recompute import checkpointed_forward
from megatron.core.tensor_parallel import random as core_random
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from types import SimpleNamespace

from mcore_bridge.model.hybrid_model import HybridModel


class _Embedding(torch.nn.Embedding):

    def forward(self, input_ids, position_ids):
        return super().forward(input_ids).transpose(0, 1)


class _LoRALayer(torch.nn.Module):
    supports_hybrid_recompute_kwargs = True

    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(4, 8) * 0.1)
        self.B = torch.nn.Parameter(torch.zeros(8, 4))

    def forward(self, hidden_states, **kwargs):
        return hidden_states + 2 * hidden_states @ self.A.T @ self.B.T, None


class _Decoder(torch.nn.Module):

    def __init__(self, config, pre_process):
        super().__init__()
        self.config = config
        self.pre_process = pre_process
        self.layers = torch.nn.ModuleList([_LoRALayer()])
        self.num_layers_per_pipeline_rank = 1
        self.pg_collection = SimpleNamespace(tp=None)
        self.input_tensor = None
        self.observed = []

    def forward(self, hidden_states, **kwargs):
        x = hidden_states if self.pre_process else self.input_tensor
        self.observed.append(x.requires_grad)
        x = HyperConnectionModule.input_expand(x, 4)
        if self.training and self.config.recompute_granularity == 'full':
            return checkpointed_forward(self, x, None, None, None, None, None, None, False)
        return self.layers[0](x)[0]


def _init_core(self, config, layer_spec, vocab_size, max_seq, **kwargs):
    torch.nn.Module.__init__(self)
    self.config = config
    self.pre_process = kwargs['pre_process']
    self.post_process = False
    self.position_embedding_type = 'none'
    self.share_embeddings_and_output_weights = False
    self.mtp_process = False
    self.embedding = _Embedding(8, 2)
    self.embedding.requires_grad_(False)
    self.decoder = _Decoder(config, self.pre_process)


@pytest.fixture(autouse=True)
def cpu_core(monkeypatch):
    # Preserve the surrounding suite's CPU RNG, and restore all patched helpers.
    state = torch.get_rng_state()

    @contextmanager
    def fork_rng():
        saved = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(saved)

    monkeypatch.setattr(CoreModel, '__init__', _init_core)
    monkeypatch.setattr(core_random, '_get_all_rng_states', lambda: (torch.get_rng_state(), ))
    monkeypatch.setattr(core_random, '_set_all_rng_states', lambda saved: torch.set_rng_state(saved))
    monkeypatch.setattr(core_random, '_fork_rng', fork_rng)
    yield
    torch.set_rng_state(state)


def _model(mode, path='embedding', guard=True):
    torch.manual_seed(57)
    config = SimpleNamespace(
        padded_vocab_size=8,
        tensor_model_parallel_size=1,
        hybrid_layer_pattern='K',
        mtp_num_layers=0,
        max_position_embeddings=8,
        untie_embeddings_and_output_weights=True,
        position_embedding_type='none',
        rotary_base=10000,
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        sequence_parallel=False,
        cuda_graph_impl='none',
        moe_num_hash_layers=0,
        freeze_base_model_for_mtp=False,
        enable_mhc_connections=True,
        recompute_granularity=mode,
        fp8=None,
        fp4=None,
        distribute_saved_activations=False,
        recompute_method='uniform',
        recompute_num_layers=1)
    model = HybridModel(config, None, pre_process=path != 'pipeline', post_process=False)
    if not guard:
        # Negative control: remove the newly registered guard, reproducing the
        # old path without changing checkpoint or the layer's trainable weights.
        model.decoder._forward_pre_hooks.clear()
    return model


def _forward(model, path):
    ids = torch.tensor([[0, 1, 2]])
    kwargs = {}
    if path == 'explicit':
        kwargs['decoder_input'] = model.embedding(ids, ids).detach()
    if path == 'pipeline':
        model.decoder.input_tensor = model.embedding(ids, ids).detach()
    return model(ids, ids, **kwargs)


def _step(mode, path='embedding', train_embedding=False):
    model = _model(mode, path)
    model.embedding.requires_grad_(train_embedding)
    embedding_before = model.embedding.weight.detach().clone()
    output = _forward(model, path)
    output.square().mean().backward()
    grad = model.decoder.layers[0].B.grad.clone()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    if not train_embedding:
        assert model.embedding.weight.grad is None
        torch.testing.assert_close(model.embedding.weight, embedding_before, rtol=0, atol=0)
    assert model.decoder.layers[0].B.detach().abs().sum() > 0
    return model, output.detach(), grad


@pytest.mark.parametrize('path', ['embedding', 'explicit', 'pipeline'])
def test_full_recompute_matches_no_recompute(path):
    broken = _model('full', path, guard=False)
    assert not _forward(broken, path).requires_grad
    assert broken.decoder.observed == [False]
    reference, ref_output, ref_grad = _step(None, path)
    fixed, output, grad = _step('full', path)
    assert reference.decoder.observed == [False]
    assert fixed.decoder.observed == [True]
    torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


def test_existing_embedding_graph_is_preserved():
    reference, _, _ = _step(None, train_embedding=True)
    fixed, _, _ = _step('full', train_embedding=True)
    assert fixed.embedding.weight.grad is not None
    torch.testing.assert_close(fixed.embedding.weight.grad, reference.embedding.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize('path', ['embedding', 'explicit', 'pipeline'])
@pytest.mark.parametrize('mode', ['eval', 'no_grad', 'none'])
def test_guard_leaves_other_execution_modes_unchanged(path, mode):
    model = _model(None if mode == 'none' else 'full', path)
    if mode == 'eval':
        model.eval()
    with torch.no_grad() if mode == 'no_grad' else nullcontext():
        _forward(model, path)
    assert model.decoder.observed == [False]


def test_guard_does_not_change_parameters_or_state_dict():
    fixed, reference = _model('full'), _model('full', guard=False)
    assert list(fixed.state_dict()) == list(reference.state_dict())
    assert len(fixed.decoder._forward_pre_hooks) == 1
    for key in fixed.state_dict():
        torch.testing.assert_close(fixed.state_dict()[key], reference.state_dict()[key], rtol=0, atol=0)
