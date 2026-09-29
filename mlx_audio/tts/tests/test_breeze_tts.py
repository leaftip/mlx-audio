import inspect
import threading

import pytest

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
except (ImportError, RuntimeError) as exc:  # pragma: no cover - no Metal host
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_audio.tts.models.breeze_tts.breeze_tts import (
    Model,
    _split_long_segment,
    _t5gemma2_attention_mask,
)
from mlx_audio.tts.models.breeze_tts.config import ModelConfig
from mlx_audio.tts.utils import get_model_and_args
from mlx_audio.utils import get_model_name_parts


def tiny_config():
    return ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_codebooks=3,
        vocab_size=8,
        text_vocab_size=32,
        text_encoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
        },
        depth_decoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-5,
            "num_codebooks": 3,
            "vocab_size": 8,
            "audio_embed_size": 16,
        },
    )


def test_config_reads_breeze_nested_sample_rate():
    config = ModelConfig.from_dict(
        {
            "model_type": "breeze",
            "codec_config": {"sampling_rate": 24000},
            "backbone_config": {
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "max_position_embeddings": 40960,
            },
        }
    )
    assert config.model_type == "breeze"
    assert config.sample_rate == 24000
    assert config.backbone_config["rms_norm_eps"] == 1e-6


def test_backbone_uses_nested_qwen_config_not_top_level_rope_settings():
    config = tiny_config()
    config.rms_norm_eps = 1e-5
    config.rope_theta = 500000
    config.max_position_embeddings = 2048
    config.rope_scaling = {"rope_type": "llama3", "factor": 32.0}
    config.backbone_config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "rope_scaling": None,
        "max_position_embeddings": 40960,
    }

    model = Model(config)
    assert model.backbone_model.args.rms_norm_eps == 1e-6
    assert model.backbone_model.args.rope_theta == 1_000_000
    assert model.backbone_model.args.max_position_embeddings == 40960
    assert model.backbone_model.args.rope_scaling is None


def test_t5gemma2_masks_are_bidirectional_with_symmetric_local_window():
    assert _t5gemma2_attention_mask(5, "full_attention", None) is None
    mask = _t5gemma2_attention_mask(5, "sliding_attention", 4)
    assert mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
    ]


def test_depth_decoder_predicts_each_remaining_codebook():
    model = Model(tiny_config())
    logits = model.depth_decoder.next_logits(
        mx.array([[0, 1]], dtype=mx.int32), mx.zeros((1, 16))
    )
    assert logits.shape == (1, 8)


def test_sanitize_keeps_main_checkpoint_weights_only():
    weights = {
        "lm_head.weight": mx.zeros((9, 16)),
        "text_encoder.layers.0.pre_self_attn_layernorm.weight": mx.zeros((16,)),
        "codec_model.decoder.weight": mx.zeros((1,)),
        "codec_model.quantizer.initialized": mx.zeros((1,)),
        "backbone_model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((1,)),
    }
    sanitized = Model.sanitize(weights)
    assert list(sanitized) == [
        "lm_head.weight",
        "text_encoder.layers.0.pre_self_attn_layernorm.weight",
    ]


def test_codec_key_validation_allows_only_generated_quantizer_flags():
    expected = {
        "decoder.weight",
        "encoder_model.quantizer.rvq_first.vq.layers.0.codebook.initialized",
    }
    Model._validate_codec_weight_keys(expected, {"decoder.weight"})
    with pytest.raises(ValueError, match="unexpected"):
        Model._validate_codec_weight_keys(expected, {"decoder.weight", "other.weight"})
    with pytest.raises(ValueError, match="missing"):
        Model._validate_codec_weight_keys(expected, set())


def test_default_generic_voice_maps_to_breeze_s0():
    model = Model(tiny_config())
    assert model._speaker("af_heart") == "[S0]"
    assert model._speaker("S2") == "[S2]"
    assert model._speaker("[S3]") == "[S3]"


def test_depth_cfg_applies_to_every_remaining_codebook(monkeypatch):
    model = Model(tiny_config())
    sampled_logits = []

    def next_logits(_token_ids, hidden, cache=None):
        # Row 0 is the conditional prompt, row 1 the unconditional one.
        return mx.broadcast_to(hidden[:, :1], (hidden.shape[0], 8))

    def sample_array(logits, **_kwargs):
        sampled_logits.append(logits)
        return mx.array([1], dtype=mx.int32)

    monkeypatch.setattr(model.depth_decoder, "next_logits", next_logits)
    monkeypatch.setattr(model, "_sample_array", sample_array)
    tokens = model._depth_tokens(
        1,
        mx.ones((1, 16)),
        unconditional_hidden=mx.zeros((1, 16)),
        cfg_scale=3.0,
        temperature=0,
        top_p=1,
        top_k=0,
    )

    assert tokens == [1, 1, 1]
    assert len(sampled_logits) == 2
    assert all(float(logits[0, 0].item()) == 3.0 for logits in sampled_logits)


def test_depth_cfg_masks_reserved_tokens_at_every_step(monkeypatch):
    model = Model(tiny_config())
    sampled_logits = []

    def next_logits(_token_ids, hidden, cache=None):
        return mx.broadcast_to(
            mx.arange(8, dtype=mx.float32)[None, :], (hidden.shape[0], 8)
        )

    def sample_array(logits, **_kwargs):
        sampled_logits.append(logits)
        return mx.array([1], dtype=mx.int32)

    monkeypatch.setattr(model.depth_decoder, "next_logits", next_logits)
    monkeypatch.setattr(model, "_sample_array", sample_array)
    model._depth_tokens(
        1,
        mx.ones((1, 16)),
        unconditional_hidden=mx.zeros((1, 16)),
        cfg_scale=2.0,
        temperature=0,
        top_p=1,
        top_k=0,
    )

    assert len(sampled_logits) == 2
    for logits in sampled_logits:
        assert logits[0, :5].tolist() == [0, 1, 2, 3, 4]
        assert all(value == float("-inf") for value in logits[0, 5:].tolist())


def test_repetition_penalty_is_sign_aware():
    logits = mx.array([[4.0, -2.0, 7.0, -5.0]])
    penalized = Model._apply_repetition_penalty(logits, [0, 1, 1, 3], 2.0)
    assert penalized.tolist() == [[2.0, -4.0, 7.0, -10.0]]


def test_codec_vocab_boundary_comes_from_config():
    config = tiny_config()
    config.codec_config = {"codebook_size": 5}
    model = Model(config)
    token = model._sample(
        mx.array([[0.0, 0.0, 1.0, 0.0, 2.0, 99.0, 98.0, 97.0, 96.0]]),
        temperature=0,
        top_p=1,
        top_k=0,
    )
    assert token == 4


def test_reference_audio_batch_is_rejected_before_codec_encode():
    model = Model(tiny_config())
    model.audio_tokenizer = object()
    with pytest.raises(ValueError, match="exactly one reference audio"):
        model._encode_reference(mx.zeros((2, 10)))


def test_multiple_reference_paths_are_rejected_before_prompt_encoding():
    model = Model(tiny_config())
    with pytest.raises(ValueError, match="exactly one reference audio"):
        model._prompt_embeddings(
            "target",
            voice=None,
            instruct=None,
            ref_audio=["a.wav", "b.wav"],
            ref_text=["a", "b"],
        )


def test_exact_prompt_template_for_cloning_and_direction(monkeypatch):
    model = Model(tiny_config())
    prompts = []

    def text_ids(value):
        prompts.append(value)
        return mx.array([1, 2], dtype=mx.int32)

    monkeypatch.setattr(model, "_text_ids", text_ids)
    monkeypatch.setattr(
        model, "_encode_reference", lambda _audio: mx.zeros((1, 1, 3), mx.int32)
    )
    model._prompt_embeddings(
        "target",
        voice="S2",
        instruct="warm",
        ref_audio="reference.wav",
        ref_text="reference text",
    )
    assert prompts == ["[S2]reference text", "[S2]<ins_bos>warm<ins_eos>target"]


def test_reference_is_encoded_once_per_clip_and_again_when_the_clip_changes(
    monkeypatch, tmp_path
):
    model = Model(tiny_config())
    encoded = []
    texts = []

    def text_ids(value):
        texts.append(value)
        return mx.array([1, 2], dtype=mx.int32)

    def encode(audio):
        encoded.append(str(audio))
        return mx.ones((1, 4, model.num_codebooks), mx.int32)

    monkeypatch.setattr(model, "_text_ids", text_ids)
    monkeypatch.setattr(model, "_encode_reference", encode)
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"first take")

    def prompt(text, audio=clip, ref_text="reference text", voice="S0"):
        return model._prompt_embeddings(
            text, voice=voice, instruct=None, ref_audio=audio, ref_text=ref_text
        )

    first = prompt("one")
    second = prompt("two")
    assert len(encoded) == 1  # the clip went through the codec once
    assert texts == ["[S0]reference text", "[S0]one", "[S0]two"]
    assert first.shape == second.shape
    assert mx.array_equal(first[:, :-2], second[:, :-2])  # the same reference part

    # The cached prompt is the prompt that is built without a cache.
    uncached = Model(tiny_config())
    uncached.update(model.parameters())
    monkeypatch.setattr(uncached, "_text_ids", lambda value: mx.array([1, 2], mx.int32))
    monkeypatch.setattr(uncached, "_encode_reference", encode)
    monkeypatch.setattr(Model, "_reference_key", staticmethod(lambda *a: None))
    plain = uncached._prompt_embeddings(
        "two", voice="S0", instruct=None, ref_audio=clip, ref_text="reference text"
    )
    monkeypatch.undo()
    monkeypatch.setattr(model, "_text_ids", text_ids)
    monkeypatch.setattr(model, "_encode_reference", encode)
    assert mx.allclose(plain, second)

    # Another transcript, another speaker tag, or a changed file is another reference.
    count = len(encoded)
    prompt("three", ref_text="other words")
    prompt("three", voice="S1")
    assert len(encoded) == count + 2
    clip.write_bytes(b"a second, longer take")
    prompt("four")
    assert len(encoded) == count + 3
    prompt("five")
    assert len(encoded) == count + 3

    # An array, or a file that is not there, is never cached.
    prompt("six", audio=mx.zeros((1, 1, 10)))
    prompt("six", audio=mx.zeros((1, 1, 10)))
    prompt("seven", audio=tmp_path / "missing.wav")
    prompt("seven", audio=tmp_path / "missing.wav")
    assert len(encoded) == count + 7

    # The cache is bounded, and it is not one of the model's parameters.
    for i in range(Model.REFERENCE_CACHE_SIZE + 3):
        other = tmp_path / f"voice{i}.wav"
        other.write_bytes(b"x" * (i + 1))
        prompt("eight", audio=other)
    assert len(model._reference_cache) == Model.REFERENCE_CACHE_SIZE
    names = [name for name, _ in tree_flatten(model.parameters())]
    assert not any("reference_cache" in name for name in names)


def test_stream_flushes_at_exact_interval_and_resets_state(monkeypatch):
    model = Model(tiny_config())

    class Decoder:
        decode_upsample_rate = 24000

        def __init__(self):
            self.reset_calls = 0

        def reset_streaming_state(self):
            self.reset_calls += 1

        def streaming_step(self, codes):
            return mx.zeros((1, 1, codes.shape[-1]))

    class AudioTokenizer:
        decode_upsample_rate = 24000

        def __init__(self):
            self.decoder = Decoder()

    class Head:
        def __init__(self):
            self.calls = 0

        def __call__(self, _hidden):
            self.calls += 1
            logits = mx.full((1, 9), -100.0)
            logits[..., 1 if self.calls <= 3 else 8] = 100.0
            return logits

    model.audio_tokenizer = AudioTokenizer()
    monkeypatch.setattr(
        model, "_prompt_embeddings", lambda *_args, **_kwargs: mx.zeros((1, 1, 16))
    )
    model.lm_head = Head()

    chunks = list(
        model.generate(
            "test", temperature=0, top_k=0, stream=True, streaming_interval=2.0
        )
    )
    assert len(chunks) == 2
    assert chunks[0].token_count == 2
    assert chunks[0].prompt["tokens"] == 2
    assert chunks[0].prompt["tokens-per-sec"] >= 0
    assert chunks[0].audio_samples["samples-per-sec"] >= 0
    assert not chunks[0].is_final_chunk
    assert chunks[1].token_count == 1
    assert chunks[1].prompt["tokens"] == 1
    assert chunks[1].prompt["tokens-per-sec"] >= 0
    assert chunks[1].audio_samples["samples-per-sec"] >= 0
    assert chunks[1].is_final_chunk
    assert model.audio_tokenizer.decoder.reset_calls == 2


def test_early_eos_yields_silent_result_instead_of_raising(monkeypatch):
    model = Model(tiny_config())

    class Decoder:
        decode_upsample_rate = 24000

        def reset_streaming_state(self):
            pass

    class AudioTokenizer:
        decode_upsample_rate = 24000
        decoder = Decoder()

        def decode(self, codes):
            assert codes.shape == (1, 1, 3)
            return mx.zeros((1, 4)), mx.array([4], dtype=mx.int32)

    class Backbone:
        def make_cache(self):
            return []

        def __call__(self, input_embeddings=None, input_ids=None, cache=None):
            del cache
            length = (
                input_embeddings.shape[1]
                if input_embeddings is not None
                else input_ids.shape[1]
            )
            return mx.zeros((1, length, 16))

    class Head:
        def __call__(self, _hidden):
            logits = mx.full((1, 9), -100.0)
            logits[..., 8] = 100.0
            return logits

    model.audio_tokenizer = AudioTokenizer()
    model.backbone_model = Backbone()
    model.lm_head = Head()
    monkeypatch.setattr(
        model, "_prompt_embeddings", lambda *_args, **_kwargs: mx.zeros((1, 1, 16))
    )

    results = list(model.generate("test", temperature=0, top_k=0))
    assert len(results) == 1
    assert results[0].is_final_chunk
    assert results[0].token_count == 0
    assert results[0].samples == 4
    assert results[0].prompt == {"tokens": 0, "tokens-per-sec": 0.0}
    assert results[0].audio_samples["samples"] == 4
    assert results[0].audio_samples["samples-per-sec"] >= 0
    assert results[0].audio.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_breeze_registry_loads_model_module():
    module, model_type = get_model_and_args(
        "breeze", get_model_name_parts("BreezeBlue/breeze-tts-2")
    )
    assert model_type == "breeze_tts"
    assert module.Model is Model


def test_breeze_generate_splits_segments(monkeypatch):
    model = Model(tiny_config())

    class Decoder:
        decode_upsample_rate = 24000

        def reset_streaming_state(self):
            pass

    class AudioTokenizer:
        decode_upsample_rate = 24000
        decoder = Decoder()

        def decode(self, codes):
            return mx.zeros((1, 4)), mx.array([4], dtype=mx.int32)

    class Backbone:
        def make_cache(self):
            return []

        def __call__(self, input_embeddings=None, input_ids=None, cache=None):
            del cache
            length = (
                input_embeddings.shape[1]
                if input_embeddings is not None
                else input_ids.shape[1]
            )
            return mx.zeros((1, length, 16))

    class Head:
        def __call__(self, _hidden):
            logits = mx.full((1, 9), -100.0)
            logits[..., 8] = 100.0
            return logits

    model.audio_tokenizer = AudioTokenizer()
    model.backbone_model = Backbone()
    model.lm_head = Head()
    prompt_calls = []
    monkeypatch.setattr(
        model,
        "_prompt_embeddings",
        lambda text, *args, **kwargs: prompt_calls.append(text) or mx.zeros((1, 1, 16)),
    )

    results = list(model.generate("Segment 1\n\nSegment 2", temperature=0, top_k=0))
    assert len(results) == 2
    assert prompt_calls == ["Segment 1", "Segment 2"]
    assert results[0].segment_idx == 0
    assert not results[0].is_final_chunk
    assert results[1].segment_idx == 1
    assert results[1].is_final_chunk


@pytest.mark.parametrize(
    "text",
    [
        "a" * 1201,
        "word " * 300,
        "这是第一句。这是第二句！这是第三句？" * 50,
    ],
)
def test_breeze_long_segments_are_bounded(text):
    chunks = _split_long_segment(text)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(len(chunk) <= 600 for chunk in chunks)


def test_breeze_generate_preserves_existing_positional_arguments():
    signature = inspect.signature(Model.generate)
    bound = signature.bind(
        None,
        "text",
        "voice",
        "instruction",
        "reference.wav",
        "reference text",
        2.0,
        321,
        0.4,
        0.8,
        7,
        1.2,
        123,
        True,
        0.5,
    )

    assert bound.arguments["max_tokens"] == 321
    assert bound.arguments["temperature"] == 0.4
    assert bound.arguments["streaming_interval"] == 0.5
    assert "split_pattern" not in bound.arguments


def test_sampler_varies_on_worker_threads_and_is_reproducible_with_a_key():
    """mlx-lm's compiled sampler freezes on non-main threads; ours must not."""
    model = Model(tiny_config())
    logits = mx.log(mx.array([[0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2]]))
    mx.eval(logits)  # the first evaluation in a process must happen on the main thread

    def draws(sampler, n=12):
        return [
            int(model._sample_array(logits, sampler=sampler, valid=8).item())
            for _ in range(n)
        ]

    results = {}

    def body():
        results["a"] = draws(model._make_sampler(0.9, 1.0, 4, 8))
        results["b"] = draws(model._make_sampler(0.9, 1.0, 4, 8))

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    assert (
        len(set(results["a"])) > 1
    ), "sampler returned one token forever on a worker thread"
    assert (
        results["a"] != results["b"]
    ), "two unseeded samplers drew identical sequences"

    seeded_1 = draws(model._make_sampler(0.9, 1.0, 4, 8, key=mx.random.key(3)))
    seeded_2 = draws(model._make_sampler(0.9, 1.0, 4, 8, key=mx.random.key(3)))
    assert seeded_1 == seeded_2
    assert max(seeded_1) <= 7 and all(
        t in (3, 7, 2, 6) for t in seeded_1
    )  # top-4 of the distribution only
