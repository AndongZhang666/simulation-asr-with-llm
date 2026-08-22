import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_whisper_nbest_one_matches_stock_decoder() -> None:
    if os.environ.get("ASR_EC_RUN_WHISPER_INTEGRATION") != "1":
        pytest.skip("set ASR_EC_RUN_WHISPER_INTEGRATION=1 to run checkpoint-backed parity")
    model_root = os.environ.get("WHISPER_TEST_MODEL_ROOT")
    audio_path = os.environ.get("WHISPER_TEST_AUDIO")
    if not model_root or not audio_path:
        pytest.skip("WHISPER_TEST_MODEL_ROOT and WHISPER_TEST_AUDIO are required")
    if not Path(audio_path).is_file():
        pytest.skip(f"test audio is unavailable: {audio_path}")

    import torch
    import whisper
    from whisper.decoding import DecodingOptions, decode

    from asr_ec.asr.whisper_nbest import decode_nbest

    model = whisper.load_model("small.en", download_root=model_root, device="cpu")
    audio = whisper.pad_or_trim(whisper.load_audio(audio_path))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = DecodingOptions(
        language="en",
        task="transcribe",
        beam_size=1,
        temperature=0,
        fp16=False,
        without_timestamps=True,
    )

    stock = decode(model, mel, options)
    nbest = decode_nbest(model, mel.unsqueeze(0), options)

    assert len(nbest) == 1
    assert len(nbest[0].candidates) == 1
    candidate = nbest[0].candidates[0]
    assert candidate.text == stock.text
    assert candidate.token_ids == tuple(stock.tokens)
    assert candidate.sequence_logscore == pytest.approx(stock.avg_logprob * (len(stock.tokens) + 1))
    assert candidate.token_logprobs_available is False
    assert torch.isfinite(torch.tensor(candidate.sequence_logscore))
