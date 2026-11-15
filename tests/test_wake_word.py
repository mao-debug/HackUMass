"""Optional KWS is exercised through fake engines and local placeholder files."""

import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import threading

import numpy as np
import pytest

from vidi.audio.wake_word import MODEL_FILES, MAX_PCM_BYTES, WakeDetector, WakeWordError, tokenize_keyword


@pytest.fixture
def model(tmp_path):
    for filename in [*MODEL_FILES.values(), "bpe.model"]:
        (tmp_path / filename).write_bytes(b"placeholder")
    keyword = tmp_path / "keywords.txt"
    keyword.write_text("TEST TOKEN\n")
    return tmp_path, keyword


class Stream:
    def __init__(self):
        self.waveforms = []

    def accept_waveform(self, rate, samples):
        self.waveforms.append((rate,samples.copy()))


class Engine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stream = Stream()
        self.ready = True
        self.result = "HEY VIDI"
        self.resets = 0

    def create_stream(self):
        return self.stream

    def is_ready(self, stream):
        return self.ready

    def decode_stream(self, stream):
        self.ready = False

    def get_result(self, stream):
        return self.result

    def reset_stream(self, stream):
        self.resets += 1


def test_constructor_is_lazy_and_pcm_is_signed_normalized(model):
    directory, keyword = model
    made = []

    def factory(**kwargs):
        made.append(Engine(**kwargs))
        return made[-1]

    detector = WakeDetector(directory, keyword, engine_factory=factory, threshold=.3)
    assert not made and detector.accept(b"") is False
    assert detector.accept(struct.pack("<hhh",-32768,0,32767)) is True
    engine = made[0]
    assert engine.kwargs == {
        **{name:str(directory / filename) for name,filename in MODEL_FILES.items()},
        "keywords_file":str(keyword), "num_threads":1, "sample_rate":16000,
        "provider":"cpu", "keywords_threshold":.3,
    }
    rate, samples = engine.stream.waveforms[0]
    assert rate == 16000 and samples.dtype == np.float32
    np.testing.assert_array_equal(samples, [-1.,0.,32767/32768])
    assert engine.resets == 1
    detector.reset()
    assert engine.resets == 2
    detector.close()
    detector.close()
    with pytest.raises(WakeWordError, match="closed"):
        detector.accept(b"\0\0")


def test_missing_model_fails_before_engine_creation(tmp_path):
    detector = WakeDetector(tmp_path,tmp_path/"missing",engine_factory=lambda **k: pytest.fail("No engine"))
    with pytest.raises(WakeWordError,match="missing"):
        detector.accept(b"\0\0")


def test_prepare_loads_once_before_audio_and_close_disables_it(model):
    made = []
    detector = WakeDetector(*model, engine_factory=lambda **kwargs: made.append(Engine(**kwargs)) or made[-1])
    detector.prepare()
    detector.prepare()
    assert len(made) == 1 and not made[0].stream.waveforms
    detector.close()
    with pytest.raises(WakeWordError, match="closed"):
        detector.prepare()


def test_close_request_does_not_wait_on_inference_and_releases_on_return(model):
    engine = Engine()
    entered, release = threading.Event(), threading.Event()

    def waveform(*args):
        entered.set()
        assert release.wait(1)

    engine.stream.accept_waveform = waveform
    detector = WakeDetector(*model,engine_factory=lambda **kwargs:engine)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(detector.accept,b"\0\0")
        assert entered.wait(1)
        try:
            detector.request_close()
            assert not task.done()
        finally:
            release.set()
        task.result(1)
    assert detector._closed and detector._engine is None and detector._stream is None
    with pytest.raises(WakeWordError,match="closed"):
        detector.accept(b"\0\0")


@pytest.mark.parametrize("content", ["", "ONE\nTWO\n", "x"*16385])
def test_keyword_file_is_one_bounded_line(model,content):
    directory, keyword = model
    keyword.write_text(content)
    with pytest.raises(WakeWordError,match="could not be loaded"):
        WakeDetector(directory,keyword,engine_factory=Engine).accept(b"\0\0")


def test_engine_exceptions_are_static_not_raw(model):
    directory, keyword = model

    def broken(**kwargs):
        raise RuntimeError("private path or native input")

    with pytest.raises(WakeWordError,match="could not be loaded") as error:
        WakeDetector(directory,keyword,engine_factory=broken).accept(b"\0\0")
    assert "private" not in str(error.value)
    engine = Engine()
    engine.stream.accept_waveform = broken
    with pytest.raises(WakeWordError,match="could not process"):
        WakeDetector(directory,keyword,engine_factory=lambda **k: engine).accept(b"\0\0")


@pytest.mark.parametrize("pcm", [b"x",b"\0"*(MAX_PCM_BYTES+2),bytearray(b"\0\0"),None])
def test_invalid_pcm_rejected_without_model(pcm,tmp_path):
    with pytest.raises(WakeWordError,match="bounded"):
        WakeDetector(tmp_path,tmp_path/"missing").accept(pcm)


@pytest.mark.parametrize("kwargs", [{"threshold":0},{"threshold":True},{"threshold":float("nan")},
                                     {"num_threads":0},{"num_threads":True},{"num_threads":5}])
def test_invalid_runtime_settings(kwargs,tmp_path):
    with pytest.raises(WakeWordError):
        WakeDetector(tmp_path,tmp_path/"missing",**kwargs)


def test_real_tokenizer_api_is_used_without_handwritten_tokens_or_writes(model):
    directory,_ = model
    before = {path.name:path.read_bytes() for path in directory.iterdir()}
    calls = []

    def tokenizer(*args,**kwargs):
        calls.append((args,kwargs))
        return [["TOKEN_A","TOKEN_B"]]

    assert tokenize_keyword(directory,tokenizer=tokenizer) == "TOKEN_A TOKEN_B\n"
    assert calls == [((["HEY VIDI"],), {"tokens":str(directory/"tokens.txt"),"tokens_type":"bpe",
                                        "bpe_model":str(directory/"bpe.model")})]
    assert {path.name:path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("rows", [[],[[]],[["BAD TOKEN"]],[["a"],["b"]],"token"])
def test_malformed_tokenization_is_rejected(model,rows):
    with pytest.raises(WakeWordError,match="could not encode"):
        tokenize_keyword(model[0],tokenizer=lambda *a,**k:rows)


def test_unapproved_phrase_and_missing_bpe_do_not_call_tokenizer(model):
    directory,_ = model
    with pytest.raises(WakeWordError,match="approved wake phrase"):
        tokenize_keyword(directory,"hey other",tokenizer=lambda *a,**k:pytest.fail("No tokenizer"))
    (directory/"bpe.model").unlink()
    with pytest.raises(WakeWordError,match="bpe.model"):
        tokenize_keyword(directory,tokenizer=lambda *a,**k:pytest.fail("No tokenizer"))


@pytest.fixture
def preparation():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/prepare_wake.py"))["main"]


def test_prepare_script_preserves_existing_output_on_tokenizer_failure(preparation,model,tmp_path,monkeypatch):
    directory,_ = model
    output = tmp_path.parent / (tmp_path.name+"-keywords.txt")
    output.write_text("existing keywords")

    def broken(_):
        raise WakeWordError("The local tokenizer failed.")

    monkeypatch.setitem(preparation.__globals__,"tokenize_keyword",broken)
    try:
        assert preparation(["--model-dir",str(directory),"--output",str(output)]) == 1
        assert output.read_text() == "existing keywords"
    finally:
        output.unlink()


def test_prepare_script_refuses_model_overwrite(preparation,model):
    directory,_ = model
    protected = directory / "tokens.txt"
    previous = protected.read_bytes()
    assert preparation(["--model-dir",str(directory),"--output",str(protected)]) == 1
    assert protected.read_bytes() == previous


def test_prepare_script_smoke_uses_only_synthetic_pcm(preparation,tmp_path,monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    output = tmp_path / "keywords.txt"
    made = []

    class FakeDetector:
        def __init__(self,model,keywords):
            assert model == directory and keywords == output
            self.count,self.prepared,self.closed = 0,False,False
            made.append(self)

        def prepare(self):
            self.prepared = True

        def accept(self,pcm):
            assert pcm == b"\0" * 640 and self.prepared
            self.count += 1
            return False

        def close(self):
            self.closed = True

    monkeypatch.setitem(preparation.__globals__,"tokenize_keyword",lambda _:"TOKENIZED\n")
    monkeypatch.setitem(preparation.__globals__,"WakeDetector",FakeDetector)
    assert preparation(["--model-dir",str(directory),"--output",str(output),"--smoke"]) == 0
    assert output.read_text() == "TOKENIZED\n"
    assert made[0].count == 250 and made[0].closed
