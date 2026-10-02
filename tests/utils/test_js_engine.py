from pathlib import Path
import stat
import sys

import pytest

from utils.js_engine import classify_error
from utils.js_engine import Engine
from utils.js_engine import is_crash
from utils.js_engine import JSError
from utils.js_engine import ReprlError
from utils.js_engine import V8Engine
from utils.js_engine import V8ReprlEngine


FAKE_D8 = Path(__file__).parent / "fake_d8.py"


@pytest.fixture(scope="module")
def fake_d8(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # posix_spawn needs a real executable, so wrap the script with this interpreter
    path = tmp_path_factory.mktemp("engine") / "d8"
    path.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{FAKE_D8}' \"$@\"\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture(params=["reprl", "subprocess"])
def engine(request: pytest.FixtureRequest, fake_d8: Path):
    engine_cls = V8ReprlEngine if request.param == "reprl" else V8Engine
    engine = engine_cls(fake_d8, timeout=2)
    yield engine
    engine.close()


class TestClassification:
    def test_js_exception_is_not_a_crash(self):
        assert classify_error("TypeError: x is not a function", True) == JSError.TypeError
        assert not is_crash(None, "")

    def test_signal_is_a_crash(self):
        assert is_crash(11, "")

    def test_sanitizer_report_is_a_crash(self):
        assert is_crash(None, "ERROR: AddressSanitizer: heap-use-after-free")

    def test_dcheck_is_a_crash(self):
        assert is_crash(None, "# Fatal error in ../../src/foo.cc\n# Debug check failed: x")

    def test_oom_is_not_a_crash(self):
        out = "# Fatal process out of memory: Zone"
        assert not is_crash(6, out)
        assert classify_error(out, True) == JSError.OutOfMemory

    def test_our_timeout_kill_is_not_a_crash(self):
        assert not is_crash(9, "")

    def test_success(self):
        assert classify_error("anything", False) == JSError.NoError


class TestEngines:
    def test_coverage(self, engine: Engine):
        data = engine.execute_text("COVER 1\nCOVER 9")
        assert data is not None
        assert data.error == JSError.NoError
        assert not data.is_crash()
        assert data.coverage.num_edges == 64
        assert data.coverage.hit_edges == 2

    def test_coverage_is_reset_between_executions(self, engine: Engine):
        engine.execute_text("COVER 1\nCOVER 2\nCOVER 3")
        data = engine.execute_text("COVER 5")
        assert data is not None
        assert data.coverage.hit_edges == 1

    def test_exception(self, engine: Engine):
        data = engine.execute_text("COVER 1\nTHROW")
        assert data is not None
        assert data.error == JSError.TypeError
        assert not data.is_crash()

    def test_crash_then_recovers(self, engine: Engine):
        data = engine.execute_text("COVER 3\nCRASH")
        assert data is not None
        assert data.is_crash()
        assert data.coverage.hit_edges == 1

        data = engine.execute_text("COVER 4")
        assert data is not None
        assert data.error == JSError.NoError
        assert not data.is_crash()

    def test_sanitizer_crash(self, engine: Engine):
        data = engine.execute_text("ASAN")
        assert data is not None
        assert data.is_crash()
        assert "AddressSanitizer" in data.err

    def test_oom_is_not_a_crash(self, engine: Engine):
        data = engine.execute_text("OOM")
        assert data is not None
        assert not data.is_crash()
        assert data.error == JSError.OutOfMemory

    def test_timeout_then_recovers(self, engine: Engine):
        engine.timeout = 0.3
        data = engine.execute_text("COVER 2\nHANG")
        assert data is not None
        assert data.error == JSError.TimeoutError
        assert not data.is_crash()

        engine.timeout = 2
        data = engine.execute_text("COVER 1")
        assert data is not None
        assert data.error == JSError.NoError

    def test_lib_path_is_loaded_first(self, engine: Engine):
        data = engine.execute_text("PRINT hello", lib_path=Path("lib.js"))
        assert data is not None
        assert data.error == JSError.NoError
        assert "hello" in data.out


class TestReprl:
    def test_process_is_reused(self, fake_d8: Path):
        with V8ReprlEngine(fake_d8, timeout=2) as engine:
            pid = engine.pid
            for _ in range(5):
                engine.execute_text("COVER 1")
            assert engine.pid == pid

    def test_engine_cannot_dump_core(self, fake_d8: Path):
        """A core dump of a real engine takes seconds, making a crash a timeout."""
        with V8ReprlEngine(fake_d8, timeout=2) as engine:
            limits = Path(f"/proc/{engine.pid}/limits")
            if limits.exists():
                line = next(l for l in limits.read_text().splitlines() if "core file" in l)
                assert line.split()[4] == "0"  # "Max core file size <soft> <hard> bytes"
            else:
                import resource

                assert resource.getrlimit(resource.RLIMIT_CORE)[0] == 0

    def test_no_handshake_raises(self, fake_d8: Path):
        with pytest.raises(ReprlError):
            V8ReprlEngine(fake_d8, timeout=2, extra_args=["--no-reprl"])

    def test_auto_falls_back_to_subprocess(self, fake_d8: Path):
        engine = Engine.get_engine("v8", fake_d8, extra_args=["--no-reprl"])
        try:
            assert isinstance(engine, V8Engine)
            data = engine.execute_text("COVER 1")
            assert data is not None and data.coverage.hit_edges == 1
        finally:
            engine.close()

    def test_auto_uses_reprl(self, fake_d8: Path):
        engine = Engine.get_engine("v8", fake_d8)
        try:
            assert isinstance(engine, V8ReprlEngine)
        finally:
            engine.close()


class TestRemoteEngine:
    def test_matches_local_engine(self, fake_d8: Path):
        import socket
        import threading

        from utils.engine_server import serve
        from utils.js_engine import RemoteEngine

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        threading.Thread(
            target=serve,
            args=("127.0.0.1", port, "v8", str(fake_d8), 2.0, []),
            daemon=True,
        ).start()

        import time

        for _ in range(50):
            try:
                engine = RemoteEngine(f"127.0.0.1:{port}")
                break
            except ConnectionRefusedError:
                time.sleep(0.1)

        try:
            data = engine.execute_text("COVER 1\nCOVER 9")
            assert data is not None
            assert data.error == JSError.NoError
            assert data.coverage.num_edges == 64 and data.coverage.hit_edges == 2

            data = engine.execute_text("THROW", lib_path=Path("lib.js"))
            assert data is not None and data.error == JSError.TypeError

            data = engine.execute_text("CRASH")
            assert data is not None and data.is_crash()

            data = engine.execute_text("COVER 3")
            assert data is not None and data.coverage.hit_edges == 1
        finally:
            engine.close()
