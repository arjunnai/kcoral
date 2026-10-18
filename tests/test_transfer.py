"""Sealed-buffer transport through real worker lifetime boundaries."""

import gc
import os
import sys
import warnings
from multiprocessing import reduction

import pytest
from support.runtime import FakeRuntime

from kcoral.protocol import compute_blob_hash
from kcoral.runtime import transfer
from kcoral.runtime.lease import NoopLeases
from kcoral.runtime.worker import Worker, WorkerCrashed, WorkerTimeout
from kcoral.schemas import GetFunction, Program, Ref, Return, Run, Upload

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires Linux")
SOURCE = "def main(t): return [sum(t.data), t]"


class BufferRuntime(FakeRuntime):
    def load_tensor(self, data, dtype, shape):
        import torch

        assert isinstance(data, bytes)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable", category=UserWarning
            )
            self.retained_tensor = torch.frombuffer(data, dtype=torch.uint8)
        return super().load_tensor(data, dtype, shape)


def tensor_program(data, source=SOURCE):
    digest = compute_blob_hash(data)
    return Program(
        instructions=[
            Upload("module", "module", source=source),
            GetFunction("fn", Ref("module"), "main", cpu_only=True),
            Upload("tensor", "tensor", blob=digest, dtype="uint8", shape=[len(data)]),
            Run("result", Ref("fn"), [Ref("tensor")]),
            Return("value", Ref("result")),
        ],
        blob_bytes={digest: data},
    )


def fd_count():
    gc.collect()
    return len(os.listdir("/proc/self/fd"))


@pytest.fixture
def worker():
    instance = Worker(
        None, BufferRuntime, sandbox="none", max_requests=0, termination_grace_seconds=0.1
    )
    try:
        yield instance
    finally:
        instance.close()


def check_result(worker, data, source=SOURCE):
    program = tensor_program(data, source)
    outcome = worker.run(program, 10, NoopLeases()).execution
    assert outcome.status == "COMPLETED", outcome.error
    values = outcome.results["value"]["value"]
    assert values[0]["value"] == 3 * len(data)
    assert list(outcome.binary_parts.values()) == [data]
    assert next(iter(program.blob_bytes.values())) is data


@pytest.mark.parametrize("size", [4096, transfer.MEMFD_THRESHOLD])
def test_large_and_small_inputs_have_identical_numeric_results(worker, size, monkeypatch):
    monkeypatch.delattr(os, "memfd_create", raising=False)
    sent, original = [], worker._conn.send

    def send(message):
        if isinstance(message, tuple):
            sent.extend(
                isinstance(value, transfer.BlobHandle) for value in message[0].blob_bytes.values()
            )
        return original(message)

    monkeypatch.setattr(worker._conn, "send", send)
    pid, before = worker.pid, fd_count()
    for _ in range(2):
        check_result(worker, b"\3" * size)
        assert worker.pid == pid and fd_count() == before
    assert sent == [size >= transfer.MEMFD_THRESHOLD] * 2


def test_memfd_unavailable_preserves_the_result(worker, monkeypatch):
    def unavailable(*args):
        raise OSError("memfd unavailable")

    monkeypatch.setattr(transfer, "_memfd_create", unavailable)
    source = (
        "from kcoral.runtime import transfer\n"
        "def unavailable(*args): raise OSError('memfd unavailable')\n"
        "transfer._memfd_create = unavailable\n" + SOURCE
    )
    before = fd_count()
    check_result(worker, b"\3" * transfer.MEMFD_THRESHOLD, source)
    assert fd_count() == before


@pytest.mark.parametrize("failure", ["crash", "timeout", "ack"])
def test_large_input_failures_do_not_leak_parent_fds(worker, failure, monkeypatch):
    data = b"\3" * transfer.MEMFD_THRESHOLD
    source = (
        "import os\ndef main(t): os._exit(3)"
        if failure == "crash"
        else "import time\ndef main(t): time.sleep(10)"
    )
    check_result(worker, data)
    before = fd_count()
    pid = worker.pid
    with monkeypatch.context() as patch:
        if failure == "ack":

            def failed_ack(*args):
                raise RuntimeError("did not receive acknowledgement of fd")

            patch.setattr(reduction, "send_handle", failed_ack)
        with pytest.raises(WorkerTimeout if failure == "timeout" else WorkerCrashed):
            worker.run(tensor_program(data, source), 0.3, NoopLeases())
    assert worker.pid != pid
    check_result(worker, data)
    assert fd_count() == before
