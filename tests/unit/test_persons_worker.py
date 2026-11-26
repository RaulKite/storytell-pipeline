"""Persons worker: the argument-passing and refusal rules that a mock cannot see through.

The reason this file exists is a defect found by running the real tool on a real GPU, which
every pure-python test in the suite passed over in silence: the worker built its
``track_video`` signature with a keyword-only ``imgsz``, added the ``--imgsz`` CLI flag, put
``imgsz`` into the arguments handed to ultralytics — and never passed ``imgsz`` from
``main()`` into ``track_video()``. Every video failed with ``TypeError`` in under three
seconds. Nothing in the stage tests touched that seam, because the stage tests hand the stage
a prepared raw document.

So the tests here are chosen for one property: each one breaks if a specific argument stops
reaching the specific call it must reach, or if a refusal is replaced by a plausible-looking
empty result.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "workers" / "persons_worker.py"


def load_worker():
    """Import the worker by path.

    It runs inside its own uv project and must stay unimportable as a package module; its
    heavy imports (ultralytics, torch) happen inside functions, so loading the module itself
    costs nothing and never touches the GPU.
    """
    spec = importlib.util.spec_from_file_location("persons_worker_under_test", WORKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- fake ultralytics


class FakeBoxes:
    """``result.boxes`` with the three tensors the worker reads.

    Every tensor is present even when empty, because the worker checks ``numel()`` on
    ``xyxy``/``conf`` rather than ``is not None`` — a stub that omitted them would pass here
    and fail against the real library, which always builds the tensor.
    """

    def __init__(self, ids: list[int], xyxy: list[list[float]], scores: list[float]) -> None:
        self.id = _Tensor(ids) if ids else None
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(scores)


class _Tensor:
    """Smallest thing that satisfies the calls the worker makes on a tensor."""

    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def numel(self) -> int:
        return len(self._values)

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> list[Any]:
        return self._values


class FakeResult:
    def __init__(self, frame_id: int, ids: list[int],
                 xyxy: list[list[float]] | None = None,
                 scores: list[float] | None = None,
                 frame: "_Frame | None" = None) -> None:
        self.frame_id = frame_id
        boxes = xyxy if xyxy is not None else [[0.0, 0.0, 10.0, 20.0] for _ in ids]
        confs = scores if scores is not None else [0.5] * len(ids)
        self.boxes = FakeBoxes(ids, boxes, confs)
        # What ultralytics hands back for the frame it actually decoded. The renderer draws on
        # this and nothing else: a second decode pass could not prove it was looking at the
        # frame these boxes came from.
        self.orig_img = frame if frame is not None else _Frame()


class FakeModel:
    """``ultralytics.YOLO`` stand-in that records the keyword arguments it was handed."""

    #: frames to emit, as (frame_id, ids) pairs
    results: list[FakeResult] = []
    #: what ``ckpt_path`` reports, i.e. where ultralytics says it loaded from
    ckpt_path: str | None = None
    last_kwargs: dict[str, Any] | None = None
    #: the constructor argument, i.e. which checkpoint the run was actually built from
    built_with: str | None = None
    frame_id_offset: int = 0

    def __init__(self, argument: str) -> None:
        self.argument = argument
        type(self).built_with = argument
        self.ckpt_path = type(self).ckpt_path

    def track(self, source: str, **kwargs: Any):
        type(self).last_kwargs = kwargs
        offset = type(self).frame_id_offset
        for result in type(self).results:
            if offset:
                result.frame_id = result.frame_id + offset
            yield result


@pytest.fixture
def worker(monkeypatch):
    """The worker module with a fake ``ultralytics`` and a GPU-less ``torch`` injected.

    Both stubs are needed for every run-path test: the worker resolves its device through
    ``torch.cuda.is_available()`` before it builds anything, and the pipeline's own test
    environment has no torch at all (that is the point of the separate uv project). The stub
    reports no GPU, so ``--device cpu`` exercises the real code path and any test that wants
    the cuda branch installs its own torch, which is more explicit than a knob here.

    Reset per test because ``FakeModel`` keeps the last kwargs on the class, and a test that
    asserts on them must not read another test's call.
    """
    FakeModel.results = []
    FakeModel.ckpt_path = None
    FakeModel.last_kwargs = None
    FakeModel.built_with = None
    FakeModel.frame_id_offset = 0
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=False))
    return load_worker()


def _torch_stub(cuda: bool):
    module = types.ModuleType("torch")
    module.__version__ = "2.8.0+cu126"
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return module


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    return video


@pytest.fixture
def frame_index_file(tmp_path):
    """A frame index the worker can read, written with the pipeline's own columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "frame_index.parquet"
    pq.write_table(pa.table({
        "frame_number": pa.array([0, 1, 2], type=pa.int64()),
        "pts_seconds": pa.array([0.0, 0.04, 0.08], type=pa.float64()),
    }), path)
    return path


def run_worker(worker, tmp_path, *, video=None, frame_index=None, extra=(),
               out=None, result=None) -> dict[str, Any]:
    """Invoke ``main`` the way the orchestrator does and return the result payload."""
    out = out or tmp_path / "raw" / "yolo_track.json"
    result = result or tmp_path / "raw" / "persons_worker_result.json"
    argv = [
        "--video", str(video or (tmp_path / "missing.mp4")),
        "--frame-index", str(frame_index or (tmp_path / "nope.parquet")),
        "--output-json", str(out),
        "--result-path", str(result),
        "--video-id", "vid_a",
        "--model", "yolo11n.pt",
        "--device", "cpu",
        "--tracker", "bytetrack",
        "--conf", "0.25",
        "--imgsz", "640",
        "--classes", "0",
        *extra,
    ]
    code = worker.main(argv)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["_exit"] = code
    # The alias tests deliberately point ``out`` at the video or the frame index, which is
    # binary-by-design; parsing it would raise before the assertion that matters can run.
    readable = out.is_file() and not any(_same_file(out, other)
                                        for other in (video, frame_index))
    payload["_document"] = json.loads(out.read_text(encoding="utf-8")) if readable else None
    return payload


def _same_file(one, two) -> bool:
    """Test-side path identity, kept separate from the worker's own guard."""
    if one is None or two is None:
        return False
    try:
        return Path(one).resolve() == Path(two).resolve()
    except OSError:
        return False


# ------------------------------------------------- arguments that must reach the run


class TestArgumentsReachUltralytics:
    """Each configured value has to arrive at ``track()``, not merely at the document.

    Recording the same number in the raw JSON is not the same as using it: the document is
    written from local variables, so a value can be reported faithfully and still never have
    been passed, which is precisely how ``imgsz`` got away with it.
    """

    def test_imgsz_reaches_the_track_call(self, worker, tmp_path, clip, frame_index_file):
        """The real defect. It failed every clip, in 3 seconds, with the GPU idle."""
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["imgsz"] == 640

    def test_a_non_default_imgsz_is_not_silently_replaced_by_the_default(
            self, worker, tmp_path, clip, frame_index_file):
        """A test on 640 alone also passes when the flag is ignored: 640 *is* the default.

        A non-default value is the only one that distinguishes "forwarded" from "dropped".
        """
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--imgsz", "320"])
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["imgsz"] == 320

    def test_conf_is_passed_explicitly_because_its_default_is_not_the_library_default(
            self, worker, tmp_path, clip, frame_index_file):
        """ultralytics' ``track()`` overrides conf to 0.1 when the caller omits it.

        Omitting it would report the counts a 0.1 threshold produced while the document said
        0.25 — a configuration claim that is false in the one direction nobody notices,
        because a lower threshold finds *more* people and more people looks like a working
        detector.
        """
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["conf"] == 0.25

    def test_the_recorded_conf_matches_the_one_the_run_used(self, worker, tmp_path, clip,
                                                           frame_index_file):
        """The document's parameters and the actual call are asserted together.

        Either half alone passes when they disagree, and the disagreement is the bug.
        """
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--conf", "0.4"])
        assert payload["status"] == "ok", payload.get("error")
        assert payload["_document"]["parameters"]["conf"] == FakeModel.last_kwargs["conf"] == 0.4

    def test_classes_reach_the_call_as_ints(self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["classes"] == [0]

    def test_an_empty_class_filter_is_omitted_rather_than_sent_as_an_empty_list(
            self, worker, tmp_path, clip, frame_index_file):
        """``classes=[]`` means "no class matches" to ultralytics, not "no filter".

        Passing an empty list would return zero detections on every frame and produce a
        perfectly valid-looking table of nobody being there.
        """
        FakeModel.results = [FakeResult(0, [])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--classes", ""])
        assert payload["status"] == "ok", payload.get("error")
        assert "classes" not in FakeModel.last_kwargs

    def test_persist_is_false_and_cannot_be_turned_on(self, worker, tmp_path, clip,
                                                     frame_index_file):
        """``persist=True`` carries track ids across sources in one process.

        It is not exposed as a knob precisely so this assertion has to be edited rather than
        a config file.
        """
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["persist"] is False

    def test_the_tracker_name_becomes_the_yaml_the_library_expects(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--tracker", "botsort"])
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["tracker"] == "botsort.yaml"

    def test_device_cpu_is_passed_and_auto_is_left_to_the_library(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["device"] == "cpu"

    def test_auto_resolves_here_and_the_resolved_device_reaches_the_call(self, worker, tmp_path,
                                                                        clip, frame_index_file,
                                                                        monkeypatch):
        """The real resolver, not a stubbed one.

        The first version of this test replaced ``resolve_device`` with a lambda returning
        ``(None, "cpu", ...)`` and asserted ``"device" not in last_kwargs`` -- a passing test for
        a code path the worker never produces, because the real resolver has no branch that
        returns ``None``. It was asserting the shape of its own fake. Native review caught it
        (R3-auto-device-dead-stub).

        So: no GPU visible, ``auto`` must arrive as an explicit ``cpu``, and the document must
        say why -- which is what makes a CPU run distinguishable after the fact.
        """
        FakeModel.results = [FakeResult(0, [1])]
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=False))
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--device", "auto"])
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["device"] == "cpu"
        assert payload["_document"]["requested_device"] == "auto"
        assert payload["_document"]["device"] == "cpu"
        assert payload["_document"]["device_fallback_reason"]

    def test_auto_on_a_gpu_arrives_as_the_index_the_library_wants(self, worker, tmp_path, clip,
                                                                 frame_index_file, monkeypatch):
        """The other auto branch, same plumbing, recorded with no fallback reason."""
        FakeModel.results = [FakeResult(0, [1])]
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=True))
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--device", "auto"])
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.last_kwargs["device"] == 0
        assert payload["_document"]["device"] == "cuda"
        assert payload["_document"]["device_fallback_reason"] is None

    def test_no_device_choice_ever_defers_the_decision_to_the_library(self, worker, monkeypatch):
        """Pin the invariant that makes the ``device_arg is not None`` guard non-decorative."""
        for requested, cuda_visible in (("auto", False), ("auto", True), ("cpu", True),
                                       ("cuda", True)):
            monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=cuda_visible))
            argument, name, _reason = worker.resolve_device(requested)
            assert argument is not None, f"{requested} deferred the choice to ultralytics"
            assert name in ("cpu", "cuda")


# --------------------------------------------------------------- refusal over empty


class TestRefusalRatherThanAnEmptyTable:
    """§20.2: a run that measured nothing must say so, not write a table of nobody.

    An empty Parquet is indistinguishable from "this clip genuinely has no people", and the
    pipeline already has a clip of that kind (pipeline_demo, 249 frames, 0 persons, verified
    on the GPU). A stage that could not tell the two apart would let a broken run pass as a
    true finding.
    """

    def test_zero_decoded_frames_is_a_failure_not_an_empty_document(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = []
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "error"
        assert "0 frames" in payload["error"]
        assert payload["_document"] is None, "a document was written for a run that saw nothing"

    def test_a_frame_id_that_disagrees_with_the_counter_is_a_failure(
            self, worker, tmp_path, clip, frame_index_file):
        """Every timestamp is built from result order, so a changed order corrupts the table
        without raising anything anywhere else."""
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, [1])]
        FakeModel.frame_id_offset = 5
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "error"
        assert "frame_id" in payload["error"]
        assert payload["_document"] is None

    def test_a_missing_video_is_refused_before_any_model_is_built(
            self, worker, tmp_path, frame_index_file):
        """The refusal happens in argument validation, so no checkpoint is ever opened."""
        payload = run_worker(worker, tmp_path, frame_index=frame_index_file)
        assert payload["status"] == "error"
        assert "input video not found" in payload["error"]
        assert FakeModel.built_with is None

    def test_a_missing_frame_index_names_the_stage_it_needs(
            self, worker, tmp_path, clip):
        payload = run_worker(worker, tmp_path, video=clip)
        assert payload["status"] == "error"
        assert "metadata stage" in payload["error"]

    def test_a_frame_index_with_no_usable_rows_is_refused(self, worker, tmp_path, clip,
                                                          frame_index_file):
        """An empty index means no timestamp can be named for any sighting."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        empty = tmp_path / "empty_index.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([], type=pa.int64()),
            "pts_seconds": pa.array([], type=pa.float64()),
        }), empty)
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=empty)
        assert payload["status"] == "error"
        assert "no usable frame rows" in payload["error"]

    def test_an_error_still_writes_a_result_payload_with_the_traceback(
            self, worker, tmp_path, clip, frame_index_file):
        """The orchestrator reads this file to decide between retry and skip."""
        FakeModel.results = []
        FakeModel.frame_id_offset = 0
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["_exit"] == 1
        assert "traceback" in payload and payload["stage"] == "persons"


class TestWeightsResolution:
    def test_a_configured_weights_dir_with_no_checkpoint_refuses_to_download(
            self, worker, tmp_path, clip, frame_index_file):
        empty = tmp_path / "weights"
        empty.mkdir()
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--weights-dir", str(empty)])
        assert payload["status"] == "error"
        assert "refusing to fall back" in payload["error"]

    def test_a_configured_weights_dir_that_is_not_a_directory_is_refused(self, worker,
                                                                        tmp_path, clip,
                                                                        frame_index_file):
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--weights-dir", str(tmp_path / "nope")])
        assert payload["status"] == "error"

    def test_an_unset_weights_dir_lets_the_library_resolve_and_the_path_is_recorded(
            self, worker, tmp_path, clip, frame_index_file, monkeypatch):
        FakeModel.results = [FakeResult(0, [1])]
        FakeModel.ckpt_path = str(tmp_path / "downloaded" / "yolo11n.pt")
        (tmp_path / "downloaded").mkdir()
        (tmp_path / "downloaded" / "yolo11n.pt").write_bytes(b"w")
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert payload["_document"]["weights_path"].endswith("yolo11n.pt")
        assert len(payload["_document"]["weights_sha256"]) == 64

    def test_an_explicit_weights_dir_is_hashed_and_recorded(self, worker, tmp_path, clip,
                                                            frame_index_file):
        import hashlib

        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "yolo11n.pt").write_bytes(b"checkpoint bytes")
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--weights-dir", str(weights)])
        assert payload["status"] == "ok", payload.get("error")
        assert payload["_document"]["weights_sha256"] == hashlib.sha256(
            b"checkpoint bytes").hexdigest()
        assert payload["weights_sha256"] == payload["_document"]["weights_sha256"]

    def test_the_model_is_built_from_the_resolved_path_not_the_bare_name(
            self, worker, tmp_path, clip, frame_index_file):
        """Handing ultralytics the bare name would let it download a second copy.

        With ``weights_dir`` set, the operator asked for *this* file; loading a name is how a
        run ends up reporting one checkpoint's sha256 while another model produced the boxes.
        """
        weights = tmp_path / "weights"
        weights.mkdir()
        (weights / "yolo11n.pt").write_bytes(b"x")
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--weights-dir", str(weights)])
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.built_with == str(weights / "yolo11n.pt")

    def test_without_a_weights_dir_the_bare_name_is_handed_over(self, worker, tmp_path, clip,
                                                                frame_index_file):
        """The one case where ultralytics' own download-on-first-use is the intended path."""
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert FakeModel.built_with == "yolo11n.pt"


class TestFrameIndexMapping:
    def test_a_frame_the_index_never_listed_gets_no_timestamp(self, worker):
        """None, not an extrapolation: an unnameable sighting cannot be placed on a timeline."""
        assert worker.resolve_timestamp({0: 0.0, 1: 0.04}, 40) is None

    def test_a_listed_frame_gets_its_own_value(self, worker):
        assert worker.resolve_timestamp({0: 0.0, 1: 0.04}, 1) == 0.04

    def test_a_frame_number_missing_from_a_real_index_lands_in_the_count(
            self, worker, tmp_path, clip):
        import pyarrow as pa
        import pyarrow.parquet as pq

        index = tmp_path / "frame_index.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([0], type=pa.int64()),
            "pts_seconds": pa.array([0.0], type=pa.float64()),
        }), index)
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=index)
        assert payload["status"] == "ok", payload.get("error")
        document = payload["_document"]
        assert document["frames_without_timestamp"] == 1
        assert [row["timestamp"] for row in document["frames"]] == [0.0, None]

    def test_the_frame_index_is_read_without_extrapolating_a_missing_column(
            self, worker, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "frame_index.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([0, 1], type=pa.int64()),
            "pts_seconds": pa.array([0.0, None], type=pa.float64()),
        }), path)
        assert worker.load_frame_index(path) == {0: 0.0}

    def test_an_unreadable_frame_index_becomes_a_worker_failure_not_a_traceback(
            self, worker, tmp_path):
        path = tmp_path / "frame_index.parquet"
        path.write_bytes(b"not parquet")
        with pytest.raises(worker.WorkerFailure, match="could not be read"):
            worker.load_frame_index(path)


class TestDetectionRows:
    def test_a_row_carries_its_own_box_and_score(self, worker):
        row = worker._detection_row(frame_number=3, timestamp=0.12, person_id=7,
                                    box=[1.0, 2.0, 30.0, 40.0], score=0.777777,
                                    persons_in_frame=2)
        assert row["frame_number"] == 3 and row["person_id"] == 7
        assert (row["x1"], row["y2"]) == (1.0, 40.0)
        assert row["confidence"] == 0.777777
        assert row["persons_in_frame"] == 2

    def test_bbox_is_the_model_output_unrescaled(self, worker):
        """A second letterbox inverse in this repo would be a second thing to get wrong."""
        row = worker._detection_row(frame_number=0, timestamp=0.0, person_id=1,
                                    box=[0.123456, 0.0, 1919.9, 1079.9], score=0.5,
                                    persons_in_frame=1)
        assert (row["x1"], row["x2"]) == (0.123, 1919.9)

    def test_persons_in_frame_counts_the_ids_on_that_frame_not_the_rows(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1, 2, 3])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        rows = payload["_document"]["frames"]
        assert [row["persons_in_frame"] for row in rows] == [3, 3, 3]
        assert payload["_document"]["max_persons_in_frame"] == 3

    def test_ids_are_sorted_so_the_raw_document_is_diffable(self, worker, tmp_path, clip,
                                                            frame_index_file):
        """Unsorted ids make every rerun look like a change to the raw artifact."""
        FakeModel.results = [FakeResult(0, [5, 2]), FakeResult(1, [5]), FakeResult(2, [2])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        document = payload["_document"]
        assert document["person_ids"] == [2, 5]
        assert list(document["person_frame_counts"]) == ["2", "5"]
        assert document["person_frame_counts"] == {"2": 2, "5": 2}

    def test_a_frame_with_no_persons_is_counted_but_produces_no_row(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, []), FakeResult(2, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        document = payload["_document"]
        assert document["frames_measured"] == 3
        assert document["frames_with_person"] == 2
        assert len(document["frames"]) == 2


class TestGmcWarningCounter:
    def test_the_warning_is_counted_rather_than_left_in_a_log(self, worker):
        counter = worker.GmcWarningCounter()
        with counter:
            logging.getLogger("ultralytics").warning("GMC failed: affine warp unavailable")
        assert counter.count == 1
        assert "affine warp" in counter.first_reason

    def test_the_handler_is_removed_on_exit(self, worker):
        counter = worker.GmcWarningCounter()
        logger = logging.getLogger("ultralytics")
        before = list(logger.handlers)
        with counter:
            assert len(logger.handlers) == len(before) + 1
        assert list(logger.handlers) == before

    def test_an_unrelated_warning_is_not_counted(self, worker):
        counter = worker.GmcWarningCounter()
        with counter:
            logging.getLogger("ultralytics").warning("Model loaded in 0.3s")
        assert counter.count == 0
        assert counter.first_reason is None

    def test_a_counted_failure_survives_into_the_document(self, worker, tmp_path, clip,
                                                          frame_index_file):
        """The reason it is data: a WARNING in a log nobody reads cannot gate a decision."""
        FakeModel.results = [FakeResult(0, [1])]
        original = worker.GmcWarningCounter

        class Leaky(original):
            def __enter__(self):
                super().__enter__()
                logging.getLogger("ultralytics").warning("GMC failed: cv2 missing")
                return self

        worker.GmcWarningCounter = Leaky
        try:
            payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        finally:
            worker.GmcWarningCounter = original
        assert payload["status"] == "ok", payload.get("error")
        assert payload["gmc_failure_count"] == 1
        assert payload["_document"]["gmc_failure_reason"]


class TestArgparseDoesNotAcceptAbbreviations:
    """The stage refuses ``--output-video`` in extra_args; argparse used to accept ``--output-v``.

    A prefix is the same flag to argparse, so a guard that compares whole flag names is only as
    strong as the parser's abbreviation rule. Both sides are closed here.
    """

    @pytest.mark.parametrize("flag", ["--output-v", "--output-vid", "--ffm", "--ffp"])
    def test_an_abbreviated_flag_is_not_accepted(self, worker, flag):
        # Every required argument is supplied, so the *only* reason this can exit is the
        # abbreviation itself: a missing-argument error would make this test pass for the
        # wrong reason and keep passing after the parser went back to accepting prefixes.
        required = ["--video", "v.mp4", "--frame-index", "f.parquet",
                    "--output-json", "o.json", "--result-path", "r.json"]
        with pytest.raises(SystemExit):
            worker.parse_args(required + [flag, "/tmp/x.mp4"])

    def test_the_exact_flags_still_parse(self, worker):
        parsed = worker.parse_args([
            "--video", "v.mp4", "--frame-index", "f.parquet",
            "--output-json", "o.json", "--result-path", "r.json",
            "--output-video", "tracks.mp4", "--ffmpeg", "ffmpeg",
            "--ffprobe", "ffprobe"])
        assert parsed.output_video == Path("tracks.mp4")
        assert parsed.ffmpeg == "ffmpeg"
        assert parsed.ffprobe == "ffprobe"


class TestArgumentValidation:
    @pytest.mark.parametrize("value", ["0", "1.0", "-0.1"])
    def test_a_confidence_outside_the_open_interval_is_refused(self, worker, tmp_path, clip,
                                                              frame_index_file, value):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--conf", value])
        assert payload["status"] == "error"
        assert "confidence" in payload["error"]

    def test_a_non_positive_imgsz_is_refused(self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--imgsz", "0"])
        assert payload["status"] == "error"

    def test_a_non_numeric_class_is_refused_by_name(self, worker):
        with pytest.raises(worker.WorkerFailure, match="not an integer"):
            worker.parse_classes("0, banana")

    def test_a_blank_model_name_is_refused(self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--model", "  "])
        assert payload["status"] == "error"


class TestOutputPathsMustNotAliasInputs:
    """A worker must never write its artifact over the data it was asked to read.

    This is not a hypothetical argument typo. ``write_json_atomic`` ends in ``os.replace``, so
    before this guard the real worker -- real GPU, real clip, the command line a caller can
    actually type -- tracked a 126-frame clip for 1.4 s, replaced the MP4 with the tracking
    JSON, and returned ``status: ok`` with the video gone. Reproduced by running the worker
    from commit b9e1c68 against a /tmp copy: ``file`` reported ``JSON data`` afterwards and the
    result document still claimed 3 people over 126 frames.
    """

    def test_an_output_that_names_the_video_is_refused_before_any_tracking(self, worker,
                                                                          tmp_path,
                                                                          clip, frame_index_file):
        """The refusal happens before the run, not after the tracking has already cost a GPU pass."""
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=clip)
        assert payload["status"] == "error"
        assert "--output-json points at the --video input" in payload["error"]
        assert FakeModel.last_kwargs is None, "tracking ran anyway"
        assert clip.read_bytes() == b"\x00", "the video was overwritten"

    def test_an_output_that_names_the_frame_index_is_refused(self, worker, tmp_path, clip,
                                                            frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        index_bytes = frame_index_file.read_bytes()
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=frame_index_file)
        assert payload["status"] == "error"
        assert "--frame-index" in payload["error"]
        assert frame_index_file.read_bytes() == index_bytes

    def test_a_result_path_that_names_the_video_is_refused_without_writing_anything(
            self, worker, tmp_path, clip, frame_index_file, caplog):
        """The guard cannot itself be delivered by destroying the input.

        The result contract is written in a ``finally`` block, so an ordinary refusal here
        would still overwrite the video with the error JSON -- the first draft of this fix did
        exactly that, and it was caught by running the real worker against a copy of a real
        clip. So this path answers on stderr, exits 1, and touches no path at all; the stage
        then fails on its own missing-result check. Writing *nothing* is also why the run does
        not report a plausible failure for a video it never read.
        """
        FakeModel.results = [FakeResult(0, [1])]
        video_bytes = clip.read_bytes()
        out = tmp_path / "raw" / "yolo_track.json"
        code = worker.main([
            "--video", str(clip), "--frame-index", str(frame_index_file),
            "--output-json", str(out), "--result-path", str(clip),
            "--video-id", "vid_a", "--device", "cpu",
        ])
        assert code == 1
        assert clip.read_bytes() == video_bytes, "the guard destroyed the video it was refusing"
        assert not out.exists(), "tracking ran, or its document was written"
        assert FakeModel.last_kwargs is None

    def test_the_two_outputs_naming_one_file_are_refused(self, worker, tmp_path, clip,
                                                        frame_index_file):
        """Otherwise the second write silently erases the tracking document.

        With ``--result-path`` on one side the refusal cannot even be delivered through the
        result contract: writing it *is* the overwrite. So nothing is written anywhere and the
        run exits non-zero, which is what the stage reports as a missing result.
        """
        FakeModel.results = [FakeResult(0, [1])]
        shared = tmp_path / "raw" / "one.json"
        code = worker.main([
            "--video", str(clip), "--frame-index", str(frame_index_file),
            "--output-json", str(shared), "--result-path", str(shared),
            "--video-id", "vid_a", "--device", "cpu",
        ])
        assert code == 1
        # The review asked for this specifically: --output-json is not an input, and a refusal
        # that calls it one teaches the reader to distrust the message on the cases that matter.
        assert not shared.exists()
        assert FakeModel.last_kwargs is None

    def test_two_outputs_naming_one_file_without_the_result_path_are_refused(
            self, worker, tmp_path, clip, frame_index_file):
        """The non-``--result-path`` case still answers through the result contract."""
        FakeModel.results = [FakeResult(0, [1])]
        shared = tmp_path / "raw" / "one.json"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=shared, extra=["--output-video", str(shared)])
        assert payload["status"] == "error"
        assert "--output-video points at the --output-json output" in payload["error"]
        assert FakeModel.last_kwargs is None

    def test_an_unresolvable_path_does_not_crash_the_guard(self, worker, tmp_path, clip,
                                                          frame_index_file):
        """A symlink loop must not escape the guard before ``main``'s try block.

        ``resolve()`` fails on a loop with ``RuntimeError``, which is not an ``OSError``, and
        the alias check runs before any handler is installed. Escaping there kills the worker
        with no result document at all, which is what this asserts against: the harness reads
        the result file, so a crash surfaces here as a missing file rather than a bad payload.

        What the run then does is a separate matter and it is not data loss: ``rename`` does not
        follow a final symlink component, so writing to the loop replaced the *link* with the
        JSON document (measured -- the status came back ``ok``). The video, which is what this
        guard exists to protect, was untouched.
        """
        FakeModel.results = [FakeResult(0, [1])]
        loop_a = tmp_path / "loop_a"
        loop_b = tmp_path / "loop_b"
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=loop_a)
        assert "Symlink loop" not in json.dumps(payload), "the guard's own resolve() escaped"
        assert clip.read_bytes() == b"\x00"

    def test_the_same_file_reached_by_a_different_spelling_is_still_caught(self, worker,
                                                                          tmp_path, clip,
                                                                          frame_index_file):
        """`..`, a symlink and a doubled separator all name the operator's video.

        The stage builds these paths from config, so a textual comparison would have been
        enough only if config were written with one canonical spelling forever.
        """
        FakeModel.results = [FakeResult(0, [1])]
        link = tmp_path / "link_to_clip.mp4"
        link.symlink_to(clip)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=tmp_path / "raw" / ".." / "link_to_clip.mp4")
        assert payload["status"] == "error"
        assert "--video" in payload["error"]
        assert clip.read_bytes() == b"\x00"

    def test_paths_that_share_a_prefix_are_not_treated_as_aliases(self, worker, tmp_path,
                                                                  clip, frame_index_file):
        """The guard must not refuse the ordinary layout the stage actually uses."""
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok"
        assert (tmp_path / "raw" / "yolo_track.json").is_file()


class TestAtomicOutput:
    def test_the_document_is_written_in_one_step(self, worker, tmp_path):
        out = tmp_path / "raw" / "yolo_track.json"
        worker.write_json_atomic(out, {"a": 1})
        assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}
        assert not list(out.parent.glob("*.tmp")), "a temporary file was left behind"

    def test_a_payload_with_non_ascii_is_preserved(self, worker, tmp_path):
        out = tmp_path / "raw" / "yolo_track.json"
        worker.write_json_atomic(out, {"clip": "Telediario — La 1"})
        assert "La 1" in out.read_text(encoding="utf-8")


class TestDeviceGate:
    def test_a_requested_cuda_that_the_environment_cannot_see_is_a_hard_failure(
            self, worker, monkeypatch):
        """Not a silent fallback to CPU.

        The measured cause in this repository's environment is a torch wheel built for a CUDA
        newer than the driver, which is why the environment pins the cu126 train. Auto-
        downgrading would turn a misconfigured environment into a very slow, plausible run.
        """
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=False))
        with pytest.raises(worker.WorkerFailure, match="cu126"):
            worker.resolve_device("cuda")

    def test_auto_falls_back_to_cpu_and_says_why(self, worker, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=False))
        argument, name, reason = worker.resolve_device("auto")
        assert (argument, name) == ("cpu", "cpu")
        assert reason

    def test_auto_uses_the_gpu_when_torch_sees_it(self, worker, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=True))
        assert worker.resolve_device("auto") == (0, "cuda", None)

    def test_a_requested_cuda_that_exists_is_passed_as_the_library_expects(self, worker,
                                                                          monkeypatch):
        """ultralytics takes an int index for the GPU and the string "cpu" for the CPU."""
        monkeypatch.setitem(sys.modules, "torch", _torch_stub(cuda=True))
        assert worker.resolve_device("cuda") == (0, "cuda", None)


# ------------------------------------------------------- rendered-preview fakes


class _Frame:
    """A decoded BGR frame: what cv2 hands back, reduced to the two things used on it.

    The real thing is a numpy array. The worker only ever asks for ``shape`` and
    ``tobytes()``, and this suite has no numpy at all (that is what the separate uv project
    is for), so this stands in without pretending to be one — and the real-ffmpeg test below
    proves the bytes it produces are the bytes a rawvideo pipe wants.
    """

    def __init__(self, height: int = 48, width: int = 64, value: int = 7) -> None:
        self.shape = (height, width, 3)
        self._bytes = bytes([value]) * (height * width * 3)

    def tobytes(self) -> bytes:
        return self._bytes


@pytest.fixture
def fake_cv2(monkeypatch):
    """Record every drawing call instead of making pixels.

    cv2 lives in ``environments/persons`` and is not installed here, and the property worth
    testing is *what was asked to be drawn* (which box, which colour, which label, inside the
    frame), not the rasterisation of a stroke.
    """
    calls: list[tuple[Any, ...]] = []
    module = types.ModuleType("cv2")
    module.FONT_HERSHEY_SIMPLEX = 0

    def rectangle(image, pt1, pt2, colour, thickness=1, *args, **kwargs):
        calls.append(("rectangle", tuple(pt1), tuple(pt2), tuple(colour), int(thickness)))

    def put_text(image, text, org, font_face, font_scale, colour, thickness=1, *a, **k):
        calls.append(("text", str(text), tuple(org), round(float(font_scale), 4)))

    module.rectangle = rectangle
    module.putText = put_text
    monkeypatch.setitem(sys.modules, "cv2", module)
    return calls


FAKE_FFMPEG = '''
import json
import sys
from pathlib import Path

program = sys.argv[0]
argv = sys.argv[1:]
# Two invocations reach this stub per rendered run: the rawvideo encode and, when the source
# has audio, the mux pass. Only the encode consumes the frame pipe; the mux reads a finished
# file and must not block on stdin.
encoder = "rawvideo" in argv
data = sys.stdin.buffer.read() if encoder else b""
calls = json.loads(Path(LOG).read_text()) if Path(LOG).exists() else []
calls.append({"program": program, "argv": argv, "bytes": len(data),
              "frames": len(data) // FRAME_BYTES if encoder else None})
Path(LOG).write_text(json.dumps(calls))
if EXIT:
    sys.exit(EXIT)
if BODY:
    Path(argv[-1]).write_bytes(BODY)
'''


def write_fake_ffmpeg(tmp_path, *, exit_code: int = 0, body: bytes = b"fake-mp4",
                      frame_bytes: int = 48 * 64 * 3, name: str = "fake-ffmpeg"):
    """A stand-in ffmpeg that consumes the rawvideo pipe and reports what it saw.

    The argv of every invocation it was handed is recorded verbatim, in order, so the
    assertions are on the exact pipeline-owned command lines (rawvideo header, libx264, the
    mux pass's stream copy) rather than on what the worker meant to run — and on the
    destination path surviving intact.
    """
    script = tmp_path / name
    log = tmp_path / f"{name}-call.json"
    script.write_text(
        f"#!{sys.executable}\n"
        f"LOG={json.dumps(str(log))}\nFRAME_BYTES={frame_bytes}\n"
        f"EXIT={int(exit_code)}\nBODY={body!r}\n" + FAKE_FFMPEG,
        encoding="utf-8")
    script.chmod(0o755)
    return script, log


def write_fake_ffprobe(tmp_path, *, audio: bool = False, avg_frame_rate: str = "25/1",
                       fail: bool = False, name: str = "fake-ffprobe"):
    """A stand-in ffprobe answering with the streams a real clip would report."""
    streams: list[dict[str, Any]] = [{"index": 0, "codec_type": "video",
                                      "avg_frame_rate": avg_frame_rate}]
    if audio:
        streams.append({"index": 1, "codec_type": "audio"})
    script = tmp_path / name
    payload = "not json at all" if fail else json.dumps({"streams": streams})
    script.write_text(
        f"#!{sys.executable}\nimport sys\nprint({payload!r})\nsys.exit({1 if fail else 0})\n",
        encoding="utf-8")
    script.chmod(0o755)
    return script


def render_args(tmp_path, *, video=None, ffmpeg=None, ffprobe=None, dest=None,
                audio: bool = False, ffprobe_kwargs: dict | None = None):
    """The three render flags as the stage would pass them, plus the ffprobe stub."""
    ffprobe = ffprobe or write_fake_ffprobe(tmp_path, audio=audio,
                                           **(ffprobe_kwargs or {}))
    ffmpeg = ffmpeg or write_fake_ffmpeg(tmp_path)[0]
    dest = dest or tmp_path / "raw" / "tracks.mp4"
    return ["--output-video", str(dest), "--ffmpeg", str(ffmpeg), "--ffprobe", str(ffprobe)]


def read_call(log: Path) -> dict[str, Any]:
    """The encoder invocation: the first ffmpeg call of the run."""
    return read_calls(log)[0]


def read_calls(log: Path) -> list[dict[str, Any]]:
    return json.loads(log.read_text(encoding="utf-8"))


def scratch_files(directory: Path) -> list[Path]:
    """Encoder scratch a finished or failed run must not leave behind.

    Both the staging destinations and the ffmpeg stderr log are dot-files beginning with the
    destination's stem, so one glob covers the whole family.
    """
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.name.startswith(".tracks") or path.name.startswith(".video")
                  or path.name.startswith(".muxed"))


def _slice_index(argv: list[str], needle: list[str]) -> int:
    """Where a contiguous run of flags appears in argv, or fail with the whole command."""
    for start in range(len(argv) - len(needle) + 1):
        if argv[start:start + len(needle)] == needle:
            return start
    raise AssertionError(f"{needle} is not a contiguous part of {argv}")


class TestRenderedPreviewIsOptIn:
    """Nothing is drawn, probed or added to the document unless ``--output-video`` is given.

    The raw document is the preserved tool output: a key added to it whenever the stage runs
    would change every person artifact on disk the moment this ships, and the pipeline's own
    reuse test reads that file's bytes.
    """

    #: The top-level keys of the pre-render document, frozen. A key added while the toggle is
    #: off fails this, which is the only thing standing between "optional feature" and
    #: "every existing raw artifact now looks different".
    OFF_KEYS = (
        "device", "device_fallback_reason", "elapsed_seconds", "frames", "frames_measured",
        "frames_with_person", "frames_without_timestamp", "gmc_failure_count",
        "gmc_failure_reason", "max_persons_in_frame", "model", "parameters", "person_frame_counts",
        "person_ids", "requested_device", "schema_version", "torch_version", "tracker",
        "ultralytics_version", "video_id", "weights_path", "weights_sha256",
        "weights_size_bytes",
    )

    def test_the_document_shape_is_frozen_when_no_video_was_asked_for(self, worker, tmp_path,
                                                                     clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, [])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file)
        assert payload["status"] == "ok", payload.get("error")
        assert sorted(payload["_document"]) == sorted(self.OFF_KEYS)
        assert "render" not in payload["_document"]
        assert "render" not in payload

    def test_no_ffmpeg_is_run_and_no_video_appears_when_the_flag_is_absent(
            self, worker, tmp_path, clip, frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--ffmpeg", str(ffmpeg)])
        assert payload["status"] == "ok", payload.get("error")
        assert not log.exists(), "ffmpeg was run although no video was asked for"

    def test_a_rendered_run_adds_the_render_block_and_the_result_reports_it(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, []), FakeResult(2, [1])]
        args = render_args(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=args)
        assert payload["status"] == "ok", payload.get("error")
        expected = {"fps": 25.0, "fps_source": "frame_index", "frames_written": 3,
                    "width": 64, "height": 48, "encoded_width": 64, "encoded_height": 48,
                    "has_audio": False}
        assert payload["_document"]["render"] == expected
        assert payload["render"] == expected

    def test_the_rendered_document_keeps_every_measurement_key_it_already_had(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """Rendering is additive: the tables must not lose their provenance to a preview."""
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "ok", payload.get("error")
        assert set(self.OFF_KEYS) <= set(payload["_document"])
        assert payload["_document"]["frames_measured"] == 1


class TestDrawnOverlay:
    def test_one_box_per_detection_with_a_label_naming_the_id_and_score(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1, 2],
                                        xyxy=[[10.0, 20.0, 110.0, 220.0],
                                              [300.0, 40.0, 380.0, 200.0]],
                                        scores=[0.912, 0.42])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "ok", payload.get("error")
        boxes = [call for call in fake_cv2 if call[0] == "rectangle"]
        labels = [call for call in fake_cv2 if call[0] == "text"]
        assert [call[1] for call in boxes[:1]] == [(10, 20)]
        assert len(boxes) == 2 and len(labels) == 2
        assert labels[0][1] == "id=1 0.91"
        assert labels[1][1] == "id=2 0.42"

    def test_the_label_is_drawn_inside_the_frame_even_for_a_box_at_the_edge(self, worker):
        """cv2 draws text with a *baseline* origin, so y=0 writes above the frame.

        A label that lands outside is invisible, and a track whose id cannot be read is
        indistinguishable from a track that was never drawn.
        """
        inside = worker.label_origin(0, 0, width=64, height=48, scale=0.5)
        outside = worker.label_origin(10_000, 10_000, width=64, height=48, scale=0.5)
        for origin in (inside, outside):
            assert 0 <= origin[0] <= 63, origin
            assert 0 < origin[1] <= 46, origin

    def test_the_stroke_scales_with_the_frame(self, worker):
        """A 1px box on a 1080p frame is invisible; the same stroke on a thumbnail is the frame."""
        small = worker.stroke_width(240, 180)
        large = worker.stroke_width(1920, 1080)
        assert small >= 1 and large > small

    def test_a_tracks_colour_is_a_function_of_its_id_alone(self, worker):
        """Stable across runs, videos and process restarts, and not ``hash()``.

        The colour is the only thing tying a box to a number on screen, so it has to be the
        same number in a rerun of the same clip. Python's ``hash`` of a str is salted per
        process, which would repaint every id on every run and make two renders of one video
        disagree.
        """
        first = {person: worker.track_colour(person) for person in (1, 2, 3, 40, 41)}
        second = {person: worker.track_colour(person) for person in (41, 40, 4, 3, 2, 1)}
        assert first == {key: second[key] for key in first}
        assert len(set(first.values())) == len(first), "two ids share a colour"
        for colour in first.values():
            assert len(colour) == 3
            assert all(0 <= channel <= 255 for channel in colour)
            assert len(set(colour)) > 1, f"{colour} is unreadable against a plain background"

    def test_the_colour_survives_a_fresh_import_of_the_worker(self, worker, tmp_path, clip,
                                                             frame_index_file):
        """Across-process stability, as far as one process can observe it."""
        again = load_worker()
        assert again.track_colour(7) == worker.track_colour(7)

    def test_the_only_text_on_a_frame_is_the_track_label(self, worker, tmp_path, clip,
                                                        frame_index_file, fake_cv2):
        """No per-frame timestamps, no model name, no watermark.

        The preview's own clock is CFR arithmetic and is *not* the authoritative timeline, so a
        burned-in time would put a wrong number where a reader trusts the picture.
        """
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "ok", payload.get("error")
        texts = [call[1] for call in fake_cv2 if call[0] == "text"]
        assert texts == ["id=1 0.50", "id=1 0.50"]


class TestFrameAccounting:
    def test_every_decoded_frame_is_encoded_including_the_frames_with_nobody(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """Otherwise the preview is shorter than the video and its clock is wrong.

        A person-free frame produces no rows at all, so the tables cannot show whether those
        frames were ever fed to the encoder; the byte count the fake ffmpeg read is the check.
        """
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, []), FakeResult(2, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg))
        assert payload["status"] == "ok", payload.get("error")
        assert read_call(log)["frames"] == 3
        assert payload["_document"]["render"]["frames_written"] == 3

    def test_a_frame_of_a_different_size_is_refused(self, worker, tmp_path, clip,
                                                   frame_index_file, fake_cv2):
        """rawvideo has one fixed frame size for the whole pipe.

        Feeding it a second size would desynchronise every later frame into coloured noise and
        still exit 0, which is the worst possible failure of a preview: it looks like a video.
        """
        FakeModel.results = [FakeResult(0, [1]),
                             FakeResult(1, [1], frame=_Frame(32, 44))]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "error"
        assert "shape" in payload["error"]
        assert payload["_document"] is None

    def test_a_short_encode_is_refused_against_the_frames_measured(self, worker, tmp_path):
        """The number in the document has to be the number the encoder took.

        Checked against ``frames_measured`` rather than trusted from the pipe, because ffmpeg
        exits 0 after a truncated stdin as often as it exits non-zero.
        """
        ffmpeg, _log = write_fake_ffmpeg(tmp_path)
        writer = worker.TrackVideoWriter(ffmpeg=str(ffmpeg),
                                        destination=tmp_path / "out.mp4", fps=25.0,
                                        audio_source=None)
        writer.push(_Frame())
        writer.push(_Frame())
        with pytest.raises(worker.WorkerFailure, match="wrote 2 of 5"):
            writer.finish(expected_frames=5)


class TestFrameRate:
    def test_the_clock_comes_from_the_frame_index_that_is_already_loaded(self, worker):
        """The corpus is 30000/1001; a 25 fps assumption drifts, and the index is on disk."""
        assert worker.derive_fps({0: 0.0, 1: 0.04, 2: 0.08}) == pytest.approx(25.0)
        ntsc = {0: 0.0, 1: 1001 / 30000, 2: 2002 / 30000}
        assert worker.derive_fps(ntsc) == pytest.approx(30000 / 1001, abs=1e-3)

    @pytest.mark.parametrize("index", [{0: 0.0}, {}, {0: 1.0, 1: 1.0}, {0: 0.0, 1: -5.0}])
    def test_a_clock_that_cannot_be_derived_says_so_instead_of_guessing(self, worker, index):
        assert worker.derive_fps(index) is None

    def test_the_sources_average_frame_rate_is_the_fallback(self, worker, tmp_path, clip,
                                                           frame_index_file, fake_cv2):
        """One frame in the index means no span, and a preview is still worth having."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        one = tmp_path / "single.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([0], type=pa.int64()),
            "pts_seconds": pa.array([0.0], type=pa.float64()),
        }), one)
        FakeModel.results = [FakeResult(0, [1])]
        probe = write_fake_ffprobe(tmp_path, avg_frame_rate="30000/1001")
        payload = run_worker(worker, tmp_path, video=clip, frame_index=one,
                             extra=render_args(tmp_path, ffprobe=probe))
        assert payload["status"] == "ok", payload.get("error")
        render = payload["_document"]["render"]
        assert render["fps_source"] == "ffprobe_average_frame_rate"
        assert render["fps"] == pytest.approx(30000 / 1001, abs=1e-3)

    def test_no_frame_rate_at_all_is_refused_rather_than_encoded_at_a_guess(
            self, worker, tmp_path, clip, fake_cv2):
        """A guessed rate silently retimes the whole preview against every table."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        one = tmp_path / "single.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([0], type=pa.int64()),
            "pts_seconds": pa.array([0.0], type=pa.float64()),
        }), one)
        FakeModel.results = [FakeResult(0, [1])]
        probe = write_fake_ffprobe(tmp_path, avg_frame_rate="0/0")
        payload = run_worker(worker, tmp_path, video=clip, frame_index=one,
                             extra=render_args(tmp_path, ffprobe=probe))
        assert payload["status"] == "error"
        assert "fps" in payload["error"].lower()


class TestEncoderCommand:
    def test_the_pipe_header_and_encoder_are_the_ones_the_bytes_match(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1]), FakeResult(1, []), FakeResult(2, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg))
        assert payload["status"] == "ok", payload.get("error")
        argv = read_call(log)["argv"]
        assert read_call(log)["program"] == str(ffmpeg)
        _slice_index(argv, ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "64x48",
                            "-r", "25.000000", "-i", "-"])
        _slice_index(argv, ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        assert argv[-3:] == ["-movflags", "+faststart", argv[-1]]
        assert "-an" in argv and "-map" not in argv

    def test_the_source_audio_is_muxed_after_the_video_is_complete(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """Two passes on purpose: the second one cannot shorten the first one.

        Measured defect this replaces: a single call with the source as a second input and
        ``-shortest`` let the audio decide the video's length. The KABC clip is 126 frames
        with 4.200 s of audio against 4.204 s of video, and ffmpeg wrote 125 of them.
        """
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg, audio=True))
        assert payload["status"] == "ok", payload.get("error")
        calls = read_calls(log)
        assert len(calls) == 2, calls
        encode, mux = calls[0], calls[1]
        assert payload["_document"]["render"]["has_audio"] is True
        # The encode never sees the source, so no audio stream can end it early.
        assert str(clip) not in encode["argv"]
        assert "-an" in encode["argv"] and "-shortest" not in encode["argv"]
        assert "-map" not in encode["argv"]
        # The mux copies the finished video and pads the audio so -shortest ends at the video.
        _slice_index(mux["argv"], ["-i", encode["argv"][-1], "-i", str(clip),
                                    "-map", "0:v:0", "-map", "1:a:0",
                                    "-c:v", "copy", "-c:a", "aac", "-af", "apad",
                                    "-shortest"])
        assert mux["frames"] is None, "the mux pass read the frame pipe"

    def test_the_destination_path_travels_as_one_argument_not_as_a_shell_string(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """The path is the operator's, and the OpenPose stage already pays for a shell."""
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        awkward = tmp_path / "raw" / "who's track; rm -rf / .mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg, dest=awkward))
        assert payload["status"] == "ok", payload.get("error")
        argv = read_call(log)["argv"]
        staging = Path(argv[-1])
        assert staging.parent == awkward.parent, "the encoder staged outside the destination"
        assert staging.name != awkward.name
        assert staging.suffix == ".mp4", "ffmpeg picks the muxer from the suffix"
        assert awkward.is_file()

    def test_a_frame_that_cannot_be_read_is_refused(self, worker, tmp_path, clip,
                                                   frame_index_file, fake_cv2):
        """``orig_img`` is the only honest source of pixels; None means the run cannot be drawn."""
        FakeModel.results = [FakeResult(0, [1])]
        FakeModel.results[0].orig_img = None
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "error"
        assert "orig_img" in payload["error"]

    def test_an_ffmpeg_that_exits_non_zero_fails_the_run_before_any_document(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """No raw JSON, no tables, and the stage's staged tree is never published.

        The preserved raw document from a previous configuration is still on the real path at
        this point, so a failed render has to look like a failed run rather than a successful
        one that lost its video.
        """
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, _log = write_fake_ffmpeg(tmp_path, exit_code=4, body=b"")
        dest = tmp_path / "raw" / "tracks.mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg, dest=dest))
        assert payload["status"] == "error"
        assert "ffmpeg" in payload["error"]
        assert payload["_document"] is None
        assert not dest.exists()
        assert not scratch_files(dest.parent)

    def test_an_empty_output_is_refused(self, worker, tmp_path, clip, frame_index_file,
                                       fake_cv2):
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, _log = write_fake_ffmpeg(tmp_path, body=b"")
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg))
        assert payload["status"] == "error"
        assert "empty" in payload["error"]

    def test_a_probe_that_fails_is_refused_rather_than_rendering_a_silent_preview(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """Losing audio silently is the exact complaint about the OpenPose path's preview."""
        FakeModel.results = [FakeResult(0, [1])]
        probe = write_fake_ffprobe(tmp_path, fail=True)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffprobe=probe))
        assert payload["status"] == "error"
        assert "ffprobe" in payload["error"]

    def test_the_preview_is_published_in_one_step(self, worker, tmp_path, clip,
                                                 frame_index_file, fake_cv2):
        """A half-written mp4 in the staged tree would be published as if it were the render."""
        FakeModel.results = [FakeResult(0, [1])]
        dest = tmp_path / "raw" / "tracks.mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, dest=dest))
        assert payload["status"] == "ok", payload.get("error")
        assert dest.read_bytes() == b"fake-mp4"
        assert not scratch_files(dest.parent), "an encoder scratch file was left behind"


class TestRenderOutputCannotAliasAnInput:
    """ffmpeg overwrites what it is told to write, and it writes it during the tracking pass.

    The JSON guard already refuses an output that names the source video; a second output path
    added later needs the same guard or the preview render destroys the operator's clip while
    the run still looks healthy.
    """

    def test_an_output_video_naming_the_source_is_refused_before_tracking(self, worker,
                                                                        tmp_path, clip,
                                                                        frame_index_file):
        FakeModel.results = [FakeResult(0, [1])]
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--output-video", str(clip), "--ffmpeg", str(ffmpeg),
                                    "--ffprobe", str(write_fake_ffprobe(tmp_path))])
        assert payload["status"] == "error"
        assert "--output-video" in payload["error"]
        assert clip.read_bytes() == b"\x00"
        assert not log.exists()
        assert FakeModel.last_kwargs is None

    def test_an_output_video_naming_the_raw_document_is_refused(self, worker, tmp_path, clip,
                                                               frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1])]
        out = tmp_path / "raw" / "yolo_track.json"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, dest=out))
        assert payload["status"] == "error"
        assert "--output-video" in payload["error"]


class TestFrameSizeAndEncoderGrid:
    def test_an_odd_frame_size_is_padded_to_the_chroma_grid(self, worker, tmp_path, clip,
                                                          frame_index_file, fake_cv2):
        """libx264 with yuv420p refuses an odd side outright (measured: 65x49).

        The filter is a no-op on an even source, so the argv carries it unconditionally, and the
        size the encoder was fed is recorded beside the size that was decoded.
        """
        FakeModel.results = [FakeResult(0, [1], frame=_Frame(49, 65))]
        ffmpeg, log = write_fake_ffmpeg(tmp_path, frame_bytes=49 * 65 * 3)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path, ffmpeg=ffmpeg))
        assert payload["status"] == "ok", payload.get("error")
        argv = read_call(log)["argv"]
        assert "-vf" in argv, argv
        assert argv[argv.index("-vf") + 1] == "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        # The rawvideo header keeps the decoded size: those are the bytes on the pipe.
        _slice_index(argv, ["-s", "65x49"])
        render = payload["_document"]["render"]
        assert (render["width"], render["height"]) == (65, 49)
        assert (render["encoded_width"], render["encoded_height"]) == (66, 50)

    def test_an_even_frame_size_is_recorded_as_encoded_unchanged(self, worker, tmp_path, clip,
                                                                frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1])]
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "ok", payload.get("error")
        assert payload["_document"]["render"]["encoded_width"] == 64
        assert payload["_document"]["render"]["encoded_height"] == 48


class TestPathsThisRunUsesArePairwiseDistinct:
    """The reviewer's second data-loss finding, both halves of it.

    Measured: ``--output-video`` and ``--result-path`` at one path made the worker refuse the
    collision and *then* deliver that refusal by writing its error JSON over the previous
    render. And the encoder's scratch name was a fixed ``.<stem>.partial.mp4``, so a source
    clip that happened to be called that was overwritten before tracking finished.
    """

    def test_output_video_and_result_path_at_one_file_write_nothing(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1])]
        shared = tmp_path / "raw" / "tracks.mp4"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"an earlier render the operator still wants")
        code = worker.main([
            "--video", str(clip), "--frame-index", str(frame_index_file),
            "--output-json", str(tmp_path / "raw" / "yolo_track.json"),
            "--result-path", str(shared),
            "--video-id", "vid_a", "--device", "cpu",
            *render_args(tmp_path, dest=shared),
        ])
        assert code == 1
        assert shared.read_bytes() == b"an earlier render the operator still wants", \
            "the refusal was delivered by overwriting the file it refused to write"
        assert not (tmp_path / "raw" / "yolo_track.json").exists(), "tracking ran"
        assert FakeModel.last_kwargs is None

    def test_output_video_and_output_json_at_one_file_write_nothing(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        FakeModel.results = [FakeResult(0, [1])]
        shared = tmp_path / "raw" / "shared.json"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=shared,
                             extra=render_args(tmp_path, dest=shared))
        assert payload["status"] == "error"
        assert "--output-video" in payload["error"]
        assert FakeModel.last_kwargs is None
        assert not shared.exists(), "one of the two writers ran anyway"

    def test_a_source_named_like_the_old_scratch_path_survives_the_render(
            self, worker, tmp_path, clip, frame_index_file, fake_cv2):
        """The encoder stages under a kernel-allocated name, not a predictable one.

        Before, a destination ``tracks.mp4`` staged to ``.tracks.partial.mp4``; an operator's
        clip with that name was truncated by ffmpeg during the tracking pass, and the run
        reported success with the input gone.
        """
        FakeModel.results = [FakeResult(0, [1])]
        directory = tmp_path / "source-and-output"
        directory.mkdir()
        clip = directory / ".tracks.partial.mp4"
        clip.write_bytes(b"the operator's clip")
        probe = write_fake_ffprobe(tmp_path)
        ffmpeg, log = write_fake_ffmpeg(tmp_path)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--output-video", str(directory / "tracks.mp4"),
                                    "--ffmpeg", str(ffmpeg), "--ffprobe", str(probe)])
        assert payload["status"] == "ok", payload.get("error")
        assert clip.read_bytes() == b"the operator's clip", "the source clip was overwritten"
        assert (directory / "tracks.mp4").read_bytes() == b"fake-mp4"
        assert Path(read_call(log)["argv"][-1]) != clip

    def test_a_staging_path_that_names_an_input_is_refused(self, worker, tmp_path, clip,
                                                          frame_index_file, fake_cv2,
                                                          monkeypatch):
        """The uniqueness guarantee is checked, not assumed.

        ``mkstemp`` cannot hand out an existing name, so forcing one is the only way to see
        this branch -- and if that ever stops being true, the run must refuse rather than
        truncate the operator's clip.
        """
        FakeModel.results = [FakeResult(0, [1])]
        monkeypatch.setattr(worker, "staging_path", lambda destination, **kw: clip)
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=render_args(tmp_path))
        assert payload["status"] == "error"
        assert "encoder staging" in payload["error"]
        assert clip.read_bytes() == b"\x00"
        assert FakeModel.last_kwargs is None

    def test_the_staging_names_are_distinct_from_each_other(self, worker, tmp_path):
        writer = worker.TrackVideoWriter(ffmpeg="ffmpeg", destination=tmp_path / "tracks.mp4",
                                         fps=25.0, audio_source=None)
        assert not worker._same_path(writer._partial, writer._muxed)
        assert not worker._same_path(writer._partial, writer.destination)
        writer.abort()
        assert not scratch_files(tmp_path)


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe unavailable")


@requires_ffmpeg
class TestAgainstRealFfmpeg:
    """The fakes above prove the argv; only the real encoder proves the bytes fit it.

    A rawvideo pipe is unforgiving about stride, dtype and channel order, and every one of
    those mistakes produces a file that plays and looks like noise. So one run goes through
    the real binary with a real clip, and the result is re-probed.
    """

    def _clip(self, tmp_path, *, with_audio: bool):
        from tests.conftest import make_test_video

        return make_test_video(tmp_path / f"clip-{with_audio}.mp4", seconds=0.2, fps=25,
                              size="64x48", with_audio=with_audio)

    @pytest.mark.parametrize("with_audio", [True, False])
    def test_the_preview_encodes_and_carries_the_sources_audio(self, worker, tmp_path, fake_cv2,
                                                              with_audio):
        import shutil

        from multimodal_pipeline.stages.metadata import ffprobe_streams

        clip = self._clip(tmp_path, with_audio=with_audio)
        index = tmp_path / "frame_index.parquet"
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({
            "frame_number": pa.array([0, 1, 2, 3, 4], type=pa.int64()),
            "pts_seconds": pa.array([i * 0.04 for i in range(5)], type=pa.float64()),
        }), index)
        FakeModel.results = [FakeResult(i, [1] if i % 2 == 0 else []) for i in range(5)]
        dest = tmp_path / "raw" / "tracks.mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=index, extra=[
            "--output-video", str(dest), "--ffmpeg", "ffmpeg", "--ffprobe", "ffprobe"])
        assert payload["status"] == "ok", payload.get("error")
        render = payload["_document"]["render"]
        assert render["frames_written"] == 5
        assert render["has_audio"] is with_audio
        assert dest.is_file() and dest.stat().st_size > 0
        kinds = {stream["codec_type"] for stream in ffprobe_streams(dest)["streams"]}
        assert kinds == ({"video", "audio"} if with_audio else {"video"})
        video = next(s for s in ffprobe_streams(dest)["streams"] if s["codec_type"] == "video")
        assert int(video["nb_frames"]) == 5, "the preview is not the frames measured"

    def test_audio_shorter_than_the_video_does_not_shorten_it(self, worker, tmp_path, fake_cv2):
        """The measured defect behind the two-pass encoder, on real media.

        A single ffmpeg call with ``-shortest`` and the source as a second input stops the
        *video* when the audio runs out: the reviewer measured 125 frames from the repository's
        126-frame KABC clip, whose audio is 4.200 s against 4.204 s of video, and measured a
        15 s clip with a 2 s tone losing thirteen seconds of picture. So the clip here has ten
        frames (0.4 s at 25 fps) and a 0.05 s tone, and every frame has to survive.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from multimodal_pipeline.stages.metadata import ffprobe_streams

        clip = tmp_path / "short-audio.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=0.4",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=0.05",
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(clip)], check=True, capture_output=True)
        index = tmp_path / "frame_index.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array(list(range(10)), type=pa.int64()),
            "pts_seconds": pa.array([i * 0.04 for i in range(10)], type=pa.float64()),
        }), index)
        FakeModel.results = [FakeResult(i, [1]) for i in range(10)]
        dest = tmp_path / "raw" / "tracks.mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=index, extra=[
            "--output-video", str(dest), "--ffmpeg", "ffmpeg", "--ffprobe", "ffprobe"])
        assert payload["status"] == "ok", payload.get("error")
        assert payload["_document"]["render"]["has_audio"] is True
        decoded = ffprobe_count_frames(dest)
        assert decoded == 10, f"the preview kept {decoded} of 10 frames"
        audio = next(s for s in ffprobe_streams(dest)["streams"] if s["codec_type"] == "audio")
        assert float(audio["duration"]) >= 0.3, audio


    def test_an_odd_frame_size_reaches_the_encoder_as_an_even_one(self, worker, tmp_path,
                                                                 fake_cv2):
        """The pad filter against the real binary: 65x49 is refused by libx264 without it.

        Measured directly: ``ffmpeg -f lavfi -i testsrc=size=65x49 ... -c:v libx264 -pix_fmt
        yuv420p`` fails with ``width not divisible by 2 (65x49)`` and writes no file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from multimodal_pipeline.stages.metadata import ffprobe_streams

        odd = _Frame(49, 65)
        FakeModel.results = [FakeResult(i, [1], frame=odd) for i in range(3)]
        # The source only has to be probeable: the frame geometry comes from the frames the
        # model hands back, not from the container.
        from tests.conftest import make_test_video

        clip = make_test_video(tmp_path / "odd-source.mp4", seconds=0.2, fps=25,
                              size="64x48", with_audio=False)
        index = tmp_path / "frame_index.parquet"
        pq.write_table(pa.table({
            "frame_number": pa.array([0, 1, 2], type=pa.int64()),
            "pts_seconds": pa.array([i * 0.04 for i in range(3)], type=pa.float64()),
        }), index)
        dest = tmp_path / "raw" / "tracks.mp4"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=index, extra=[
            "--output-video", str(dest), "--ffmpeg", "ffmpeg", "--ffprobe", "ffprobe"])
        assert payload["status"] == "ok", payload.get("error")
        video = next(s for s in ffprobe_streams(dest)["streams"] if s["codec_type"] == "video")
        assert (video["width"], video["height"]) == (66, 50)
        assert payload["_document"]["render"]["encoded_width"] == 66
        assert ffprobe_count_frames(dest) == 3


def ffprobe_count_frames(path: Path) -> int:
    """Decoded frame count of a file's video stream, which is what a player sees."""
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout)["streams"]
    return int(streams[0]["nb_read_frames"])
