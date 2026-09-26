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
                 scores: list[float] | None = None) -> None:
        self.frame_id = frame_id
        boxes = xyxy if xyxy is not None else [[0.0, 0.0, 10.0, 20.0] for _ in ids]
        confs = scores if scores is not None else [0.5] * len(ids)
        self.boxes = FakeBoxes(ids, boxes, confs)


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

    def test_auto_device_omits_the_argument_so_the_library_chooses(
            self, worker, tmp_path, clip, frame_index_file, monkeypatch):
        FakeModel.results = [FakeResult(0, [1])]
        monkeypatch.setattr(worker, "resolve_device",
                            lambda requested: (None, "cpu", "torch.cuda.is_available() was False"))
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             extra=["--device", "auto"])
        assert payload["status"] == "ok", payload.get("error")
        assert "device" not in FakeModel.last_kwargs
        assert payload["_document"]["device_fallback_reason"]


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
        """Otherwise the second write silently erases the tracking document."""
        FakeModel.results = [FakeResult(0, [1])]
        shared = tmp_path / "raw" / "one.json"
        payload = run_worker(worker, tmp_path, video=clip, frame_index=frame_index_file,
                             out=shared, result=shared)
        assert payload["status"] == "error"
        assert "--output-json" in payload["error"]

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
