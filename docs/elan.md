# ELAN export

The last stage writes a linked `elan/annotations.eaf` for inspection in ELAN.
It is a derived summary of selected tables, not the full dataset or a new measurement.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Tier inventory](#tier-inventory)
- [Timing and label rules](#timing-and-label-rules)
- [Coverage and validation](#coverage-and-validation)
- [Preview and refresh](#preview-and-refresh)
- [Share a bundle](#share-a-bundle)

## Tier inventory

There are up to **15 fixed, flat tiers**, with no parent/child hierarchy. A missing
input skips its tier with a reason; an existing empty input can yield an empty tier.
The stage needs transcript words or segments; absence of both skips the stage with
a reason. If those inputs disappear after enablement, execution can fail.

| Tier | Input / reading |
|---|---|
| `words` | Words with word ID, segment ID and assigned speaker |
| `segments_src` | Source transcript segments |
| `gloss_en` | English segment translation — **not a word gloss** |
| `turns_pyannote` | pyannote speaker turns |
| `turns_nemotron` | Nemotron turns, with independent speaker IDs |
| `fusion_pyannote` | pyannote turn × face agreement and arithmetic |
| `fusion_nemotron` | Nemotron turn × face agreement and arithmetic |
| `asd_speaking` | Selected-face activity runs; not a verdict about all audio |
| `face_tracks` | TalkNet face-track summaries; scores are not probabilities |
| `person_tracks` | Sighting runs proven by detections and source-frame adjacency |
| `pose_presence` | Body-present runs with a keypoint confidence floor of 0.3 |
| `voiced_blocks` | Runs of acoustic frames carrying pitch |
| `f0_blocks` | Low-median pitch bin in a 100 ms window |
| `intensity_blocks` | Low-median intensity bin in a 100 ms window, voiced or not |
| `formant_blocks` | F1/F2/F3 low-median bins in a 100 ms window |

No tier exports `linguistic/*`, acoustic segment statistics, hand/face landmarks,
normalized pose coordinates or `stories` candidates. These remain in the dataset.
`person_frames` and `frame_index` support the sightings tier without becoming
independent tiers. Exact mappings live in [`elan.py`](../src/multimodal_pipeline/elan.py).

## Timing and label rules

- Seconds become integer milliseconds. A valid zero-width mark widens to 1 ms for
  display; this does not claim a measured duration.
- Missing, non-finite and materially negative times are refused and counted, never
  placed at second zero. A half-millisecond negative tolerance is display rounding.
- Missing numerical evidence prints `unknown`, not a measured zero. Where a
  producer explicitly answered absence, labels can say `none` instead.
- ASD distinguishes `no face`, `not evaluated` and `not speaking`. The last means
  only that the **selected face** was judged inactive. An imputed tail score is
  labelled on either activity state.
- State runs (`asd_speaking`, `pose_presence`, `voiced_blocks`) extend by the median
  grid step. Person sightings **do not**: they stop at the last measured PTS, and
  unverified adjacency becomes isolated marks instead of invented continuity.
- Acoustic measurement windows use the **low median**, an actual value in the
  window, not the average of two middle values. Fewer than half the window's frames
  measured means no bar; a refused window breaks a run. Pitch bins are semitones
  (A4 = 440 Hz), intensity bins are 2 dB, and formant bins are 300/600/1000 Hz.
- Same-tier overlaps are partitioned at their original millisecond endpoints.
  Simultaneous labels become a JSON list with all texts retained, not staggered
  events or silently merged identities.

Person tables record detections, not proof that every frame was sampled. Even
verified source adjacency is not a claim of detector sampling coverage. Single
sightings cannot establish a gap statistic; their displayed gap is `unknown`.

## Coverage and validation

The export embeds self-description properties:

| Property | Purpose |
|---|---|
| `pipeline-tier-semantics` | Label and timing conventions |
| `pipeline-coverage` | Table treatment: exported, summarised, present-not-exported or absent |
| `pipeline-overlap-projection` | Original logical intervals/labels and their mapping to projected bars |

Coverage is not accuracy or complete information preservation. Bars, logical rows
and input-table rows can have different counts. `validate` checks XML time slots,
positive intervals, same-tier nonoverlap, media links and coverage consistency;
valid XML alone is not enough.

The stage's validation results in `status.json` include coverage/projection details.
Rows refused while building and thin-window counts are reported in `logs/elan.log`,
not recoverable as complete build counters from the finished XML.

Both absolute and relative video URLs are written. The relative URL is based on
the `.eaf` directory, **not** the dataset directory. Moving an export without its
referenced media layout can break playback. Neither the HTML preview nor unit
validation proves the labels are comfortable to use in ELAN on a desktop.

## Preview and refresh

```bash
uv run python scripts/make_elan_view.py \
  data/processed/pipeline_demo/elan/annotations.eaf /tmp/view.html
uv run multimodal-pipeline status -c config/config.local.yaml --plan
uv run multimodal-pipeline run -c config/config.local.yaml --only-stage elan
```

Replace the demo dataset path if necessary. The HTML is self-contained for tier
inspection, but **does not embed/play the source video** and is not another validator.
It preserves simultaneous labels and empty tiers and names the basis of its time axis.

ELAN reuse fingerprints include consumed tables, dependency mappings and builder
code. `--only-stage elan` includes prerequisites: it avoids producer work only when
those prerequisites are reusable. Inspect the plan first; turning ELAN off deletes
nothing and does not make other dataset outputs unavailable.

## Share a bundle

`scripts/make_review_bundle.sh` is a **local corpus-specific helper**, not a generic
export command. It selects two hardcoded broadcast datasets under `data/processed/`
and requires `config/config.local.yaml`; inspect the script before using another
corpus. It copies data into a staging directory,
generates HTML previews and emits a `.tgz` plus checksum.

```bash
BUNDLE_README=/tmp/bundle-notes/README.md \
  scripts/make_review_bundle.sh /tmp/storytel-bundle storytel-demo
```

The helper derives clip placement from each `.eaf`'s `RELATIVE_MEDIA_URL` and refuses
an unsupported layout rather than guessing. It discovers endpoint hosts from copied
provenance, redacts them in the copy and checks they are absent before packaging.
Original datasets are not rewritten. An optional cover note is supplied through
`BUNDLE_README`; missing notes warn rather than abort.

**Inspect before sharing:** media, transcript content, machine hostnames, absolute
paths and raw responses may still be sensitive or copyrighted. Endpoint redaction
is not complete anonymization, and creating a local bundle does not authorize
publishing it. For the full output files, see [Datasets](datasets.md).
