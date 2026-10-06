# ELAN export

The `.eaf` the last stage writes: which fifteen tiers it builds, the semantics every label obeys, what the file says it left out, and how to eyeball or ship an export without ELAN installed.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [ELAN export: `elan`](#elan-export-elan)
- [Tier inventory](#tier-inventory)
- [Labels and timing](#labels-and-timing)
- [Measured corpus census](#measured-corpus-census)
- [Coverage, omissions and validation](#coverage-omissions-and-validation)
- [Reuse and timestamp refusal](#reuse-and-timestamp-refusal)
- [Eyeballing an `.eaf` without ELAN](#eyeballing-an-eaf-without-elan)
- [Sending an `.eaf` to someone who will open it in ELAN](#sending-an-eaf-to-someone-who-will-open-it-in-elan)

---

## ELAN export: `elan`

The last stage writes one `elan/annotations.eaf` per dataset: an ELAN annotation file that
links the source video and puts a fixed set of fifteen flat tiers over it, so the corpus is
readable in the tool linguists already use instead of only through Parquet. It is a
*derived summary*, not a new measurement and not a copy: it reads the tables the other stages
wrote, changes no number in them, and writes down the label runs an analyst reads — which means
it deliberately leaves things out (see the coverage note below). That is why it costs seconds,
why it runs after `finalization`, and why it is **on by default** while `persons` and
`diarization_nemotron` are off: it needs no GPU, no download, no credential.

Fifteen tiers, fixed names, no hierarchy — and **not** one tier per module. Those fifteen read
fourteen of the twenty-three normalised tables the pipeline publishes, because
`voiced_blocks`, `f0_blocks`, `intensity_blocks` and `formant_blocks` are four views of one
table (`acoustic/frame_features.parquet`) and `person_tracks` additionally reads
`persons/frames.parquet` and `source/frame_index.parquet` to decide where a sighting run ends
(below). Those two are **consumed support for the sightings tier** — they place the sightings the
tier reports — and neither is an exported analysis of its own: no tier is written *from* the frame
list or the per-frame detection table. The four frame tiers are flat peers of each other and of
the visual tiers: no tier is a parent of another, because a parent/child chain would make the
file's shape depend on the data and two datasets could no longer be compared tier-for-tier. What
gets no tier is `linguistic/*` (four tables), `acoustic/segment_features.parquet`, `pose/hands`,
`pose/face`, `pose/normalized` and `stories` — and the file says so. A tier is never silently
renamed to cover them (`manifest.artifacts_not_generated` is where absence is declared). Two
diarizers and two fusions are why fifteen tiers is more tiers than producers: an absent engine is
an absent tier, not a shared one.

### Tier inventory

| Tier | From | Annotation text |
|---|---|---|
| `words` | `speech/words.parquet` | `Hello · SPEAKER_00 · seg000001-w00000 · [seg000001]` |
| `segments_src` | `speech/segments.parquet` | `SPEAKER_00: source text · [seg000001]` |
| `gloss_en` | `translation/segments_en.parquet` | `SPEAKER_00: English text · [seg000001]` — **segment-level translation, not a word gloss** |
| `turns_pyannote` / `turns_nemotron` | each engine's turn table | `speaker SPEAKER_00 (pyannote, exclusive) · turn000001` |
| `fusion_pyannote` / `fusion_nemotron` | `speaker/fusion_*.parquet` | `face_matched: turn turn000001 · turn speaker SPEAKER_00 (pyannote)` then `\| face track 0 \|` and the verdict's arithmetic, verbatim |
| `asd_speaking` | `speaker/active_speaker_frames.parquet` | `speaking track 0` / `not speaking` / `not evaluated` / `no face`, **collapsed into runs** — a score carried forward adds `(imputed tail score)` on either activity state |
| `face_tracks` | `speaker/active_speaker_tracks.parquet` | `track 0 · 75/78 act · mean 2.505` — `mean` is a TalkNet logit-like score, unbounded, **not a probability** |
| `person_tracks` | `persons/tracks.parquet` + `persons/frames.parquet` + `source/frame_index.parquet` | `person 1 · sighting run 126 frames of 126 · conf 0.928 · track coverage 1.000 · max gap reported 0.033 s · covers src 0-125 · source adjacency verified` — and `sighting mark 1 frame` where adjacency did not hold |
| `pose_presence` | `pose/body.parquet` | `body present` runs (any keypoint ≥ 0.3) |
| `voiced_blocks` | `acoustic/frame_features.parquet` | `voiced (f0)` runs — contiguous runs of frames that carry a pitch at all |
| `f0_blocks` | `acoustic/frame_features.parquet` | `f0 med E3` on a 100 ms window: the semitone E3 contains the **low median** of the `f0_hz` values measured inside that window |
| `intensity_blocks` | `acoustic/frame_features.parquet` | `int med 62 to 64 dB` on the same 100 ms grid: the 2 dB bin holding that window's low median intensity, over **every** frame, voiced or not |
| `formant_blocks` | `acoustic/frame_features.parquet` | `med F1~600Hz F2~1200Hz F3~2000Hz` on the same grid: the band each of F1/F2/F3 falls in, from that window's three medians (300 Hz bins for F1, 600 Hz for F2, 1000 Hz for F3; `F3~1000Hz` means [1000, 2000) Hz) |

### Labels and timing

Seven rules the labels obey, because a tier value is the part of an `.eaf` that leaves the file
— into a screenshot, an issue, a talk — without this README next to it. They are also written
into the document itself as the `pipeline-tier-semantics` property, so the file explains its own
notation:

- **Ids are printed, and each says where it came from.** `words` carries its own `word_id` and the
  `segment_id` it belongs to; `segments_src` and `gloss_en` carry the `segment_id` they are keyed
  by, so the three text tiers link without eyeballing timestamps. `turn_id` appears on both the
  turn tier and the fusion tier built from it. What is *not* a link: pyannote's `SPEAKER_00`,
  Nemotron's arrival-ordered `speaker_0` and YOLO's `person_id` are three separate id spaces whose
  matching digits mean nothing (§20.2), so every label says which space its id came from — and the
  fusion tier prints the row's own `engine` column, because a quoted verdict otherwise arrives with
  no tier header to say which diarizer clustered its speaker. What *is* a link, and was described
  wrongly here until it was measured: the fusion row's `face_track_id` **is** the ASD tables'
  `track_id` — `fuse_turn_table` copies the winning frame's id through — so `fusion_*` and
  `face_tracks` join on it. Measured on this corpus, every non-null `face_track_id` is a member of
  the same dataset's `active_speaker_tracks.track_id` (KABC `{0}` and `{0, 1}` ⊆ `[0, 1]`; La-1
  `{0, 4}` ⊆ `[0, 1, 2, 4]`). It is not a *speaker* id, which is the distinction the schema is
  actually making.
- **A missing value says `unknown`.** A null, NaN or infinite score prints `unknown`, never
  `0.000`, because `mean 0.000` is a measurement and an analyst cannot tell the two apart; a real
  zero still prints `0.000`. Ids obey the same rule, including the nullable `speaker_id` at the
  head of `segments_src` and `gloss_en`, so an unassigned segment says `unknown:` rather than
  printing the word `None` in the part of the label an analyst quotes.
- **`not evaluated` is not `not speaking`, and neither is a claim about the audio.** A frame where
  a face was located but TalkNet never scored it (`face_status='tracked_unscored'`, or one of the
  five reasons that carry no measurement) reads `not evaluated`, and it outranks a stale
  `is_active_speaker` flag on the same row. Turning missing evidence into evidence of silence is
  the collapse the ASD schema grew `face_status` to prevent. `not speaking` *is* a measurement,
  but it is TalkNet's verdict on the **one track selected for that frame**: an off-screen narrator
  or a second person in shot can still be talking in the same second (the fusion table's
  `no_face_visible` verdict names that case), so the tier's own semantics property says the
  verdict is about that face's mouth and not about the audio. A score carried forward from the
  previous frame (`frame_reason='imputed_tail'`) says `(imputed tail score)` on **either**
  activity state — the provenance belongs to the number, not to the verdict, and this corpus has
  seven imputed rows across four datasets: five active and two **not** (La-1 frame 60 at 2.40 s,
  carried score −1.4667; `person_demo` frame 96 at 3.84 s). Before this correction both printed a
  plain `not speaking`.

- **A person id is a trajectory, and a sighting run is only as long as the source frames prove.**
  `person_id` is ByteTracker's, so ids are recycled and lost: `person_demo` reports **75 ids over
  205 sampled frames**, and the count of ids is not a count of people. The tier used to print one
  annotation per id from `first_timestamp` to `last_timestamp`, which is a *span* — it says the id
  was seen at both ends and nothing about the frames between. `persons/frames.parquet` records
  detections only (no stride, no sampling grid, no record of frames looked at and found empty), so
  the one adjacency that can be verified comes from `source/frame_index.parquet`: consecutive
  source frames the index places, whose PTS matches the row's own timestamp. Anything else — a
  frame the index does not name, a disagreeing timestamp, a missing or unreadable index, a frame
  the detector measured and did not report this id in — ends the run, and the label says which of
  the four states it is in (`source adjacency verified`, `run split at an unverifiable source
  frame`, `source adjacency unverified`, and `source adjacency unverified (no source frame index,
  coverage unknown)` when the index itself is gone). This is why La-1's id 10 is now **two runs** rather than
  one: source frames 112–114 and 144–239, with 29 source frames between them carrying no row for
  that id. Whether those frames were *sampled* and empty or never sampled is not in any table:
  La-1's raw document happens to report `frames_measured` 240 of the index's 240, but that number
  never reaches a Parquet column, so the tier prints neither inference. `max gap reported S s` is
  the track table's own `longest_gap_seconds` **printed as reported and never recomputed here** —
  naming the column is what makes the number checkable in the table instead of trusting the label's
  paraphrase. That column is the elapsed time between two consecutive **sightings**, which on a clip
  sampled at a fixed interval contains the sampling interval itself, so it is not a measure of
  absence: nothing here can distinguish "looked and did not see" from "never looked", and no label
  or property claims sampling coverage even where the grouping is verified. It prints `unknown`
  where the id has a single sighting, because the producer writes `0.0` there as a placeholder for
  "no pair exists" and `0.000` in a tier reads as "never lost sight of them". A person interval
  spans its own sightings' endpoints — it ends at the last sighting's measured **PTS**, with no grid
  step added, because nothing says the next frame was ever sampled. A single sighting keeps its
  measured time and is 1 ms wide only because ELAN cannot store a zero-width annotation and that
  width is a display minimum, not a duration. If
  `persons/frames.parquet` is absent or unreadable the tier is **skipped** with that dependency
  named: a track span is not a sighting history, so it is not exported under the word.
- **A bar on a Praat frame tier claims a bin, not a value, and only where the window was
  measured.** These three tiers exist because ELAN's time-series annotation type is not something the
  writer in use can produce, so a continuous signal enters as blocks — and a block invites the
  reading "this held for this long", which is exactly what a 10 ms frame measurement cannot support.
  Four rules keep that reading honest, and all four are in `pipeline-tier-semantics` inside the file.
  **Windows are 100 ms of wall-clock time, aligned to the clock**, not runs of an equal label, and a
  window's two edges come out of one function (`window_edges`) that both the membership test and the
  printed slot use. They were two float expressions until a corpus test caught a frame sitting inside
  window 9's median while window 9's bar ended at that frame's own instant: a bar narrower than the
  frames its label was computed from, invisible to every fixture because it needed the real 10 ms grid
  with its real float residues. **The label is the bin containing the window's low median**, and "low"
  is load-bearing: with an even count the ordinary median averages the middle two values, and the
  average of 100 Hz and 200 Hz is a pitch no frame ever had, while the low median is always one of the
  window's own measurements (`test_the_low_median_is_always_one_of_the_values`). That is why each
  label carries a `med` prefix and prints its bin: `f0 med E3` means *the semitone E3 contains the
  median of the frames under this bar*. **A window with fewer than half its frames measured gets no
  bar** — the rule is `fill < FRAME_MIN_FILL`, so 4 pitched frames in 10 is refused and 5 are labelled
  (`test_the_window_rule_refuses_below_half_and_labels_at_exactly_half`) — and a refused window
  **breaks** a bar rather than being spanned by one, so the hole stays visible
  (`test_a_refused_window_breaks_a_bar_rather_than_being_spanned_by_one`). Refusals are counted per
  tier and written to the run log, never into the document, because a reader of a sparse tier cannot
  reconstruct them from it. Refused windows per dataset, counted from each dataset's own frame table:

  | dataset | frames | `f0_blocks` | `intensity_blocks` | `formant_blocks` |
  |---|---|---|---|---|
  | KABC | 417 | 9 of 42 | 0 of 42 | 0 of 42 |
  | CNN | 410 | 14 of 41 | 0 of 41 | 0 of 41 |
  | La-1 | 798 | 31 of 80 | 0 of 80 | 0 of 80 |
  | `person_demo` | 408 | 32 of 41 | 0 of 41 | 0 of 41 |
  | `pipeline_demo` / `_ntsc` | 1,001 | 34 of 101 | 1 of 101 | 1 of 101 |
  | `pipeline_silent` | 400 | 40 of 40 | 0 of 40 | 40 of 40 |

  Pitch loses windows everywhere, which is what a pitch tracker does on unvoiced audio; the silent
  clip is the limit case, where Praat found no pitch and no formant in any of its 40 windows and both
  tiers are empty rather than guessed. An **empty** window is not counted as refused: "Praat measured
  too little here" and "the grid ended here" are two defects, and one number would mean both
  (`test_a_window_with_no_frames_at_all_is_not_counted_as_refused`). Adjacent windows whose *printed*
  label comes out alike share one bar — the only merging that happens, done on the printed string so
  two values quantising to the same bin cannot disagree about whether they merged.
- **The three frame tiers are three views of one table, and they are meant to disagree.**
  `voiced_blocks` groups contiguous runs of frames that carry a pitch at all; `f0_blocks` bins the
  same column's magnitude by window. Same column, different rules, so a `voiced_blocks` bar is longer
  and coarser than the `f0_blocks` bars inside it and neither is wrong. `intensity_blocks` is
  deliberately **not** voicing-filtered — Praat measured a level for every frame, voiced or not, and
  filtering by pitch presence would discard those measurements — which on `pipeline_silent` means the
  one clip with no pitch at all prints `int med -300 to -298 dB`, Praat's floor, while
  `f0_blocks`/`formant_blocks` are empty there. Formants are where a null bites, and it bites by
  **count, not by a hole in the label**: a null F2 thins the windows it sits in and past the line
  refuses them, for the formant tier only, because the other two never read F2
  (`test_one_null_formant_makes_only_the_formant_windows_thin`,
  `test_a_majority_of_null_formants_refuses_the_formant_window_only`). There is no per-formant marker:
  a label is built from three medians that all exist, and printing a band for a frequency nobody
  measured is the invented claim this export refuses everywhere else. The quantisations are pinned
  against the code, not the prose: 12 semitones per octave with A4 = 440 Hz, so 466.1638 Hz is `A#4`
  and 466.16 Hz is `A4` — the bin is a floor, and rounding upward would move a measurement to a pitch
  it never had; a 2 dB floor for intensity, printed `x to y dB` rather than `x-y dB` because Praat's
  floor is **-300 dB** and `-300--298 dB` is unreadable; formant bin widths of 300/600/1000 Hz for F1/F2/F3 respectively,
  each floored to its bin's lower edge — `F3~1000Hz` means [1000, 2000) Hz.
- **`none` and `unknown` are two words, and which one a null takes is decided per column.** The
  producers write the two states with different expressions, so one rule for both was a false claim.
  An empty value a producer writes *because the analysis answered "nothing here"* prints `none`; a
  null, NaN or infinite value, which means nothing reached the column, prints `unknown`
  (`ABSENT_DISPLAY`, `UNKNOWN_DISPLAY`, and `_field`'s `null_is_answer` flag is where a tier says
  which of its columns carry a measured absence). `conf` goes the other way again: the worker writes
  a *measured* `0.0` on every `unmatched`/`no_timing` row, so that prints `conf 0.000` and not
  `conf unknown`. A unit is never printed on a non-measurement: `unknown`, not `unknown Hz` or
  `unknown s`.

Per-frame signals collapse because 1,001 one-frame annotations per tier would be unusable in ELAN and
true to nothing. Two different rules collapse two different kinds of signal, and conflating them was
the defect the frame tiers nearly reintroduced. The tiers over a *timeline of states*
(`asd_speaking`, `pose_presence`) and over pitch presence (`voiced_blocks`) collapse into runs of an
equal label, and the block's end is one median grid-step past the last frame that carried the label —
stated because it is a choice, and a block then covers `start <= t < end` in ELAN's integer
milliseconds, where `end` is that extended value and not a further step on top of it. The three Praat
measurement tiers add **no** step: a bar ends at the window boundary it was computed from, because
that edge is a boundary of the interval the label describes, not a claim about how long one frame's
measurement lasted. **And none of this is true of `person_tracks`**, which adds no step either and
stops at the last sighting's own PTS; stating the grid-step extension as a global rule made a person
run look like a claim that the subject was still on screen one step past the last frame that placed
them, which is the inference this tier exists to refuse.
**Rows that overlap inside one tier are re-cut, because an ELAN tier is independent.** The manual is
explicit that two annotations in the same tier may not overlap in time, and the pipeline's own
tables contain simultaneous rows today: two people sighted in the same source frame, two TalkNet
face tracks alive over the same second, two Nemotron turns a fraction apart. pympi neither prevents
that nor reports it — the corpus's own `.eaf` files contained overlapping bars, and the round-trip
test never noticed because it only asked whether the bytes came back. So `build_eaf` now projects
each tier after the millisecond conversion and the missing/non-finite drops: the overlapping rows
are partitioned into the **disjoint half-open segments of a sweep over those rows' own endpoints**,
and every segment carries the text of every row active in it — one text as the plain label, two or
more as a JSON list (a delimiter would be ambiguous the moment a producer's own text contained it,
and ` · ` already sits inside these labels). Nothing else changes: no annotation is dropped, nothing
is staggered in time, no offset is invented, no two identities are equated and no text is
deduplicated, and a tier with no overlap is written exactly as before — same intervals, same plain
labels, no mention of any of this. Segments are a *projection of what the tables already said*, not
new events, and they are not always more numerous: where every instant of a span carries the same
set of labels, that span stays one annotation (CNN's four simultaneous sightings become **one**
bar holding four labels). What the segments cannot carry — each producer row's own interval, its
ids, its text, and which segments it ended up in — goes into the document's
`pipeline-overlap-projection` property, per affected tier only, alongside
`logical_row_count` (rows the table carried) and `final_annotation_count` (bars the tier emits).
Those are two different facts, the tier census counts only the second, and `validate` reads both
back out of the document and returns them as `projected_tiers`, which is how the difference reaches
`status.json` — that return value is the only thing the orchestrator persists from this stage, so it
is where a claim about the record has to be made or dropped. The property is what
makes the mapping from row to bars checkable after a split. `ElanStage.validate` then checks the
rule against the XML's own `TIME_SLOT` values — not against that property, because a metadata block
written by the same code that wrote the bars is not evidence about them — and reports same-tier
overlap, an unresolvable slot reference, and any interval with `start >= end`.

### Measured corpus census

Measured on the corpus, tier by tier, from the seven `.eaf` files on disk. **These are the current
export:** all seven were regenerated on 2026-10-05 by `multimodal-pipeline run --only-stage elan`
(7 completed, 0 failed, 11 s) after the five withdrawn tiers and the 100 ms window rule landed. The
previous table in this section compared an old file against a new one; this one states the state,
because the interesting fact is no longer what moved but what the frame tiers did to the totals.

| dataset | words | segments_src | gloss_en | turns_pyannote | turns_nemotron | fusion_pyannote | fusion_nemotron | asd_speaking | face_tracks | person_tracks | pose_presence | voiced_blocks | f0_blocks | intensity_blocks | formant_blocks | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KABC | 20 | 2 | 2 | 1 | 4 | 1 | 4 | 4 | 2 | 17 | 1 | 8 | 24 | 37 | 27 | 154 |
| CNN | 13 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 10 | 24 | 38 | 35 | 131 |
| La-1 | 17 | 4 | 4 | 2 | 6 | 2 | 6 | 11 | 5 | 9 | 1 | 29 | 45 | 64 | 44 | 249 |
| `person_demo` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 157 | 1 | 13 | 7 | 16 | 2 | 202 |
| `pipeline_demo` | 23 | 1 | 1 | 2 | 1 | 2 | 1 | 1 | 0 | 0 | 0 | 27 | 43 | 85 | 89 | 276 |
| `pipeline_demo_ntsc` | 23 | 1 | 1 | 2 | 1 | 2 | 1 | 1 | 0 | 0 | 0 | 27 | 43 | 85 | 89 | 276 |
| `pipeline_silent` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 |

Four things this table says out loud:

- **The frame tiers dominate the counts on a clip with speech.** On `pipeline_demo`, 217 of its 276
  annotations are `voiced_blocks`/`f0_blocks`/`intensity_blocks`/`formant_blocks`. That is what one
  bar per 100 ms window over 10 s of audio is, and it is the price of putting a continuous signal
  into a tool that only has blocks.
- **`pipeline_silent` emits 2 annotations and they are the right two**: one `asd_speaking` block and
  one `intensity_blocks` bar at Praat's floor. Every pitch and formant window on that clip was
  refused, so those tiers are empty, and the export does not stretch an empty tier into a bar.
- **`person_demo` has no transcript tiers at all** (its words table has 0 rows) and 157
  `person_tracks` bars, which are 138 logical sightings projected because ELAN tiers are independent.
- **The two `pipeline_demo` variants agree row for row.** Same tables, same projection, same numbers;
  they differ in the video container, not in what any stage measured.

All seven pass `validate`, and the projection was checked on the files rather than in memory: across
the **9 projected tiers**, **180 logical rows, 0 labels lost and 0 logical segments uncovered** —
every label the property records appears in some emitted bar, and every segment a row was split into
is covered by an annotation with no gap in between.

### Coverage, omissions and validation

Seven things the export does that are worth knowing before you open one:
- **The file says what it left out, and "left out" is two states.** An opened `.eaf` proves what
  it contains; nothing in it proves what was never exported. So a missing tier is ambiguous between
  "this clip has no person data" and "this export never represents pose", and only the first is a
  fact about the video. The `pipeline-coverage` property therefore names **every normalised Parquet
  table in the registry** — all 23, not the 12 the export builds tiers from — and gives each exactly one state:
  `exported` (a tier of this document is built from it, and the entry carries the tier's name),
  `summarised` (a tier reads it as support: `persons/frames.parquet` places every sighting bar and
  no bar is built *from* it), `present, not exported` (the file is on disk, nothing reads it, and the
  entry carries the reason), or `absent` (no file, so nothing could have been exported whatever the
  tier set said). Measured on this corpus: **12 exported, 2 summarised, 9 present-not-exported, 0
  absent** in the five datasets where every producer ran, and **12 + 2 + 8 + 1** on `person_demo` and
  `pipeline_silent`, where the one `absent` entry is `stories` — those two were processed before the
  stories stage existed, so no file was ever written for it. An entry naming a tier is a **list**
  whenever a table feeds several: `acoustic_frames` carries
  `["voiced_blocks", "f0_blocks", "intensity_blocks", "formant_blocks"]`, which is the only
  many-to-one mapping in the registry and the reason the count is 12 exported entries against 15
  tiers. The `absent` state used to be exercised only by synthetic fixtures; two real datasets now
  show it, and the distinction that matters is unchanged — a table with 0 rows is
  `present, not exported`, not `absent`, which is what keeps "the stage ran and found nothing" from
  reading as "the stage never ran". Two rules keep the property honest. It is **derived** from `ARTIFACT_LAYOUT`, `TIERS` and
  `SECONDARY_INPUTS` rather than typed into a list, so a table added to the registry tomorrow appears
  in it with the right state and no edit to a coverage list; and it consults the **disk first**, so a
  tier that exists in the code but whose producer never ran cannot report its table as `exported`.
  `absent` deliberately carries no reason: an artifact nobody wrote involved no export decision, and
  a reason there would be a true sentence about a deferral printed where a reader would take it for
  the cause of the absence. `present, not exported` carries a specific one — `pose_face` and
  `pose_hands`: *dense per-joint numeric tracks are not represented by this export* — and
  `pose_normalized` gets its own wording, because it is a change of basis over the same BODY_25
  keypoints `pose_presence` already blocks over, not a third set of joints. A table that is unread
  with no specific reason recorded says *that*, rather than borrowing a neighbour's sentence. Those
  three unread tables are not inputs to the stage — nothing is read from them — so the fingerprint
  also carries `inventory_present`, their existence flags. Without it, `pose/face.parquet` appearing
  would change what the document claims while leaving every config value, every table digest and the
  dependency hash identical, and the reuse check would keep a file that now says `absent` about a
  table that exists. Flags, not contents: unread bytes cannot move a state.
- **A tier with fewer bars than its table has rows says which rule took them.** `projected_tiers`
  answers "why are there more" (two rows shared an instant and were re-cut — a layout choice), and it
  reaches `status.json` through `validate`. The opposite question is answered by two counters,
  `missing_time` and `non_finite`, counted per tier and printed as two log lines: rows are refused,
  never placed at `t=0`, so the count is the only trace the refused rows leave. Those two are **run
  log and run output only** — they come from the build, not from the document, and `validate` reads
  the document, so nothing about them reaches `status.json`. Say plainly: on this corpus the answer
  is in `logs/elan.log`, and measured across all seven datasets every tier dropped **zero** rows, so
  no log line had to be read to find that out. The non-zero path is proven by fixtures the corpus
  does not contain (`test_a_built_tier_reports_its_two_drop_counters`,
  `test_the_record_reports_dropped_rows_only_for_the_tiers_that_refused_some`,
  `test_the_two_drop_states_are_never_summed_into_one_number`). Nothing about the drop rule changed:
  same two exceptions, same two log lines, same refusal.
- **The video is linked twice.** The `MEDIA_DESCRIPTOR` carries both an absolute `file://`
  URL and a `RELATIVE_MEDIA_URL` (`../../../input_videos/<name>` — relative to the `.eaf`
  itself in `elan/`, which is the base ELAN resolves against), because neither alone
  works: without the absolute one ELAN can't find the media on a normal open; without the
  relative one, copying `data/processed/` to another disk breaks every link although the
  video is still beside it. Measured on all seven datasets: both URLs resolve from disk.
  (The first version computed the relative path from the dataset directory instead of the
  `.eaf` directory — one `../` short, so all seven relative links were dead and the stage's
  own validation agreed with the writer. `test_validate_rejects_a_relative_url_based_on_the_dataset_directory`
  is the mutant-kill that keeps it from coming back.)
- **An unknown container gets an empty `MIME_TYPE`, never a guess.** `.mp4`/`.mov`/`.m4v`
  map to their real types; anything else declares nothing, because the attribute is
  optional, ELAN plays off the extension anyway, and a wrong type is a lie in the file.
  The rule is measured, not defensive: this corpus's `person_demo.avi` is really a
  QuickTime container, so a catch-all `video/mp4` would have been false on disk.
- **A missing table skips its tier with a logged reason; a missing transcript fails.** No
  words *and* no segments is a dataset nobody asked to annotate, so the stage fails loudly
  there; anything else (translation off, ASD off) just exports fewer tiers. A tier's *secondary*
  input is missing is the same outcome, with the dependency named — `person_tracks` is skipped
  when `persons/frames.parquet` is not there, and the other fourteen tiers still build. Absent and
  unreadable are reported as one state, because what the tier can do about either is the same.
- **Reuse sees the extra tables.** The fingerprint hashes every file in
  `ElanStage.inputs` — the twelve tier tables *plus* `person_frames` and `frame_index`, fourteen
  artifact names because `acoustic_frames` is one name that four tiers read — keyed by
  artifact name with `null` for an absent one, and it also carries the tier→dependency map itself,
  so a tier that starts reading one more table changes the hash even when every file is
  byte-identical.
- **`validate` checks the bars, not the export's story about them.** It resolves the document's own
  `TIME_SLOT` values and reports same-tier overlap, a slot reference that names no slot, and any
  interval with `start >= end` — while pointedly *not* reading `pipeline-overlap-projection`, which
  is the writer's account of the same file. It does cross-check `pipeline-coverage` against the
  document's own `TIER` elements — every tier the file declares must be claimed by some entry, and
  an entry must either name a tier or state one of the four known words — and that is the one
  property it reads, because the check is *internal* consistency rather than a claim about the
  tables. **One direction only, on purpose:** an entry naming a tier the document omits is what a
  *skipped* tier looks like (a table that exists but cannot be parsed loses its tier and keeps its
  artifact `exported`), and refusing that would put the reuse gate into a rerun loop over a partial
  export the writer produces deliberately. The census check already reports a tier deleted from the
  document, with the tier's name. An entry whose `tiers` is not a list, or whose list holds a
  non-string, is reported as an unreadable claim rather than acted on: the value comes out of JSON and
  goes into a set, so without that guard a hand edit raised `TypeError: unhashable type` out of
  `validate` — and since `validate`
  is the reuse gate, that crash stopped the pipeline where "no, rebuild it" was the answer that
  repairs the file. The registry is never consulted: the tree
  changes when a stage reruns, and a finished export must not become unvalidateable because a
  producer later wrote one more file. A missing property fails nothing (a document written before
  coverage existed still opens); a property that is not JSON, or a version this code cannot read,
  is refused. Since `validate` is also the reuse gate, a document that overlaps (an old export, or
  one edited by hand) is re-exported rather than reused.

### Reuse and timestamp refusal

Reuse works like every derived stage: its fingerprint mixes the digests of the tables it
reads **and** the Python that builds the tiers (§31), so editing `elan.py` re-runs it and an
untouched dataset re-uses in 0 s (`status --plan`: `valid previous result`). Times go
seconds → integer milliseconds (ELAN's unit), a zero-width interval widens by 1 ms because
pympi refuses a zero-length annotation, and no interval is ever negative. **A row whose producer
wrote no time is dropped and counted, never placed at t=0** — and the same refusal covers a time
**materially** below zero, which the converter's clamp used to launder into a `[0, 1)` ms bar at the
start of the clip; only the half-millisecond band the grid cannot tell apart from t=0 stays clamped
(`NEGATIVE_TOLERANCE_SECONDS`, a display-grid tolerance, not a recovered measurement). The
rule used to be that a null
timestamp landed at zero "so the annotation stays visible next to its siblings", and that was a
false claim: ELAN has no *time unknown* annotation, so an untimed row looked like something that
happened when the clip started — a person sighting with no timestamp exported as being on screen at
second zero, a word with a null `end_time` exported over `[0, 1)` ms. The check sits in
`interval_ms`, which every tier's rows pass through, so no builder can invent a time by omission,
and the drop is logged separately from a non-finite one (`dropped N of M annotation(s) with a
missing timestamp`) because the two describe different upstream defects.

### Eyeballing an `.eaf` without ELAN

`scripts/make_elan_view.py` renders one `.eaf` as a self-contained HTML page — one row per
tier, one block per annotation, full label on hover — for the machine that has no ELAN (this
server has none, so this is how the export was reviewed here). It never embeds the video: the
clips are gigabytes and a copy would be a second, staler source of truth. It is a viewing
aid, not a second validator, and it refuses three lies: the time axis names whether it is
the source media duration, a latest-annotation end, or an annotation end past the duration
(real diarizer turns run past it — La-1's last annotation ends at 8097 ms on an 8008 ms
clip); a bar carrying simultaneous labels keeps **all** of them and is flagged, because
collapsing it would hide the very simultaneity the export partitions to preserve; and an
empty tier prints as empty rather than vanishing.

```bash
uv run python scripts/make_elan_view.py \
  data/processed/<dataset>/elan/annotations.eaf /tmp/view.html
```

### Sending an `.eaf` to someone who will open it in ELAN

`scripts/make_review_bundle.sh` builds the `.tgz` a collaborator actually needs: two datasets'
full trees, each clip at the path its own `.eaf` points at, a regenerated `view.html` per dataset,
a cover note, and a sha256. It is a script rather than a directory copy because the two things that
silently break a hand-assembled bundle are invisible until ELAN is open, hours later, on someone
else's laptop:

- **The media layout is derived, not chosen.** The export computes `RELATIVE_MEDIA_URL` from the
  real output tree it was written into, so the `.eaf` asks for `../../../input_videos/<name>`. A
  bundle that picks a tidier name for the clip directory opens in ELAN with an empty grid and no
  waveform, which reads as "the export is broken" when the bundle was assembled wrong. The script
  parses that URL, refuses to guess if its depth is not the three levels it expects, and then
  resolves the URL from the `.eaf` before it will emit the tarball.
- **The endpoint host is discovered, not written here.** The script finds every `*base_url` host in
  the bundle's own `provenance/config.json`, rewrites it to `LLM-ENDPOINT.REDACTED` in the bundle
  copy, and then refuses to emit the tarball if the host is still anywhere in the tree. It does not
  name the host: a redaction that spells out what it hides publishes it, and the repository is the
  copy every future clone reads. API keys are already `***masked***` before they reach disk.
  Nothing under `data/` is ever rewritten — a bundle is a copy, and the byte-identical rule is about
  the corpus, not about what leaves it.

What the script deliberately leaves alone: `provenance/tools.json` names the machine that ran the
pipeline, and the stage logs quote absolute build paths. Both are the record that makes a number in a
bundle checkable, so falsifying them to look tidier trades a real property for a cosmetic one. A
bundle therefore carries a hostname and a filesystem layout; say so out loud when you forward one.

The cover note is deliberately not kept in the repository: every number in it is measured from one
export, so it is regenerated per bundle and passed in as `BUNDLE_README=/path/to/README.md`. The
script warns rather than fails when the file is missing, because the rebuild that deletes a
previously good note is the failure mode here.

```bash
BUNDLE_README=/tmp/bundle-notes/README-<date>.md \
  scripts/make_review_bundle.sh /tmp/storytel-bundle storytel-demo-<date>
```
