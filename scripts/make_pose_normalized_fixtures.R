#!/usr/bin/env Rscript
# Regenerate the committed dfMaker reference fixtures for the pose_normalized stage.
#
# Why this script is committed: tests/fixtures/pose_normalized/*.csv are the ground truth
# the Python transform is asserted against (ODD section 20/21), and a ground truth that
# nobody can say how it was made is a number someone typed. This is the whole recipe.
#
# The reference is CRAN `multimolang`, whose only exported function is dfMaker, and whose
# only import is `arrow`. Install it somewhere private -- do NOT install into the
# operator's R library:
#
#   mkdir -p /tmp/rlibs
#   Rscript -e 'r <- getOption("repos"); r["CRAN"] <- "https://cloud.r-project.org"; \
#     options(repos = r); install.packages("multimolang", lib = "/tmp/rlibs", \
#     dependencies = "Imports")'
#
# Usage:
#   Rscript scripts/make_pose_normalized_fixtures.R <processed-dir> <frames> <out-dir>
#
# Example (writes the two committed files, after checking them against git first):
#   Rscript scripts/make_pose_normalized_fixtures.R data/processed 5 /tmp/fixtures
#
# What is deliberately NOT reproduced here, and why it is still honest:
# the CSVs are an exact row prefix of dfMaker's output -- every kept row is verbatim, no
# coordinate is touched -- so each one remains "what dfMaker 0.1.1 printed for this
# frame". They are not the whole output: the full run over all 695 frames of the four
# videos with pose produced 6300 + 11450 + 12400 + 45575 rows. The committed subset stops
# at 5 frames because the full set is ~350 KB of digits, which is neither reviewable
# content nor a reviewable candidate (the native reviewer budget rejected a change
# carrying it). 5 frames is the smallest prefix that still contains both basis states --
# rows dfMaker could place and rows it refused to place -- which is what the guard needs
# to be able to see.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: make_pose_normalized_fixtures.R <processed-dir> <frames> <out-dir>")
processed <- args[1]
max_frame <- as.integer(args[2])
out_dir <- args[3]

.libPaths(c("/tmp/rlibs", .libPaths()))
suppressMessages(library(multimolang, lib.loc = "/tmp/rlibs"))
cat("multimolang", as.character(packageVersion("multimolang")), "| R", R.version.string, "\n")

# The triple dfMaker would default to is c(1, 1, 5, 5): origin Neck, basis Neck->LShoulder.
# That is a 17.7 px basis on this corpus and its normalised coordinates reach p99 = 556,
# so the pipeline uses MidHip (8) as origin and Neck (1) as the basis vector instead -- a
# 72.4 px basis, p99 = 2.07. `fast_scaling = FALSE` is what selects the change-of-basis
# branch rather than the divide-by-one-axis branch.
# See ODD section 20 for the measurement table.
TRANSFORMATION_COORDS <- c(1, 8, 1, 1)

# One fixture per video, chosen to cover two shots and two speaker counts rather than to
# be a sample: KABC is a two-person interview, CNN a four-person panel.
CLIPS <- list(
  kabc = "2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear",
  cnn = "2017-12-30_1930_US_CNN_Global_Warning_Arctic_Melt_1237_273_1241_393_hear"
)

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (tag in names(CLIPS)) {
  raw_dir <- file.path(processed, CLIPS[[tag]], "pose", "raw")
  if (!dir.exists(raw_dir)) stop("no raw pose JSON at ", raw_dir)
  # Sorted, then the first N: the frame number is zero-padded to 12 digits in the file
  # name, so lexical order is numeric order and "the first N files" is "frames 0..N-1".
  files <- sort(list.files(raw_dir, pattern = "*.json", full.names = TRUE))
  if (length(files) < max_frame) {
    stop("only ", length(files), " raw frames available, asked for ", max_frame)
  }
  keep <- head(files, max_frame)

  staged <- file.path(tempdir(), paste0("ref_", tag))
  unlink(staged, recursive = TRUE); dir.create(staged)
  file.copy(keep, staged)

  # no_save = TRUE keeps dfMaker from writing, then it is read back from the returned
  # data frame: this is the same object, no file format in between to differ in.
  result <- as.data.frame(dfMaker(
    input.folder = staged,
    no_save = TRUE,
    fast_scaling = FALSE,
    transformation_coords = TRANSFORMATION_COORDS
  ))

  # pose_keypoints only: hands and face have their own point counts, and the stage
  # normalises the body table. `points` stays OpenPose's 0-based index, which is the same
  # number as our keypoint_id -- verified, not assumed.
  result <- result[result$type_points == "pose_keypoints",
                   c("id", "frame", "people_id", "points", "x", "y", "c", "nx", "ny")]

  out <- file.path(out_dir, sprintf("dfmaker_0.1.1_%s_midhip_neck.csv", tag))
  # quote = "" because the committed files are read by csv.DictReader on the Python side and
  # the id contains no delimiter: quoting it would make regeneration differ from what is
  # committed in bytes while agreeing in content, which is a bad thing to discover at 1am.
  write.csv(result, out, row.names = FALSE, na = "NA", quote = FALSE)
  # nx is numeric NA in the returned frame, so "!= \"NA\"" would coerce and return NA.
  placed <- sum(!is.na(result$nx))
  cat(sprintf("%-5s %5d rows (%d placed, %d refused) -> %s\n",
              tag, nrow(result), placed, nrow(result) - placed, out))
}
