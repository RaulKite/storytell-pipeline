"""Sequential multimodal video-processing pipeline.

Turns a directory of video recordings into one self-contained multimodal
dataset directory per video: media metadata, audio, WhisperX transcription
with word alignment, Pyannote diarization, speaker-assigned transcript,
English translation, spaCy linguistics, Parselmouth acoustics and OpenPose
keypoints, normalised to Parquet on a single video timeline.
"""

__version__ = "0.1.0"

SCHEMA_VERSION = "1.0"
