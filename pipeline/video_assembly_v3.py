"""v3 assembly: mixed generate/broll/stock segments, real durations, no
forced stretching. cell 38, unchanged logic."""
import os

from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, vfx, afx,
)

from . import config
from .video_assembly import ken_burns_clip


def build_segment_clip(segment, frame_size=None):
    """Turn one resolved plan segment into a moviepy clip of its own (real) duration."""
    frame_size = frame_size or config.FRAME_SIZE_V3
    fw, fh = frame_size

    if segment["source_type"] in ("broll", "stock"):
        clip = VideoFileClip(segment["path"]).without_audio()
        cover_scale = max(fw / clip.w, fh / clip.h)
        clip = clip.resized(cover_scale)
        clip = clip.with_position("center")
        return CompositeVideoClip([clip], size=frame_size).with_duration(clip.duration)

    return ken_burns_clip(segment["path"], segment["duration"], frame_size)


def build_clip_sequence_from_plan(segments, frame_size=None, transition_secs=None):
    frame_size = frame_size or config.FRAME_SIZE_V3
    transition_secs = config.TRANSITION_SECONDS if transition_secs is None else transition_secs

    clips = [build_segment_clip(s, frame_size) for s in segments]
    faded = []
    for i, c in enumerate(clips):
        effects = []
        if i > 0:
            effects.append(vfx.CrossFadeIn(transition_secs))
        if i < len(clips) - 1:
            effects.append(vfx.CrossFadeOut(transition_secs))
        faded.append(c.with_effects(effects) if effects else c)

    return concatenate_videoclips(faded, method="compose", padding=-transition_secs)


def assemble_chunk_video_v3(segments, narration_path, output_path, frame_size=None):
    """Same contract as assemble_chunk_video, but built from a mixed edit-list plan."""
    frame_size = frame_size or config.FRAME_SIZE_V3
    video = build_clip_sequence_from_plan(segments, frame_size)
    narration = AudioFileClip(narration_path).with_effects([afx.MultiplyVolume(config.NARRATION_VOLUME)])

    target_duration = max(video.duration, narration.duration)
    if video.duration < target_duration:
        video = video.with_effects([vfx.Loop(duration=target_duration)])
    video = video.with_duration(target_duration)

    # NOTE: the whoosh-transition sfx track assumes a fixed clip_duration to
    # compute transition timestamps, which v3's variable-length segments
    # don't have — narration-only audio for now.
    video = video.with_audio(narration)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", logger=None)
    return output_path
