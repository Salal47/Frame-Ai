"""v2 video assembly: Ken-Burns clips, crossfade transitions, whoosh sfx,
soft ambient pad fallback, final concatenation + background music. cells 22-24."""
import os
import random
import wave

import numpy as np
from moviepy import (
    ImageClip, VideoFileClip, AudioFileClip,
    CompositeVideoClip, CompositeAudioClip,
    concatenate_videoclips, vfx, afx,
)

from . import config

KEN_BURNS_EFFECTS = [
    "zoom_in", "zoom_out",
    "zoom_in_pan_left", "zoom_in_pan_right", "zoom_in_pan_up", "zoom_in_pan_down",
    "zoom_out_pan_left", "zoom_out_pan_right", "zoom_out_pan_up", "zoom_out_pan_down",
]


def ken_burns_clip(image_path, duration, frame_size=None, effect=None):
    """Turn a still image into an animated clip with a randomised zoom/pan move."""
    frame_size = frame_size or config.FRAME_SIZE
    effect = effect or random.choice(KEN_BURNS_EFFECTS)
    zoom_in = "zoom_out" not in effect
    pan = next((d for d in ("pan_left", "pan_right", "pan_up", "pan_down") if d in effect), None)
    fw, fh = frame_size

    base = ImageClip(image_path, duration=duration)
    cover_scale = max(fw / base.w, fh / base.h)
    base = base.resized(cover_scale)

    z0, z1 = (1.0, 1.0 + config.ZOOM_AMOUNT) if zoom_in else (1.0 + config.ZOOM_AMOUNT, 1.0)

    def scale_at(t):
        p = min(max(t / duration, 0), 1)
        return z0 + (z1 - z0) * p

    zoomed = base.resized(scale_at)

    def position_at(t):
        p = min(max(t / duration, 0), 1)
        s = scale_at(t)
        cur_w, cur_h = base.w * s, base.h * s
        max_dx, max_dy = max(cur_w - fw, 0), max(cur_h - fh, 0)
        x_frac = {"pan_left": 1 - p, "pan_right": p}.get(pan, 0.5)
        y_frac = {"pan_up": 1 - p, "pan_down": p}.get(pan, 0.5)
        return (-(max_dx * x_frac), -(max_dy * y_frac))

    positioned = zoomed.with_position(position_at)
    return CompositeVideoClip([positioned], size=frame_size).with_duration(duration)


def build_clip_sequence(image_paths, clip_duration=None, frame_size=None, transition_secs=None):
    """Turn a list of images into one video: animated clips crossfading into each other."""
    clip_duration = clip_duration or config.CLIP_SECONDS
    frame_size = frame_size or config.FRAME_SIZE
    transition_secs = config.TRANSITION_SECONDS if transition_secs is None else transition_secs

    used_effects = []
    clips = []
    for p in image_paths:
        choices = [e for e in KEN_BURNS_EFFECTS if not used_effects or e != used_effects[-1]]
        effect = random.choice(choices)
        used_effects.append(effect)
        clips.append(ken_burns_clip(p, clip_duration, frame_size, effect))

    faded = []
    for i, c in enumerate(clips):
        effects = []
        if i > 0:
            effects.append(vfx.CrossFadeIn(transition_secs))
        if i < len(clips) - 1:
            effects.append(vfx.CrossFadeOut(transition_secs))
        faded.append(c.with_effects(effects) if effects else c)

    return concatenate_videoclips(faded, method="compose", padding=-transition_secs)


def synthesize_whoosh(path, duration=0.5, sr=24000):
    """Generate a short descending-pitch 'whoosh' for clip transitions (no external assets needed)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq_sweep = np.linspace(1200, 200, len(t))
    tone = np.sin(2 * np.pi * np.cumsum(freq_sweep) / sr)
    noise = np.random.normal(0, 0.15, len(t))
    envelope = np.sin(np.pi * t / duration)
    signal = (tone * 0.6 + noise) * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8
    data = np.int16(signal * 32767)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


def synthesize_soft_pad(path, duration, sr=24000):
    """Generate a soft, low ambient pad as a background-music fallback (no external assets needed)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [110.0, 165.0, 220.0]
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    breathing = 0.7 + 0.3 * np.sin(2 * np.pi * t / 8.0)
    signal = signal * breathing
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.5
    data = np.int16(signal * 32767)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


def transition_timestamps(num_clips, clip_duration, transition_secs):
    step = clip_duration - transition_secs
    return [(i + 1) * step + transition_secs / 2 for i in range(num_clips - 1)]


def build_transition_sfx_track(num_clips, clip_duration, transition_secs, sfx_path, volume):
    if num_clips < 2:
        return None
    if not os.path.isfile(sfx_path):
        synthesize_whoosh(sfx_path)

    base_sfx = AudioFileClip(sfx_path).with_effects([afx.MultiplyVolume(volume)])
    timestamps = transition_timestamps(num_clips, clip_duration, transition_secs)
    sfx_clips = [base_sfx.with_start(max(ts - base_sfx.duration / 2, 0)) for ts in timestamps]
    return CompositeAudioClip(sfx_clips)


def assemble_chunk_video(image_paths, narration_path, output_path,
                          clip_duration=None, frame_size=None):
    """One narration chunk -> one video: animated clips + transitions + narration + whoosh sfx."""
    clip_duration = clip_duration or config.CLIP_SECONDS
    frame_size = frame_size or config.FRAME_SIZE

    video = build_clip_sequence(image_paths, clip_duration, frame_size)
    narration = AudioFileClip(narration_path).with_effects([afx.MultiplyVolume(config.NARRATION_VOLUME)])

    target_duration = max(video.duration, narration.duration)
    if video.duration < target_duration:
        video = video.with_effects([vfx.Loop(duration=target_duration)])
    video = video.with_duration(target_duration)

    sfx_track = build_transition_sfx_track(
        len(image_paths), clip_duration, config.TRANSITION_SECONDS,
        config.TRANSITION_SFX_PATH, config.TRANSITION_SFX_VOLUME,
    )
    audio_layers = [narration] + ([sfx_track] if sfx_track else [])
    video = video.with_audio(CompositeAudioClip(audio_layers))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", logger=None)
    return output_path


def assemble_final_video(chunk_video_paths, output_path, bg_music_path=None, bg_volume=None):
    """Concatenate all chunk videos and mix in a soft, low-volume background track."""
    bg_music_path = bg_music_path or config.BG_MUSIC_PATH
    bg_volume = config.BG_MUSIC_VOLUME if bg_volume is None else bg_volume

    clips = [VideoFileClip(p) for p in chunk_video_paths]
    final = concatenate_videoclips(clips, method="compose")

    if not os.path.isfile(bg_music_path):
        print("   no assets/bg_music.mp3 found — synthesizing a soft ambient pad instead")
        synthesize_soft_pad(bg_music_path, duration=final.duration)

    music = AudioFileClip(bg_music_path)
    if music.duration < final.duration:
        music = music.with_effects([afx.AudioLoop(duration=final.duration)])
    else:
        music = music.subclipped(0, final.duration)
    music = music.with_effects([afx.MultiplyVolume(bg_volume)])

    final = final.with_audio(CompositeAudioClip([final.audio, music]))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", logger=None)
    return output_path
