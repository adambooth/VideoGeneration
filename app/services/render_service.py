from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


class RenderService:
    def ensure_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("FFmpeg and ffprobe must be installed and available on PATH.")

    def get_video_duration(self, video_path: str) -> float:
        self.ensure_ffmpeg()
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe failed to inspect video duration: {completed.stderr[-1000:]}")
        return max(0.1, float(completed.stdout.strip()))

    def get_audio_duration(self, audio_path: str) -> float:
        self.ensure_ffmpeg()
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe failed to inspect audio duration: {completed.stderr[-1000:]}")
        return max(1.0, float(completed.stdout.strip()))

    def render_video(
        self,
        project_dir: str,
        scene_assets: list[dict],
        voice_path: str,
        music_path: str,
        output_path: str,
        fps: int,
        quality: str,
        voice_segment_durations: list[float] | None = None,
    ) -> str:
        self.ensure_ffmpeg()
        project_path = Path(project_dir)
        working_path = project_path / "render_work"
        working_path.mkdir(exist_ok=True)

        native_audio_mode = not bool(voice_path)
        voice_duration = self.get_audio_duration(voice_path) if voice_path else 0.0
        durations = self._build_durations(scene_assets, voice_duration, voice_segment_durations or [])

        scene_clips: list[Path] = []
        for index, asset in enumerate(scene_assets):
            clip_path = working_path / f"clip_{index:02d}.mp4"
            if asset.get("asset_type") == "video" and asset.get("asset_path"):
                self._create_video_clip(asset, durations[index], clip_path, fps, keep_audio=native_audio_mode)
            else:
                self._create_image_clip(asset, durations[index], clip_path, fps, include_silent_audio=native_audio_mode)
            scene_clips.append(clip_path)

        concat_list = working_path / "clips.txt"
        concat_list.write_text(
            "\n".join(f"file '{clip.as_posix()}'" for clip in scene_clips),
            encoding="utf-8",
        )
        stitched_video = working_path / "stitched.mp4"
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-fflags",
                "+genpts",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                str(stitched_video),
            ]
        )

        crf = {"Low": "28", "Medium": "23", "High": "18"}.get(quality, "18")
        if not native_audio_mode:
            music_input = [
                "-stream_loop",
                "-1",
                "-i",
                music_path,
            ]
            filter_complex = (
                "[2:a]volume=0.08[bg];"
                "[bg][1:a]sidechaincompress=threshold=0.025:ratio=10:attack=15:release=380[ducked];"
                "[1:a][ducked]amix=inputs=2:duration=first:weights='1 0.38'[mix]"
            )
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(stitched_video),
                    "-i",
                    voice_path,
                    *music_input,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "0:v:0",
                    "-map",
                    "[mix]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    crf,
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    output_path,
                ]
            )
            return output_path

        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stitched_video),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                crf,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        return output_path

    def extract_audio(self, video_path: str, output_path: str) -> str:
        self.ensure_ffmpeg()
        if not self._has_audio_stream(video_path):
            raise RuntimeError(f"The Veo clip has no audio stream to extract: {video_path}")
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path,
            ]
        )
        return output_path

    def split_audio_for_turns(
        self,
        audio_path: str,
        expected_count: int,
        output_dir: str,
        prefix: str,
    ) -> list[str]:
        self.ensure_ffmpeg()
        total_duration = self.get_audio_duration(audio_path)
        if expected_count <= 1:
            output = Path(output_dir) / f"{prefix}_turn_01.mp3"
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio_path, output)
            return [str(output)]

        boundaries = self._turn_boundaries_from_silence(audio_path, expected_count, total_duration)
        segment_paths: list[str] = []
        segment_dir = Path(output_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)
        start_time = 0.0
        points = [*boundaries, total_duration]
        for index, end_time in enumerate(points, start=1):
            segment_path = segment_dir / f"{prefix}_turn_{index:02d}.mp3"
            duration = max(0.28, round(end_time - start_time, 3))
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_time:.3f}",
                    "-i",
                    audio_path,
                    "-t",
                    f"{duration:.3f}",
                    "-ac",
                    "2",
                    "-ar",
                    "44100",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(segment_path),
                ]
            )
            segment_paths.append(str(segment_path))
            start_time = end_time
        return segment_paths

    def concat_audio_segments(self, segment_paths: list[str], output_path: str) -> str:
        self.ensure_ffmpeg()
        if not segment_paths:
            raise ValueError("No audio segments were provided for concatenation.")
        if len(segment_paths) == 1:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(segment_paths[0], target)
            return str(target)

        concat_list = Path(output_path).with_name(f"{Path(output_path).stem}_segments.txt")
        concat_list.write_text(
            "\n".join(f"file '{Path(path).resolve().as_posix()}'" for path in segment_paths),
            encoding="utf-8",
        )
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path,
            ]
        )
        return output_path

    def create_silence(self, output_path: str, duration_seconds: float) -> str:
        self.ensure_ffmpeg()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t",
                f"{max(0.2, duration_seconds):.2f}",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                output_path,
            ]
        )
        return output_path

    def extract_last_frame(
        self,
        video_path: str,
        output_path: str,
        prefer_two_subjects: bool = False,
        exact_last_frame: bool = False,
    ) -> str:
        self.ensure_ffmpeg()
        if exact_last_frame:
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-sseof",
                    "-0.08",
                    "-i",
                    video_path,
                    "-update",
                    "1",
                    "-frames:v",
                    "1",
                    output_path,
                ]
            )
            return output_path
        duration = self.get_video_duration(video_path)
        candidate_ratios = [0.68, 0.74, 0.80, 0.86, 0.92]
        candidate_paths: list[tuple[Path, float]] = []
        output_target = Path(output_path)
        output_target.parent.mkdir(parents=True, exist_ok=True)
        for idx, ratio in enumerate(candidate_ratios, start=1):
            candidate_path = output_target.with_name(f"{output_target.stem}_candidate_{idx:02d}{output_target.suffix}")
            timestamp = max(0.1, round(duration * ratio, 2))
            try:
                self._run_ffmpeg(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        f"{timestamp:.2f}",
                        "-i",
                        video_path,
                        "-update",
                        "1",
                        "-frames:v",
                        "1",
                        str(candidate_path),
                    ]
                )
                candidate_paths.append((candidate_path, ratio))
            except Exception:
                continue

        if not candidate_paths:
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-sseof",
                    "-0.08",
                    "-i",
                    video_path,
                    "-update",
                    "1",
                    "-frames:v",
                    "1",
                    output_path,
                ]
            )
            return output_path

        best_candidate = max(
            candidate_paths,
            key=lambda item: self._score_continuity_frame(item[0], prefer_two_subjects=prefer_two_subjects, late_ratio=item[1]),
        )[0]
        shutil.copyfile(best_candidate, output_target)
        return output_path

    def _score_continuity_frame(self, image_path: Path, *, prefer_two_subjects: bool, late_ratio: float) -> float:
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                edges = rgb.filter(ImageFilter.FIND_EDGES).convert("L")
                thirds = [
                    edges.crop((0, 0, width // 3, height)),
                    edges.crop((width // 3, 0, (width * 2) // 3, height)),
                    edges.crop(((width * 2) // 3, 0, width, height)),
                ]
                left_score = ImageStat.Stat(thirds[0]).mean[0]
                center_score = ImageStat.Stat(thirds[1]).mean[0]
                right_score = ImageStat.Stat(thirds[2]).mean[0]
                overall_score = ImageStat.Stat(edges).mean[0]
                balance_score = min(left_score, right_score)
                if prefer_two_subjects:
                    return balance_score * 2.2 + center_score * 0.8 + overall_score * 0.4 + late_ratio * 4
                return max(left_score, center_score, right_score) * 1.4 + overall_score * 0.5 + late_ratio * 4
        except Exception:
            return late_ratio * 4

    def _turn_boundaries_from_silence(self, audio_path: str, expected_count: int, total_duration: float) -> list[float]:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_path,
                "-af",
                "silencedetect=noise=-30dB:d=0.14",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        silence_pairs: list[tuple[float, float, float]] = []
        current_start: float | None = None
        for line in completed.stderr.splitlines():
            start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
            if start_match:
                current_start = float(start_match.group(1))
                continue
            end_match = re.search(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", line)
            if end_match and current_start is not None:
                end_value = float(end_match.group(1))
                duration_value = float(end_match.group(2))
                if current_start > 0.18 and end_value < total_duration - 0.18:
                    silence_pairs.append((current_start, end_value, duration_value))
                current_start = None

        desired_breaks = max(0, expected_count - 1)
        selected: list[float] = []
        if silence_pairs:
            strongest = sorted(silence_pairs, key=lambda item: item[2], reverse=True)[:desired_breaks]
            selected = sorted((start + end) / 2 for start, end, _ in strongest)

        if len(selected) < desired_breaks:
            selected = [
                round(total_duration * step / expected_count, 3)
                for step in range(1, expected_count)
            ]

        normalized: list[float] = []
        last_point = 0.0
        for point in selected:
            bounded = max(last_point + 0.28, min(point, total_duration - 0.28))
            normalized.append(round(bounded, 3))
            last_point = bounded
        return normalized[:desired_breaks]

    def render_scene_audio_video(
        self,
        *,
        project_dir: str,
        scene_assets: list[dict],
        scene_audio_paths: list[str],
        output_path: str,
        quality: str,
    ) -> str:
        self.ensure_ffmpeg()
        if not scene_assets or not scene_audio_paths:
            raise ValueError("Scene video/audio assets are required for render.")
        if len(scene_assets) != len(scene_audio_paths):
            raise ValueError("Scene video/audio counts do not match.")

        project_path = Path(project_dir)
        working_path = project_path / "render_work"
        working_path.mkdir(exist_ok=True)

        video_segments: list[Path] = []
        audio_segments: list[Path] = []
        for index, (asset, audio_path) in enumerate(zip(scene_assets, scene_audio_paths, strict=True)):
            if asset.get("asset_type") != "video" or not asset.get("asset_path"):
                raise ValueError("Scene-audio rendering currently requires video assets for every scene.")
            segment_video = working_path / f"scene_mux_video_{index:02d}.mp4"
            segment_audio = working_path / f"scene_mux_audio_{index:02d}.mp3"
            target_duration = min(
                self.get_video_duration(asset["asset_path"]),
                self.get_audio_duration(audio_path),
            )
            target_duration = max(0.75, round(target_duration, 2))
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    asset["asset_path"],
                    "-t",
                    f"{target_duration:.2f}",
                    "-vf",
                    "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(segment_video),
                ]
            )
            self._run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-t",
                    f"{target_duration:.2f}",
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    "192k",
                    str(segment_audio),
                ]
            )
            video_segments.append(segment_video)
            audio_segments.append(segment_audio)

        video_concat = working_path / "scene_video_concat.txt"
        audio_concat = working_path / "scene_audio_concat.txt"
        video_concat.write_text("\n".join(f"file '{segment.as_posix()}'" for segment in video_segments), encoding="utf-8")
        audio_concat.write_text("\n".join(f"file '{segment.as_posix()}'" for segment in audio_segments), encoding="utf-8")

        stitched_video = working_path / "scene_stitched_video.mp4"
        stitched_audio = working_path / "scene_stitched_audio.mp3"
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_concat),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(stitched_video),
            ]
        )
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_concat),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(stitched_audio),
            ]
        )

        crf = {"Low": "28", "Medium": "23", "High": "18"}.get(quality, "18")
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stitched_video),
                "-i",
                str(stitched_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                crf,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]
        )
        return output_path

    def render_interleaved_scene_audio_video(
        self,
        *,
        project_dir: str,
        video_sequence: list[dict],
        quality: str,
        output_path: str,
        fps: int,
    ) -> str:
        self.ensure_ffmpeg()
        if not video_sequence:
            raise ValueError("Video sequence is required for interleaved render.")

        project_path = Path(project_dir)
        working_path = project_path / "render_work"
        working_path.mkdir(exist_ok=True)

        video_segments: list[Path] = []
        audio_segments: list[Path] = []

        for index, item in enumerate(video_sequence):
            asset_path = str(item.get("asset_path", "")).strip()
            asset_type = str(item.get("asset_type", "video")).strip()
            audio_path = str(item.get("audio_path", "")).strip()
            duration_hint = float(item.get("duration_hint", 1.6) or 1.6)
            segment_video = working_path / f"mix_video_{index:02d}.mp4"
            segment_audio = working_path / f"mix_audio_{index:02d}.mp3"

            if asset_type == "video":
                video_duration = self.get_video_duration(asset_path)
                if audio_path:
                    target_duration = min(video_duration, self.get_audio_duration(audio_path))
                else:
                    target_duration = min(video_duration, max(0.8, duration_hint))
                target_duration = max(0.75, round(target_duration, 2))
                self._run_ffmpeg(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        asset_path,
                        "-t",
                        f"{target_duration:.2f}",
                        "-vf",
                        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(segment_video),
                    ]
                )
            else:
                target_duration = max(0.9, round(duration_hint, 2))
                self._create_image_clip({"asset_path": asset_path}, target_duration, segment_video, fps, include_silent_audio=False)

            if audio_path:
                self._run_ffmpeg(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        audio_path,
                        "-t",
                        f"{target_duration:.2f}",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "192k",
                        str(segment_audio),
                    ]
                )
            else:
                self.create_silence(str(segment_audio), target_duration)

            video_segments.append(segment_video)
            audio_segments.append(segment_audio)

        video_concat = working_path / "mix_video_concat.txt"
        audio_concat = working_path / "mix_audio_concat.txt"
        video_concat.write_text("\n".join(f"file '{segment.as_posix()}'" for segment in video_segments), encoding="utf-8")
        audio_concat.write_text("\n".join(f"file '{segment.as_posix()}'" for segment in audio_segments), encoding="utf-8")

        stitched_video = working_path / "mix_stitched_video.mp4"
        stitched_audio = working_path / "mix_stitched_audio.mp3"
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_concat),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(stitched_video),
            ]
        )
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_concat),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(stitched_audio),
            ]
        )

        crf = {"Low": "28", "Medium": "23", "High": "18"}.get(quality, "18")
        self._run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stitched_video),
                "-i",
                str(stitched_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                crf,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-shortest",
                output_path,
            ]
        )
        return output_path

    def _build_durations(self, scene_assets: list[dict], voice_duration: float, voice_segment_durations: list[float]) -> list[float]:
        if not scene_assets:
            return []
        count = len(scene_assets)
        if voice_segment_durations:
            if len(voice_segment_durations) == count + 1:
                intro_duration = voice_segment_durations[0]
                body_durations = voice_segment_durations[1:]
                durations = [round(body_durations[0] + intro_duration, 2), *[round(value, 2) for value in body_durations[1:]]]
                durations = [max(1.45, duration) for duration in durations[:count]]
                difference = round(voice_duration - sum(durations), 2)
                if durations and abs(difference) >= 0.01:
                    durations[-1] = max(1.45, round(durations[-1] + difference, 2))
                return durations
            if len(voice_segment_durations) == count:
                durations = [max(1.45, round(value, 2)) for value in voice_segment_durations]
                difference = round(voice_duration - sum(durations), 2)
                if durations and abs(difference) >= 0.01:
                    durations[-1] = max(1.45, round(durations[-1] + difference, 2))
                return durations
        hooks = min(2, count)
        hints = [max(1.5, float(scene.get("duration_hint", 3.0))) for scene in scene_assets]
        shaped_hints: list[float] = []
        for index, hint in enumerate(hints):
            if index < hooks:
                shaped = min(hint, 2.1)
            elif index == count - 1:
                shaped = min(hint, 2.6)
            else:
                shaped = min(hint, 3.2)
            shaped_hints.append(max(1.45, shaped))

        min_total = 1.45 * count
        target_total = max(min_total, voice_duration)
        total_hint = sum(shaped_hints)
        ratio = target_total / total_hint if total_hint else 1.0
        durations = [max(1.45, round(hint * ratio, 2)) for hint in shaped_hints]

        difference = round(target_total - sum(durations), 2)
        if durations and abs(difference) >= 0.01:
            durations[-1] = max(1.45, round(durations[-1] + difference, 2))
        return durations

    def _create_image_clip(self, asset: dict, duration: float, output_path: Path, fps: int, include_silent_audio: bool = False) -> None:
        image_path = asset["asset_path"]
        zoom_frames = max(1, int(duration * fps))
        command = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            image_path,
        ]
        if include_silent_audio:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
        command += [
            "-t",
            f"{duration:.2f}",
            "-vf",
            (
                f"scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,"
                f"zoompan=z='min(zoom+0.0008,1.14)':d={zoom_frames}:x='iw/2-(iw/zoom/2)':"
                f"y='ih/2-(ih/zoom/2)',fps={fps},"
                f"eq=contrast=1.03:saturation=1.02,"
                f"fade=t=in:st=0:d=0.28,fade=t=out:st={max(0.0, duration - 0.32):.2f}:d=0.32"
            ),
        ]
        if include_silent_audio:
            command += ["-shortest", "-c:a", "aac", "-b:a", "128k"]
        else:
            command += ["-an"]
        command += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _create_video_clip(self, asset: dict, duration: float, output_path: Path, fps: int, keep_audio: bool = False) -> None:
        video_path = asset["asset_path"]
        source_duration = self.get_video_duration(video_path)
        usable_duration = min(duration, max(0.75, source_duration - 0.05))
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-t",
            f"{usable_duration:.2f}",
            "-vf",
            (
                f"scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,"
                f"fps={fps},"
                f"eq=contrast=1.04:saturation=0.98,"
                f"fade=t=in:st=0:d=0.18,fade=t=out:st={max(0.0, usable_duration - 0.20):.2f}:d=0.20"
            ),
        ]
        if keep_audio and self._has_audio_stream(video_path):
            command += ["-c:a", "aac", "-b:a", "192k"]
        else:
            command += ["-an"]
        command += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        self._run_ffmpeg(command)

    def _has_audio_stream(self, media_path: str) -> bool:
        self.ensure_ffmpeg()
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                media_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())

    def _run_ffmpeg(self, command: list[str]) -> None:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            details = {
                "command": command,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
            raise RuntimeError(f"FFmpeg command failed: {json.dumps(details, indent=2)}")
