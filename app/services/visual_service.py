from __future__ import annotations

import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import AppSettings
from app.models import SceneSpec


class VisualService:
    SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
    PEXELS_VIDEO_URL = "https://api.pexels.com/v1/videos/search"
    PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
    PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
    PIXABAY_IMAGE_URL = "https://pixabay.com/api/"
    DEFAULT_HEADERS = {
        "User-Agent": "AutomatedVideoCreator/1.0",
        "Accept": "application/json,text/plain,*/*",
    }

    TRUE_CRIME_PALETTES = [
        ("#0B0F14", "#661010", "#D8B15A", "#EAE6DC"),
        ("#121212", "#3D0C11", "#8B9EB7", "#F4EFE6"),
    ]
    FINANCE_PALETTES = [
        ("#081C15", "#1B4332", "#95D5B2", "#EDF6F9"),
        ("#0F172A", "#1D4ED8", "#60A5FA", "#EFF6FF"),
    ]
    VIRAL_PALETTES = [
        ("#172033", "#8B5CF6", "#F97316", "#FFF7ED"),
        ("#111827", "#F43F5E", "#22D3EE", "#F8FAFC"),
    ]
    CINEMATIC_QUERY_SUFFIX = {
        "True Crime Mode": "cinematic moody reenactment portrait dramatic shadows suspense b-roll",
        "Finance Tips Mode": "cinematic business portrait close-up modern office city dramatic b-roll",
        "General Viral Mode": "cinematic portrait emotional dramatic story-driven b-roll",
    }

    def build_scene_assets(self, project_dir: str, scenes: list[SceneSpec], settings: AppSettings) -> list[dict]:
        asset_dir = Path(project_dir) / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[dict] = []

        for index, scene in enumerate(scenes, start=1):
            query = self._scene_query(scene)
            provider = settings.video.visual_provider
            self._attach_scene_asset(asset_dir, index, scene, provider, query, settings)
            rendered.append(scene.to_dict())
        return rendered

    def build_fact_bridge_assets(self, project_dir: str, scenes: list[dict], settings: AppSettings) -> list[dict]:
        asset_dir = Path(project_dir) / "fact_bridges"
        asset_dir.mkdir(parents=True, exist_ok=True)
        bridges: list[dict] = []
        for index in range(max(0, len(scenes) - 1)):
            current_scene = scenes[index]
            next_scene = scenes[index + 1]
            bridge_scene = SceneSpec(
                headline=f"Bridge {index + 1}",
                supporting_text=str(next_scene.get("supporting_text") or next_scene.get("headline") or "").strip(),
                visual_keywords=list(next_scene.get("visual_keywords") or [])[:4],
                mood=str(next_scene.get("mood") or "animated"),
                narration_text="",
                action_prompt="",
                audio_dialogue_cue="",
                duration_hint=1.6,
                purpose="bridge",
                negative_prompt="no people speaking, no subtitles, no logos, no on-screen text",
                camera_style="quick insert shot, cinematic cutaway, no dialogue",
                style_notes="fast bridge clip between fact beats",
                visual_query=self._build_fact_bridge_query(current_scene, next_scene),
            )
            provider = "Stock Footage"
            self._attach_scene_asset(asset_dir, index + 1, bridge_scene, provider, bridge_scene.visual_query, settings)
            bridges.append(bridge_scene.to_dict())
        return bridges

    def test_pexels_connection(self, api_key: str) -> tuple[bool, str]:
        api_key = self._normalize_api_key(api_key)
        if not api_key:
            return False, "Missing API key"
        try:
            response = requests.get(
                self.PEXELS_PHOTO_URL,
                timeout=20,
                headers={**self.DEFAULT_HEADERS, "Authorization": api_key},
                params={"query": "city", "per_page": 1},
            )
            if response.ok:
                return True, "Connected"
            if response.status_code in {401, 403}:
                return False, "Invalid Key"
            return False, f"Error {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Error: {exc}"

    def test_serpapi_connection(self, api_key: str) -> tuple[bool, str]:
        api_key = self._normalize_api_key(api_key)
        if not api_key:
            return False, "Missing API key"
        try:
            response = requests.get(
                self.SERPAPI_SEARCH_URL,
                timeout=20,
                headers=self.DEFAULT_HEADERS,
                params={
                    "engine": "google_images",
                    "q": "test",
                    "api_key": api_key,
                    "ijn": 0,
                },
            )
            payload = response.json()
            if response.ok and isinstance(payload.get("images_results"), list):
                return True, "Connected"
            if payload.get("error"):
                return False, str(payload.get("error"))
            return False, f"Error {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Error: {exc}"

    def test_pixabay_connection(self, api_key: str) -> tuple[bool, str]:
        api_key = self._normalize_api_key(api_key)
        if not api_key:
            return False, "Missing API key"
        try:
            image_response = requests.get(
                self.PIXABAY_IMAGE_URL,
                timeout=20,
                headers=self.DEFAULT_HEADERS,
                params={"key": api_key, "q": "test", "per_page": 3},
            )
            image_ok, image_message = self._validate_pixabay_response(image_response, "images")
            if image_ok:
                return True, f"Connected ({image_message})"

            video_response = requests.get(
                self.PIXABAY_VIDEO_URL,
                timeout=20,
                headers=self.DEFAULT_HEADERS,
                params={"key": api_key, "q": "test", "per_page": 3},
            )
            video_ok, video_message = self._validate_pixabay_response(video_response, "videos")
            if video_ok:
                return True, f"Connected ({video_message})"

            return False, f"Images: {image_message} | Videos: {video_message}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Error: {exc}"

    def search_reference_images(self, query: str, settings: AppSettings, limit: int = 12) -> list[dict[str, str]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Search query is required.")
        results: list[dict[str, str]] = []
        if settings.serpapi_api_key:
            results.extend(self._search_serpapi_images(cleaned_query, settings.serpapi_api_key, limit=limit))
        if settings.pexels_api_key and len(results) < limit:
            results.extend(self._search_pexels_images(cleaned_query, settings.pexels_api_key, limit=limit))
        if settings.pixabay_api_key and len(results) < limit:
            results.extend(self._search_pixabay_images(cleaned_query, settings.pixabay_api_key, limit=max(1, limit - len(results))))
        deduped: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for item in results:
            asset_url = item.get("asset_url", "").strip()
            if not asset_url or asset_url in seen_urls:
                continue
            seen_urls.add(asset_url)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def import_reference_image(self, image_urls: list[str], filename_hint: str, output_dir: str) -> str:
        cleaned_urls = [str(url).strip() for url in image_urls if str(url).strip()]
        if not cleaned_urls:
            raise ValueError("At least one image URL is required.")
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename_hint or "reference.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        stem = Path(filename_hint or "reference").stem.strip() or "reference"
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem).strip("-") or "reference"
        target_path = target_dir / f"{safe_stem}{suffix}"
        counter = 2
        while target_path.exists():
            target_path = target_dir / f"{safe_stem}-{counter}{suffix}"
            counter += 1
        errors: list[str] = []
        for candidate_url in cleaned_urls:
            try:
                self._download_binary(candidate_url, target_path)
                return str(target_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate_url}: {exc}")
                if target_path.exists():
                    target_path.unlink(missing_ok=True)
                continue
        raise RuntimeError("Image download failed for all candidate URLs: " + " | ".join(errors[:3]))

    def _search_serpapi_images(self, query: str, api_key: str, limit: int = 12) -> list[dict[str, str]]:
        normalized_key = self._normalize_api_key(api_key)
        if not normalized_key:
            return []
        response = requests.get(
            self.SERPAPI_SEARCH_URL,
            timeout=30,
            headers=self.DEFAULT_HEADERS,
            params={
                "engine": "google_images",
                "q": query,
                "api_key": normalized_key,
                "ijn": 0,
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en",
                "safe": "off",
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(str(payload.get("error")))
        results: list[dict[str, str]] = []
        for index, item in enumerate(payload.get("images_results", []), start=1):
            thumbnail_url = str(item.get("thumbnail", "")).strip()
            asset_url = str(item.get("original", "")).strip() or thumbnail_url
            if not thumbnail_url or not asset_url:
                continue
            result_id = item.get("position", index)
            title = str(item.get("title", "")).strip()
            source = str(item.get("source", "")).strip()
            ext = Path(urlparse(asset_url).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            base_name = f"serpapi-{result_id}{ext}"
            results.append({
                "id": f"serpapi-{result_id}",
                "provider": "SerpAPI",
                "thumbnail_url": thumbnail_url,
                "asset_url": asset_url,
                "fallback_url": thumbnail_url,
                "page_url": str(item.get("link", "")).strip(),
                "credit": title if title else (source or "SerpAPI result"),
                "filename_hint": base_name,
            })
            if len(results) >= limit:
                break
        return results

    def _attach_scene_asset(
        self,
        asset_dir: Path,
        index: int,
        scene: SceneSpec,
        provider: str,
        query: str,
        settings: AppSettings,
    ) -> None:
        if provider in {"Stock Footage", "Hybrid"}:
            if self._try_stock_footage(asset_dir, index, scene, query, settings):
                if provider == "Hybrid":
                    self._apply_hybrid_grade(scene, settings.content_mode)
                return

        if provider in {"AI Images", "Hybrid"}:
            if self._try_ai_image_mode(asset_dir, index, scene, settings.content_mode):
                return

        self._render_local_fallback(scene, asset_dir / f"scene_{index:02d}.png", settings.content_mode)

    def _try_stock_footage(
        self,
        asset_dir: Path,
        index: int,
        scene: SceneSpec,
        query: str,
        settings: AppSettings,
    ) -> bool:
        if settings.pexels_api_key:
            asset = self._download_pexels_asset(asset_dir, index, scene, query, settings.pexels_api_key)
            if asset:
                return True
        if settings.pixabay_api_key:
            asset = self._download_pixabay_asset(asset_dir, index, scene, query, settings.pixabay_api_key)
            if asset:
                return True
        return False

    def _download_pexels_asset(
        self,
        asset_dir: Path,
        index: int,
        scene: SceneSpec,
        query: str,
        api_key: str,
    ) -> bool:
        api_key = self._normalize_api_key(api_key)
        stock_query = self._build_cinematic_query(scene)
        video_resp = requests.get(
            self.PEXELS_VIDEO_URL,
            timeout=30,
            headers={**self.DEFAULT_HEADERS, "Authorization": api_key},
            params={
                "query": stock_query,
                "per_page": 10,
                "orientation": "portrait",
                "size": "medium",
            },
        )
        if video_resp.ok:
            videos = video_resp.json().get("videos", [])
            chosen_video = self._pick_pexels_video(videos)
            if chosen_video:
                file_url = chosen_video["link"]
                output_path = asset_dir / f"scene_{index:02d}.mp4"
                self._download_binary(file_url, output_path)
                scene.asset_path = str(output_path)
                scene.asset_type = "video"
                scene.poster_path = ""
                scene.source_name = "Pexels"
                scene.source_url = chosen_video["page_url"]
                scene.source_credit = chosen_video["credit"]
                return True

        image_resp = requests.get(
            self.PEXELS_PHOTO_URL,
            timeout=30,
            headers={**self.DEFAULT_HEADERS, "Authorization": api_key},
            params={"query": stock_query, "per_page": 8, "orientation": "portrait"},
        )
        if image_resp.ok:
            photos = image_resp.json().get("photos", [])
            if photos:
                photo = photos[0]
                image_url = photo["src"].get("large2x") or photo["src"].get("large") or photo["src"].get("original")
                if image_url:
                    output_path = asset_dir / f"scene_{index:02d}.jpg"
                    self._download_binary(image_url, output_path)
                    scene.asset_path = str(output_path)
                    scene.asset_type = "image"
                    scene.poster_path = str(output_path)
                    scene.source_name = "Pexels"
                    scene.source_url = photo.get("url", "")
                    scene.source_credit = f"Photo by {photo.get('photographer', 'Pexels creator')} on Pexels"
                    return True
        return False

    def _search_pexels_images(self, query: str, api_key: str, limit: int = 12) -> list[dict[str, str]]:
        normalized_key = self._normalize_api_key(api_key)
        if not normalized_key:
            return []
        response = requests.get(
            self.PEXELS_PHOTO_URL,
            timeout=30,
            headers={**self.DEFAULT_HEADERS, "Authorization": normalized_key},
            params={"query": query, "per_page": max(1, min(limit, 20)), "orientation": "portrait"},
        )
        response.raise_for_status()
        results: list[dict[str, str]] = []
        for index, photo in enumerate(response.json().get("photos", []), start=1):
            src = photo.get("src", {}) or {}
            thumbnail_url = src.get("medium") or src.get("small") or src.get("large")
            asset_url = src.get("large2x") or src.get("large") or src.get("original") or thumbnail_url
            if not thumbnail_url or not asset_url:
                continue
            photographer = str(photo.get("photographer", "Pexels creator")).strip()
            results.append({
                "id": f"pexels-{photo.get('id', index)}",
                "provider": "Pexels",
                "thumbnail_url": str(thumbnail_url),
                "asset_url": str(asset_url),
                "page_url": str(photo.get("url", "")),
                "credit": f"Photo by {photographer} on Pexels",
                "filename_hint": f"pexels-{photo.get('id', index)}.jpg",
            })
        return results

    def _download_pixabay_asset(
        self,
        asset_dir: Path,
        index: int,
        scene: SceneSpec,
        query: str,
        api_key: str,
    ) -> bool:
        api_key = self._normalize_api_key(api_key)
        stock_query = self._build_cinematic_query(scene)
        video_resp = requests.get(
            self.PIXABAY_VIDEO_URL,
            timeout=30,
            headers=self.DEFAULT_HEADERS,
            params={
                "key": api_key,
                "q": stock_query,
                "per_page": 10,
                "orientation": "vertical",
                "video_type": "all",
            },
        )
        if video_resp.ok:
            hits = video_resp.json().get("hits", [])
            if hits:
                video = self._pick_pixabay_video(hits[0])
                if video:
                    output_path = asset_dir / f"scene_{index:02d}.mp4"
                    self._download_binary(video["url"], output_path)
                    scene.asset_path = str(output_path)
                    scene.asset_type = "video"
                    scene.poster_path = video.get("thumbnail", "")
                    scene.source_name = "Pixabay"
                    scene.source_url = hits[0].get("pageURL", "")
                    scene.source_credit = f"Video by {hits[0].get('user', 'Pixabay creator')} on Pixabay"
                    return True

        image_resp = requests.get(
            self.PIXABAY_IMAGE_URL,
            timeout=30,
            headers=self.DEFAULT_HEADERS,
            params={
                "key": api_key,
                "q": stock_query,
                "per_page": 8,
                "orientation": "vertical",
                "image_type": "photo",
            },
        )
        if image_resp.ok:
            hits = image_resp.json().get("hits", [])
            if hits:
                hit = hits[0]
                image_url = hit.get("largeImageURL") or hit.get("webformatURL")
                if image_url:
                    output_path = asset_dir / f"scene_{index:02d}.jpg"
                    self._download_binary(image_url, output_path)
                    scene.asset_path = str(output_path)
                    scene.asset_type = "image"
                    scene.poster_path = str(output_path)
                    scene.source_name = "Pixabay"
                    scene.source_url = hit.get("pageURL", "")
                    scene.source_credit = f"Image by {hit.get('user', 'Pixabay creator')} on Pixabay"
                    return True
        return False

    def _search_pixabay_images(self, query: str, api_key: str, limit: int = 12) -> list[dict[str, str]]:
        normalized_key = self._normalize_api_key(api_key)
        if not normalized_key:
            return []
        response = requests.get(
            self.PIXABAY_IMAGE_URL,
            timeout=30,
            headers=self.DEFAULT_HEADERS,
            params={
                "key": normalized_key,
                "q": query,
                "per_page": max(1, min(limit, 20)),
                "orientation": "vertical",
                "image_type": "photo",
            },
        )
        response.raise_for_status()
        results: list[dict[str, str]] = []
        for index, hit in enumerate(response.json().get("hits", []), start=1):
            thumbnail_url = hit.get("webformatURL") or hit.get("previewURL")
            asset_url = hit.get("largeImageURL") or thumbnail_url
            if not thumbnail_url or not asset_url:
                continue
            creator = str(hit.get("user", "Pixabay creator")).strip()
            image_id = hit.get("id", index)
            results.append({
                "id": f"pixabay-{image_id}",
                "provider": "Pixabay",
                "thumbnail_url": str(thumbnail_url),
                "asset_url": str(asset_url),
                "page_url": str(hit.get("pageURL", "")),
                "credit": f"Image by {creator} on Pixabay",
                "filename_hint": f"pixabay-{image_id}.jpg",
            })
        return results

    def _validate_pixabay_response(self, response: requests.Response, endpoint_name: str) -> tuple[bool, str]:
        try:
            payload = response.json()
        except Exception:
            snippet = (response.text or "").strip().replace("\r", " ").replace("\n", " ")
            snippet = snippet[:180] if snippet else "no response body"
            return False, f"{endpoint_name} returned non-JSON response (HTTP {response.status_code}): {snippet}"

        if response.status_code == 200:
            if any(field in payload for field in ("total", "totalHits", "hits")):
                total_hits = payload.get("totalHits", payload.get("total", "ok"))
                return True, f"{endpoint_name} endpoint OK, totalHits={total_hits}"
            return False, f"{endpoint_name} HTTP 200 but missing expected fields"

        if isinstance(payload, dict):
            for key in ("error", "message", "errors"):
                if key in payload and payload[key]:
                    return False, f"{endpoint_name}: {payload[key]}"
        return False, f"{endpoint_name} HTTP {response.status_code}"

    def _normalize_api_key(self, value: str) -> str:
        value = (value or "").strip().strip("\"' ")
        if not value:
            return ""
        if "pixabay.com/api" in value or "key=" in value:
            parsed = urlparse(value if "://" in value else f"https://dummy.local/?{value.lstrip('?')}")
            params = parse_qs(parsed.query)
            extracted = params.get("key", [""])[0].strip()
            if extracted:
                return extracted
        if value.startswith("key="):
            return value[4:].strip()
        return value

    def _try_ai_image_mode(self, asset_dir: Path, index: int, scene: SceneSpec, content_mode: str) -> bool:
        output_path = asset_dir / f"scene_{index:02d}.png"
        self._render_local_fallback(scene, output_path, content_mode, ai_style_hint=True)
        return True

    def _apply_hybrid_grade(self, scene: SceneSpec, content_mode: str) -> None:
        if scene.asset_type != "image" or not scene.asset_path:
            return
        path = Path(scene.asset_path)
        if not path.exists():
            return
        image = Image.open(path).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.12 if content_mode == "True Crime Mode" else 1.08)
        image = ImageEnhance.Color(image).enhance(0.82 if content_mode == "True Crime Mode" else 0.96)
        image = ImageEnhance.Sharpness(image).enhance(1.08)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if content_mode == "True Crime Mode":
            draw.rectangle((0, 0, image.size[0], image.size[1]), fill=(8, 4, 6, 54))
            draw.ellipse((760, -120, 1260, 340), fill=(118, 32, 24, 52))
        elif content_mode == "Finance Tips Mode":
            draw.rectangle((0, 0, image.size[0], image.size[1]), fill=(6, 16, 30, 32))
            draw.rectangle((0, 1460, image.size[0], image.size[1]), fill=(10, 18, 28, 72))
        else:
            draw.rectangle((0, 0, image.size[0], image.size[1]), fill=(10, 10, 18, 34))
        combined = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        combined.save(path, quality=95)
        scene.poster_path = str(path)

    def _render_local_fallback(
        self,
        scene: SceneSpec,
        output_path: Path,
        content_mode: str,
        ai_style_hint: bool = False,
    ) -> None:
        width, height = 1080, 1920
        palettes = self._palette_for_mode(content_mode)
        background, accent, highlight, _ = random.choice(palettes)
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)

        self._draw_background_motif(draw, width, height, background, accent, highlight, content_mode)
        if ai_style_hint:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
            draw = ImageDraw.Draw(image)

        if content_mode == "True Crime Mode":
            self._draw_true_crime_overlay(draw, width, height, highlight)
        elif content_mode == "Finance Tips Mode":
            self._draw_finance_overlay(draw, width, height, highlight)
        else:
            self._draw_general_story_overlay(draw, width, height, accent, highlight)

        image = image.filter(ImageFilter.GaussianBlur(radius=0.2))
        image.save(output_path)
        scene.asset_path = str(output_path)
        scene.poster_path = str(output_path)
        scene.asset_type = "image"
        scene.source_name = "Local Fallback"
        scene.source_credit = "Generated cinematic fallback"

    def generate_video_poster(self, video_path: str, output_path: str) -> str:
        if not shutil.which("ffmpeg"):
            return ""
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                "thumbnail,scale=540:-1",
                "-frames:v",
                "1",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return ""
        return output_path

    def _pick_pexels_video(self, videos: list[dict[str, Any]]) -> dict[str, str] | None:
        best_choice: dict[str, str] | None = None
        best_score = -1
        for video in videos:
            files = video.get("video_files", [])
            portrait_files = [item for item in files if item.get("height", 0) >= item.get("width", 0)]
            candidates = portrait_files or files
            if not candidates:
                continue
            chosen = sorted(candidates, key=lambda item: item.get("width", 0), reverse=True)[0]
            score = self._score_motion_asset(
                width=int(chosen.get("width", 0) or 0),
                height=int(chosen.get("height", 0) or 0),
                duration=float(video.get("duration", 0) or 0),
            )
            if score <= best_score:
                continue
            best_score = score
            best_choice = {
                "link": chosen.get("link", ""),
                "page_url": video.get("url", ""),
                "credit": f"Video by {video.get('user', {}).get('name', 'Pexels creator')} on Pexels",
            }
        return best_choice

    def _pick_pixabay_video(self, hit: dict[str, Any]) -> dict[str, str] | None:
        videos = hit.get("videos", {})
        for name in ("medium", "small", "tiny"):
            item = videos.get(name)
            if item and item.get("url"):
                return {"url": item["url"], "thumbnail": item.get("thumbnail", "")}
        return None

    def _download_binary(self, url: str, output_path: Path) -> None:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)

    def _scene_query(self, scene: SceneSpec) -> str:
        if scene.visual_query:
            return scene.visual_query
        tokens = [scene.headline, *scene.visual_keywords[:3], scene.mood]
        return " ".join(token for token in tokens if token).strip()

    def _build_cinematic_query(self, scene: SceneSpec) -> str:
        mode_hint = self.CINEMATIC_QUERY_SUFFIX.get(
            self._mode_from_mood(scene.mood),
            self.CINEMATIC_QUERY_SUFFIX["General Viral Mode"],
        )
        base = scene.visual_query or self._scene_query(scene)
        keywords = " ".join(scene.visual_keywords[:4])
        return " ".join(part for part in f"{base} {keywords} {mode_hint}".split() if part)

    def _build_fact_bridge_query(self, current_scene: dict, next_scene: dict) -> str:
        next_headline = str(next_scene.get("headline") or "").strip()
        next_keywords = " ".join(str(item).strip() for item in list(next_scene.get("visual_keywords") or [])[:4] if str(item).strip())
        next_visual = str(next_scene.get("visual_query") or next_scene.get("supporting_text") or "").strip()
        parts = [
            next_headline,
            next_keywords,
            next_visual,
            "animated illustration close-up detail cinematic b-roll vertical",
        ]
        return " ".join(part for part in parts if part).strip()

    def _palette_for_mode(self, content_mode: str) -> list[tuple[str, str, str, str]]:
        if content_mode == "True Crime Mode":
            return self.TRUE_CRIME_PALETTES
        if content_mode == "Finance Tips Mode":
            return self.FINANCE_PALETTES
        return self.VIRAL_PALETTES

    def _draw_background_motif(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        background: str,
        accent: str,
        highlight: str,
        content_mode: str,
    ) -> None:
        accent_rgb = self._hex_to_rgb(accent)
        highlight_rgb = self._hex_to_rgb(highlight)
        for step in range(height):
            mix = step / height
            color = (
                int((1 - mix) * self._hex_to_rgb(background)[0] + mix * accent_rgb[0] * 0.45),
                int((1 - mix) * self._hex_to_rgb(background)[1] + mix * accent_rgb[1] * 0.45),
                int((1 - mix) * self._hex_to_rgb(background)[2] + mix * accent_rgb[2] * 0.45),
            )
            draw.line((0, step, width, step), fill=color)

        if content_mode == "True Crime Mode":
            for idx in range(9):
                y = 180 + idx * 180
                draw.line((0, y, width, y + 120), fill=accent, width=14)
            for _ in range(12):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(80, 190)
                draw.ellipse((x, y, x + radius, y + radius), outline=highlight, width=4)
        elif content_mode == "Finance Tips Mode":
            base_y = height - 420
            for idx in range(8):
                left = 90 + idx * 110
                top = base_y - random.randint(120, 380)
                draw.rounded_rectangle((left, top, left + 66, base_y), radius=12, fill=accent)
            points = [(72, base_y - 180), (260, base_y - 270), (460, base_y - 210), (700, base_y - 360), (960, base_y - 460)]
            draw.line(points, fill=highlight, width=8, joint="curve")
            for x, y in points:
                draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=highlight)
        else:
            for angle in range(0, 360, 24):
                x = int(width * 0.8 + math.cos(math.radians(angle)) * 220)
                y = int(height * 0.18 + math.sin(math.radians(angle)) * 220)
                draw.line((width * 0.8, height * 0.18, x, y), fill=highlight, width=4)

    def _draw_true_crime_overlay(self, draw: ImageDraw.ImageDraw, width: int, height: int, highlight: str) -> None:
        for idx in range(3):
            inset = 88 + idx * 18
            draw.rounded_rectangle((inset, inset + 30, width - inset, height - inset - 40), radius=36, outline=highlight, width=2)
        draw.line((120, 260, 940, 260), fill=highlight, width=3)
        draw.line((220, 1180, 860, 980), fill=highlight, width=4)
        draw.line((740, 540, 400, 1460), fill=highlight, width=2)

    def _draw_finance_overlay(self, draw: ImageDraw.ImageDraw, width: int, height: int, highlight: str) -> None:
        icon_font = self._load_font(28, bold=True)
        for idx in range(4):
            x = 96 + idx * 200
            top = 1380 + idx * 20
            draw.rounded_rectangle((x, top, x + 160, top + 100), radius=20, outline=highlight, width=3)
            draw.line((x + 20, top + 72, x + 60, top + 48), fill=highlight, width=3)
            draw.line((x + 60, top + 48, x + 104, top + 58), fill=highlight, width=3)
            draw.line((x + 104, top + 58, x + 136, top + 24), fill=highlight, width=3)
            draw.text((x + 18, top + 14), f"+{idx + 2}.{idx}%", font=icon_font, fill=highlight)

    def _draw_general_story_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        accent: str,
        highlight: str,
    ) -> None:
        for radius in (180, 280, 380):
            draw.ellipse(
                (width - radius - 140, 120 - radius // 3, width + radius // 2, 120 + radius),
                outline=highlight if radius == 180 else accent,
                width=3 if radius == 180 else 2,
            )
        draw.line((120, 1520, 940, 1280), fill=highlight, width=3)
        draw.line((240, 1680, 860, 1100), fill=accent, width=2)

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _hex_to_rgb(self, hex_value: str) -> tuple[int, int, int]:
        hex_value = hex_value.lstrip("#")
        return tuple(int(hex_value[index : index + 2], 16) for index in (0, 2, 4))

    def _mode_from_mood(self, mood: str) -> str:
        lowered = (mood or "").lower()
        if lowered in {"suspenseful", "ominous", "dramatic"}:
            return "True Crime Mode"
        if lowered in {"confident", "smart", "practical"}:
            return "Finance Tips Mode"
        return "General Viral Mode"

    def _score_motion_asset(self, width: int, height: int, duration: float) -> int:
        score = 0
        if height >= width:
            score += 5
        if height >= 1920:
            score += 4
        elif height >= 1280:
            score += 2
        if 4 <= duration <= 18:
            score += 4
        elif 2 <= duration <= 30:
            score += 2
        score += min(4, width // 540)
        return score
