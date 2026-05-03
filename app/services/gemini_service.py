from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from app.models import ReuseEstimate, SceneSpec, ScriptPackage


class GeminiService:
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    DEFAULT_MODEL = "gemini-2.5-flash"
    FLASH_LITE_FALLBACK = "gemini-2.5-flash-lite"
    PRO_FALLBACK = FLASH_LITE_FALLBACK

    MODE_GUIDANCE = {
        "True Crime Mode": "Suspenseful, dramatic, documentary style, emotionally engaging, curiosity-driven.",
        "Finance Tips Mode": "Clear, practical, trustworthy, punchy, smart, viral short-form educator tone.",
        "Facts / Listicle Mode": "Fast, punchy, surprising, presenter-friendly, curiosity-driven viral facts delivery.",
        "General Viral Mode": "Broadly entertaining, high-retention, high-energy, socially shareable.",
    }

    def _video_prompt_word_guidance(self, video_length: str) -> str:
        try:
            seconds = max(1, int(float(str(video_length).strip() or "15")))
        except Exception:
            seconds = 15
        if seconds >= 30:
            return "For a 30 second WAN clip, write substantially fuller prompt dialogue and action beats, aiming for roughly 60 to 95 words of spoken dialogue per prompt."
        if seconds >= 20:
            return "For a 20 second WAN clip, aim for roughly 40 to 65 words of spoken dialogue per prompt."
        if seconds >= 15:
            return "For a 15 second WAN clip, aim for roughly 25 to 45 words of spoken dialogue per prompt."
        if seconds >= 10:
            return "For a 10 second WAN clip, aim for roughly 16 to 30 words of spoken dialogue per prompt."
        return "For a very short clip, keep the dialogue tight at roughly 8 to 18 words per prompt."

    def _wan_constraint_text(self, character_count: str) -> str:
        if character_count == "1":
            return "[TECHNICAL CONSTRAINTS: one speaker at a time, the main character fully visible throughout, no new characters introduced, no camera cuts or scene changes]"
        if character_count == "2":
            return "[TECHNICAL CONSTRAINTS: one speaker at a time, clear pause between speakers, both characters fully visible throughout, no new characters introduced, no camera cuts or scene changes]"
        return "[TECHNICAL CONSTRAINTS: one speaker at a time, the lead character remains clearly readable throughout, supporting characters remain visually consistent, no new main characters introduced, no camera cuts or scene changes]"

    def _normalize_rewritten_wan_prompt(self, prompt: str, character_mode: str) -> str:
        normalized = prompt.strip()
        if character_mode == "Two Character Conversation":
            normalized = re.sub(r"\bCharacter A\b", "the right-side character", normalized)
            normalized = re.sub(r"\bCharacter B\b", "the left-side character", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized.strip()

    def generate_script(
        self,
        api_key: str,
        model: str,
        content_mode: str,
        length_target: int,
        source_text: str,
        planning_mode: str = "Auto",
        visual_style: str = "Sketchbook Storytelling",
        character_mode: str = "Auto",
    ) -> ScriptPackage:
        if not api_key:
            raise ValueError("Gemini API key is required.")

        prompt = self._build_prompt(content_mode, length_target, source_text, planning_mode, visual_style, character_mode)
        payload = None
        last_error: Exception | None = None
        candidate_models = self._candidate_models(model)
        for candidate_model in candidate_models:
            try:
                payload = self._generate(
                    api_key,
                    candidate_model,
                    prompt,
                    self._max_output_tokens(length_target),
                    response_mime_type="text/plain",
                )
                content_text = self._extract_text(payload)
                script_data = self._parse_script_data(content_text, api_key, candidate_models)
                script_data = self._normalize_script_data(script_data, content_mode, length_target, planning_mode, visual_style)
                if self._is_complete_script_package(script_data, length_target):
                    return self._to_script_package(script_data)
                retry_prompt = self._build_retry_prompt(content_mode, length_target, source_text, script_data, planning_mode, visual_style, character_mode)
                retry_payload = self._generate(
                    api_key,
                    candidate_model,
                    retry_prompt,
                    self._max_output_tokens(length_target),
                    response_mime_type="text/plain",
                )
                retry_text = self._extract_text(retry_payload)
                retry_data = self._parse_script_data(retry_text, api_key, candidate_models)
                retry_data = self._normalize_script_data(retry_data, content_mode, length_target, planning_mode, visual_style)
                if self._is_complete_script_package(retry_data, length_target):
                    return self._to_script_package(retry_data)
                raise ValueError("Gemini returned an incomplete script package.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        if payload is None:
            raise RuntimeError(f"Gemini script generation failed across model fallbacks: {last_error}")
        raise RuntimeError(f"Gemini script generation failed across model fallbacks: {last_error}")

    def test_connection(self, api_key: str, model: str | None = None) -> tuple[bool, str]:
        if not api_key:
            return False, "Missing API key"
        last_message = "Connection failed"
        for candidate_model in self._candidate_models(model or self.DEFAULT_MODEL):
            try:
                payload = self._generate(
                    api_key,
                    candidate_model,
                    "Reply with exactly OK",
                    16,
                    response_mime_type="text/plain",
                )
                text = self._extract_text(payload, allow_non_text_success=True)
                if text:
                    return True, f"Connected ({candidate_model})"
            except Exception as exc:  # noqa: BLE001
                last_message = str(exc)
        if "API key" in last_message or "permission" in last_message.lower() or "unauthorized" in last_message.lower():
            return False, "Invalid Key"
        return False, last_message

    def generate_flux_storyboard_prompts(
        self,
        api_key: str,
        model: str,
        concept: str,
        prompt_count: int = 4,
        character_mode: str = "Solo",
    ) -> list[str]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        multi_character_guidance = (
            "The reference image contains TWO important characters. Every prompt must preserve both characters from the reference image, keep their identities distinct, keep their left/right relationship readable when possible, and never replace either with a new person."
            if character_mode == "Two Character Conversation"
            else "Treat the reference image as a single-character identity anchor unless the concept explicitly requires another person."
        )
        prompt = f"""
You are writing storyboard image prompts for Flux image generation.

The app will supply a single character reference image directly to Flux, so your job is NOT to redefine the character with conflicting visual details.
Instead, write prompts that preserve the SAME reference character while changing only the setting, action, and shot composition.

Return JSON only in this exact shape:
{{
  "prompts": [
    "prompt 1",
    "prompt 2",
    "prompt 3",
    "prompt 4",
    "prompt 5"
  ]
}}

Rules:
- Return exactly {prompt_count} prompts.
- Each prompt must be a single line of text.
- Every prompt must describe the SAME character from the reference image.
- Use wording like "the same reference character" or "the same character from the reference image".
- {multi_character_guidance}
- Do not invent extra characters beyond what the concept and reference image require.
- Keep the character fully visible, centered, readable, and compositionally clean.
- Favor expressive 3D animated feature-film style, cinematic lighting, soft shadows, detailed textures, clean composition, sharp focus.
- Keep prompts very descriptive and production-ready.
- These prompts are for 4 WAN videos, with 1 strong scene anchor image per video.
- Write 4 prompts only, one for each WAN clip.
- Keep the same character identity across all 4 images.
- Make the 4 scenes feel like one coherent short with smooth progression in mood and story world.
- Do not jump wildly between unrelated environments.
- Scene changes should feel connected and believable as parts of the same short narrative.
- Keep prompts vertical-friendly and suitable for WAN clips that start from one strong anchor frame.
- Do not use markdown fences.

Concept:
{concept[:4000]}
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key,
                    candidate_model,
                    prompt,
                    2200,
                    response_mime_type="application/json",
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                prompts = [str(item).strip() for item in data.get("prompts", []) if str(item).strip()]
                if len(prompts) == prompt_count:
                    return prompts
                raise ValueError(f"Expected {prompt_count} storyboard prompts, got {len(prompts)}.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Gemini Flux storyboard prompt generation failed across model fallbacks: {last_error}")

    def generate_flux_image_prompts(
        self,
        api_key: str,
        model: str,
        prompt_ideas: list[str],
        reference_image_path: str,
        character_mode: str = "Solo",
    ) -> list[str]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        prompt_count = len(prompt_ideas)
        if prompt_count < 1:
            raise ValueError("At least one prompt idea is required.")
        multi_character_guidance = (
            "The reference image contains two important characters. Preserve both characters, keep their identities distinct, and keep their relationship readable."
            if character_mode == "Two Character Conversation"
            else "Treat the reference image as a single-character identity anchor unless the user's prompt idea clearly requires more."
        )
        numbered_ideas = "\n".join(f"{index + 1}. {idea}" for index, idea in enumerate(prompt_ideas))
        prompt = f"""
You are writing high-quality Flux image prompts.

The app will supply a single reference image directly to Flux. Your job is to turn the user's rough prompt ideas into stronger production-ready image prompts while preserving the same reference character identity.

Return JSON only in exactly this shape:
{{
  "prompts": [
    "prompt 1",
    "prompt 2"
  ]
}}

Rules:
- Return exactly {prompt_count} prompts.
- Each prompt must be a single line of text.
- Keep the same reference character identity across all prompts.
- {multi_character_guidance}
- Treat the reference image as the source of truth.
- Write prompts as if you are editing or styling the reference image, not replacing it with a different subject.
- Preserve the same face, body, species, age vibe, proportions, and overall identity from the reference image.
- Do not use franchise names, character names, or external identity labels like "Phineas", "Peppa Pig", or "Mario" unless the user explicitly wrote that exact name in the prompt idea.
- Refer to subjects using reference-image language such as "the character on the left", "the taller character", "the same boy from the reference image", or "the same pig character from the reference image".
- When position is visually clear, anchor the subject by position like left side, right side, center, foreground, or background.
- Do not invent unrelated characters, a new main subject, or a different world unless the user's idea explicitly asks for a stylized variation of the same scene.
- Prefer changes like clothing, accessories, props, pose, expression, camera angle, lighting, background styling, and art style treatment layered onto the same reference image.
- If a prompt idea is blank or very short, invent a stronger funny visual variation yourself while still preserving the same reference subject.
- Lean into bold comedic upgrades when they fit: gold chains, rings, flashy jackets, sunglasses, accessories, lowriders, vans, scooters, custom cars, bouncing hydraulics, loud props, cheeky signs of swagger, or other over-the-top but funny details.
- Put the reference character in or around vehicles, interiors, or exaggerated lifestyle props when that makes the image funnier and more memorable.
- If the user asks for clothes or styling, describe it like an edit to the existing character in the photo.
- Do not describe a full character from scratch when the reference image already provides that information.
- Keep the subject fully visible, readable, and compositionally clean.
- Favor expressive 3D animated feature-film style, cinematic lighting, soft shadows, detailed textures, clean composition, and sharp focus.
- Make each prompt richer, clearer, and more visual than the user's rough idea.
- Keep the prompts suitable for direct Flux image generation.
- Do not use markdown fences.

User prompt ideas:
{numbered_ideas}
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=2200,
                    response_mime_type="application/json",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                prompts = [str(item).strip() for item in data.get("prompts", []) if str(item).strip()]
                if len(prompts) == prompt_count:
                    return prompts
                raise ValueError(f"Expected {prompt_count} Flux image prompts, got {len(prompts)}.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Gemini Flux image prompt generation failed across model fallbacks: {last_error}")

    def generate_wan_production_pack(
        self,
        api_key: str,
        model: str,
        num_videos: int,
        video_length: str,
        context_vibe: str,
        character_a: str,
        character_b: str,
        reference_image_path: str = "",
    ) -> dict[str, Any]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        prompt = f"""

You are a professional AI Cinematographer and Film Director. Your task is to look at the provided Reference Image and the user's Context/Vibe to generate short video script prompts for WAN2GP.

Your task: Generate a PRODUCTION PACK for a {num_videos}-video series. Each clip is approximately {video_length} seconds long.

Context and vibe: {context_vibe}

Character A (RIGHT Side): {character_a}
Character B (LEFT Side): {character_b}

CRITICAL RULES:
1. Visual Extraction: You MUST extract the setting, lighting, and character identities directly from the attached image. Do not invent new locations.
2. Dialogue & Persona: The dialogue must strictly reflect the provided Context/Vibe (e.g. if 'Vulgar UK Humour' is specified, use British slang, raw language, and reflect the power dynamics shown in the photo).
3. Prop Injection: If the dialogue or context mentions a physical object (weapon, drink, phone, etc.), you MUST explicitly describe the character physically holding or interacting with that object in the prompt directions.
4. Only ONE character speaks at a time. Clear pause before the other responds.
5. Both characters must remain fully visible in every shot. No new characters.
6. ONE single continuous line of text per prompt.
7. For a {video_length}s clip aim for 25–40 words of dialogue.
8. Every single video prompt MUST end with the bracketed technical string: [TECHNICAL CONSTRAINTS: one speaker at a time, clear pause between speakers, both characters fully visible throughout, no new characters introduced, no camera cuts or scene changes].

OUTPUT FORMAT:
Code block 1 — open with: ```VIDEO PROMPTS
Contain exactly {num_videos} prompts, each on its own numbered line.
Format: [Physical scene setup matching image] [Character RIGHT ({character_a}): emotion, then speaks: "dialogue"] [brief pause] [Character LEFT ({character_b}): reaction, then replies: "dialogue"] [TECHNICAL CONSTRAINTS: one speaker at a time, clear pause between speakers, both characters fully visible throughout, no new characters introduced, no camera cuts or scene changes]

Code block 2 — open with: ```STORYBOARD CONCEPT
Write exactly 200 words as a master visual style guide for Flux.
- Extract the exact environment, lighting, framing, and mood from the image.
- Match the vibe: {context_vibe}
Do not write any text outside the two code blocks.""".strip()

        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=3200,
                    response_mime_type="text/plain",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                return {"raw_output": text}
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"WAN production pack generation failed: {last_error}")

    def generate_character_descriptions(
        self,
        api_key: str,
        model: str,
        character_count: str,
        reference_image_path: str,
    ) -> dict[str, str]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        if not reference_image_path:
            raise ValueError("Reference image path is required.")
        if not Path(reference_image_path).is_file():
            raise ValueError("Reference image was not found.")
        if character_count == "1":
            schema_text = """
{
  "character_a": "One concise but visually useful description of the main character.",
  "character_b": ""
}
""".strip()
            rules_text = "- Describe the single main character only.\n- Leave character_b empty."
        elif character_count == "2":
            schema_text = """
{
  "character_a": "Description of the right-side or first important character.",
  "character_b": "Description of the left-side or second important character."
}
""".strip()
            rules_text = "- Describe two distinct characters.\n- If left/right is clear from the image, preserve that in the descriptions."
        else:
            schema_text = """
{
  "character_a": "Description of the main character or lead figure.",
  "character_b": "Short description of the other visible characters, group members, or supporting cast."
}
""".strip()
            rules_text = "- Focus character_a on the most important person.\n- Use character_b for the rest of the cast or group."
        prompt = f"""
Look at the attached image and extract practical character descriptions for a video prompting UI.

Return JSON only in exactly this shape:
{schema_text}

Rules:
{rules_text}
- Be concise but specific.
- Describe visible clothing, age vibe, expression, hairstyle, posture, and any important props.
- Do not invent names unless the image clearly contains a known character.
- Do not add markdown fences.
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=500,
                    response_mime_type="application/json",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                return {
                    "character_a": str(data.get("character_a", "")).strip(),
                    "character_b": str(data.get("character_b", "")).strip(),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Character description generation failed: {last_error}")

    def generate_context_vibe(
        self,
        api_key: str,
        model: str,
        character_count: str,
        character_a: str,
        character_b: str,
        reference_image_path: str = "",
    ) -> str:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        if not reference_image_path or not Path(reference_image_path).is_file():
            raise ValueError("A valid reference image is required.")
        prompt = f"""
You are writing a single strong Context / Vibe field for an adult comedy AI video workflow.

Return plain text only. Do not use markdown fences, headings, or bullet points.

Goal:
- Write one vivid paragraph that gives the app a funny, dark, rude, UK-slang-heavy scenario.
- The tone should feel like dark humour, vulgar comedy, swearing, cheeky insults, criminal swagger, chaotic low-life energy, and adult 18+ language.
- If the image supports it, you can lean into ideas like drug dealer swagger, pimp energy, chasing women, loud flexing, dodgy schemes, lowriders, bouncing hydraulics, chains, rings, fur coats, tacky luxury, or stupid criminal confidence.
- If something is not visible or plausible from the image, do not force it. Use the image as your anchor.
- Make it specific and scenario-based, not just a list of adjectives.
- Keep it funny, rude, and cinematic.

Character count: {character_count}
Character A: {character_a}
Character B: {character_b}

Rules:
- Mention what the characters are like, what ridiculous situation they are in, and what style of rude dialogue they should use.
- Use words like UK slang, vulgar, swearing, dark humour, rude comedy, criminal swagger, or pimp energy naturally when useful.
- If a vehicle or flashy prop is visible in the reference image, work it into the vibe.
- Keep it to roughly 80 to 150 words.
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=700,
                    response_mime_type="text/plain",
                    image_path=reference_image_path,
                )
                text = self._strip_code_fences(self._extract_text(payload)).strip()
                if text:
                    return " ".join(text.split())
                raise ValueError("Gemini returned an empty context/vibe.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Context/vibe generation failed: {last_error}")

    def generate_wan_production_pack_structured(
        self,
        api_key: str,
        model: str,
        num_videos: int,
        video_length: str,
        character_count: str,
        context_vibe: str,
        character_a: str,
        character_b: str,
        reference_image_path: str = "",
    ) -> dict[str, Any]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        prompt_count = max(1, int(num_videos))
        constraint_text = self._wan_constraint_text(character_count)
        word_guidance = self._video_prompt_word_guidance(video_length)
        prompt = f"""
You are a professional AI video prompt writer and social packaging assistant.

Look at the provided reference image and the user's context/vibe, then generate a production pack for a {prompt_count}-video series. Each clip is approximately {video_length} seconds long.

Context and vibe: {context_vibe}

Character A (RIGHT Side): {character_a}
Character B (LEFT Side): {character_b}

Return JSON only in exactly this shape:
{{
  "video_prompts": [
    "prompt 1",
    "prompt 2"
  ],
  "youtube": {{
    "title": "YouTube title",
    "description": "YouTube description with optional hashtags at the end",
    "tags": ["tag1", "tag2", "tag3"]
  }},
  "tiktok": {{
    "description": "TikTok caption with hashtags included"
  }}
}}

Rules:
- Return exactly {prompt_count} items in video_prompts.
- Each video prompt must be a single continuous line of text.
- Numbering is not needed inside the JSON values. The UI will number them separately.
- Extract the setting, lighting, framing, and character identity from the image. Do not invent a new location unless the user vibe clearly requires a stylized variation of the same world.
- Match the context/vibe closely in tone, dialogue style, and power dynamics.
- If the context/vibe implies adult rude comedy, do not sanitize it into softer language. Keep the swearing, dark humour, UK slang, and insulting energy readable and intentional.
- Avoid accidental safe-word substitutions that weaken the joke or tone. Keep obvious rude slang intact when the context clearly wants that vibe.
- If the context mentions an object like a weapon, drink, phone, money, or prop, explicitly describe a character physically holding or using it.
- For two-character scenes, only one character speaks at a time and there must be a clear pause before the reply.
- For one-character scenes, keep the main character fully visible and do not write two-character constraints.
- Keep both important characters fully visible only when the image shows two important characters.
- No new main characters unless the user's instructions clearly require them.
- {word_guidance}
- End every video prompt with this exact bracketed text:
{constraint_text}
- The YouTube title should be clickable and platform-friendly.
- The YouTube tags must be plain tags only, no hashtags, returned as an array.
- The TikTok description should include hashtags naturally.
- Do not include markdown fences.
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=3200,
                    response_mime_type="application/json",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                video_prompts = [str(item).strip() for item in data.get("video_prompts", []) if str(item).strip()]
                youtube = data.get("youtube", {}) if isinstance(data.get("youtube", {}), dict) else {}
                tiktok = data.get("tiktok", {}) if isinstance(data.get("tiktok", {}), dict) else {}
                if len(video_prompts) != prompt_count:
                    raise ValueError(f"Expected {prompt_count} video prompts, got {len(video_prompts)}.")
                reviewed_prompts = self.review_wan_video_prompts(
                    api_key=api_key,
                    model=candidate_model,
                    prompts=video_prompts,
                    video_length=video_length,
                    character_count=character_count,
                    context_vibe=context_vibe,
                    reference_image_path=reference_image_path,
                )
                return {
                    "video_prompts": reviewed_prompts,
                    "youtube": {
                        "title": str(youtube.get("title", "")).strip(),
                        "description": str(youtube.get("description", "")).strip(),
                        "tags": [str(tag).strip() for tag in youtube.get("tags", []) if str(tag).strip()],
                    },
                    "tiktok": {
                        "description": str(tiktok.get("description", "")).strip(),
                    },
                    "raw_output": text,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"WAN structured production pack generation failed: {last_error}")

    def rewrite_wan_prompts_with_character_labels(
        self,
        api_key: str,
        model: str,
        prompts: list[str],
        reference_image_paths: list[str],
        character_mode: str,
        character_a: str,
        character_b: str,
    ) -> list[str]:
        if not api_key:
            raise ValueError("Gemini API key is required.")
        cleaned_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not cleaned_prompts:
            raise ValueError("At least one prompt is required.")
        reference_image_path = next((str(path).strip() for path in reference_image_paths if str(path).strip()), "")
        if not reference_image_path or not Path(reference_image_path).is_file():
            raise ValueError("A valid reference image is required.")
        character_label_guidance = (
            f'Use "Character A" for the right-side or first main character and "Character B" for the left-side or second main character. Character A description: {character_a}. Character B description: {character_b}.'
            if character_mode == "Two Character Conversation"
            else f'Use "Main Character" or "the same character from the reference image". Main character description: {character_a}.'
        )
        technical_constraint_guidance = (
            'If the prompt includes technical constraints, keep them appropriate for exactly two characters, such as "both characters fully visible throughout", only when two characters are actually present.'
            if character_mode == "Two Character Conversation"
            else 'If the prompt includes technical constraints, rewrite them for a single-subject scene, for example "the main character remains fully visible throughout", and never say "both characters" when only one character is present.'
        )
        prompt_lines = "\n".join(f"{index + 1}. {prompt}" for index, prompt in enumerate(cleaned_prompts))
        prompt = f"""
Rewrite the user's WAN video prompts so they refer to subjects using the reference image and character labels instead of franchise names or remembered identities.

Return JSON only in exactly this shape:
{{
  "prompts": [
    "rewritten prompt 1",
    "rewritten prompt 2"
  ]
}}

  Rules:
  - Return exactly {len(cleaned_prompts)} prompts.
  - Keep each prompt as a single continuous line.
  - Preserve the user's intended action, tone, environment, and dialogue as much as possible.
  - Preserve the intended speaker order and preserve the user's quoted dialogue as closely as possible.
  - Keep spoken dialogue in quotation marks when the user's original prompt used quotation marks.
  - Do not paraphrase, summarize, soften, or invent new dialogue when the user already supplied exact lines.
  - Use the character descriptions only as hidden identity guidance for correct mapping.
  - Do not paste, restate, summarize, or prepend the full character descriptions in the rewritten prompt.
  - Do not start the prompt with a long physical description of Character A or Character B unless the user's original prompt already started that way.
  - For two-character prompts, prefer "the right-side character" and "the left-side character" over "Character A" and "Character B".
- Character A must always refer to this exact character description: {character_a}
- Character B must always refer to this exact character description: {character_b}
- Do not swap Character A and Character B.
  - If a line of dialogue says another character's name, treat that as the addressee, not the speaker.
  - Treat Character A as the right-side speaker and Character B as the left-side speaker in the final rewritten output.
  - Make the first spoken line explicitly belong to the correct side, for example "the right-side character leans in and says, \"...\" while the left-side character listens".
  - If Character A says "Patrick" or Character B says "SpongeBob", that is only the spoken name inside the dialogue and must not change which character is speaking.
  - Never add subtitles, captions, text overlay, burned-in dialogue, labels on screen, or any visible on-screen writing.
  - Never rewrite the prompt in a way that suggests the dialogue should appear as text on screen.
  - Do not use screenplay formatting or standalone speaker-label formatting such as "A:", "B:", or separate subtitle-style lines.
  - Fold the dialogue naturally into the scene description using plain prose with inline attribution, while keeping it visually clear who is speaking from the surrounding sentence.
  - For two-character prompts, use this shape whenever possible: scene setup, the right-side character action/expression while speaking first, the left-side character reaction/action while speaking second, then any final reaction, then the technical constraints.
  - Prefer phrasing like "the right-side character glares at the left-side character and says, \"...\"" or "the left-side character smiles nervously before replying, \"...\"".
  - Keep speaker attribution in plain prose, not as a standalone speaker label line.
  - If there are three spoken lines, keep the third line attached to the correct speaker in the same natural-prose style with the original wording preserved.
- Do not use names like SpongeBob, Patrick, Phineas, Peppa, Mario, etc unless the user explicitly asked to keep that exact name.
- {character_label_guidance}
- {technical_constraint_guidance}
- Refer to people by labels tied to the reference image, like "Character A", "Character B", "the character on the left", "the taller character", or "the same character from the reference image".
- Keep the prompt compatible with WAN video generation.
- The final rewritten prompt should read like a clean WAN video prompt, not like a character sheet or an image-caption analysis.
- Do not use markdown fences.

Prompts to rewrite:
{prompt_lines}
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=2200,
                    response_mime_type="application/json",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                rewritten = [str(item).strip() for item in data.get("prompts", []) if str(item).strip()]
                if len(rewritten) == len(cleaned_prompts):
                    return [self._normalize_rewritten_wan_prompt(item, character_mode) for item in rewritten]
                raise ValueError(f"Expected {len(cleaned_prompts)} rewritten prompts, got {len(rewritten)}.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"WAN prompt rewrite failed: {last_error}")

    def review_wan_video_prompts(
        self,
        api_key: str,
        model: str,
        prompts: list[str],
        video_length: str,
        character_count: str,
        context_vibe: str,
        reference_image_path: str = "",
    ) -> list[str]:
        if not prompts:
            raise ValueError("At least one prompt is required.")
        constraint_text = self._wan_constraint_text(character_count)
        word_guidance = self._video_prompt_word_guidance(video_length)
        numbered_prompts = "\n".join(f"{index + 1}. {prompt}" for index, prompt in enumerate(prompts))
        prompt = f"""
Review and polish these WAN video prompts.

Return JSON only in exactly this shape:
{{
  "prompts": [
    "polished prompt 1",
    "polished prompt 2"
  ]
}}

Rules:
- Return exactly {len(prompts)} prompts.
- Keep each prompt as one continuous line.
- Preserve the same scenario and characters.
- Keep the tone aligned with this context/vibe: {context_vibe}
- If the context/vibe clearly wants vulgar adult comedy, keep the swearing, rude insults, dark humour, cheeky criminal tone, and UK slang sharp instead of sanitizing it.
- Fix accidental softening or mistaken word choices that break the intended joke or adult tone.
- Do not replace intended rude slang with safer words unless the user clearly asked for that.
- {word_guidance}
- End each prompt with this exact technical constraint text: {constraint_text}
- For one-character prompts, never say "both characters".
- Do not use markdown fences.

Prompts:
{numbered_prompts}
""".strip()
        last_error: Exception | None = None
        for candidate_model in self._candidate_models(model):
            try:
                payload = self._generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    max_tokens=2600,
                    response_mime_type="application/json",
                    image_path=reference_image_path,
                )
                text = self._extract_text(payload)
                data = self._load_json_object(self._strip_code_fences(text))
                reviewed = [str(item).strip() for item in data.get("prompts", []) if str(item).strip()]
                if len(reviewed) == len(prompts):
                    return reviewed
                raise ValueError(f"Expected {len(prompts)} reviewed prompts, got {len(reviewed)}.")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"WAN prompt review failed: {last_error}")

    def _generate(
        self,
        api_key: str,
        model: str,
        prompt: str,
        max_tokens: int,
        response_mime_type: str = "application/json",
        image_path: str = "",
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "responseMimeType": response_mime_type,
            "temperature": 0.35,
            "maxOutputTokens": max_tokens,
        }
        if "gemini-2.5-flash" in model:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            
        parts = [{"text": prompt}]
        
        if image_path and Path(image_path).is_file():
            import base64
            import mimetypes
            
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/jpeg"
            
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
                
            parts.insert(0, {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_data
                }
            })

        response = requests.post(
            self.API_URL_TEMPLATE.format(model=model),
            timeout=120,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": parts}],
                "generationConfig": generation_config,
            },
        )
        if not response.ok:
            raise RuntimeError(self._extract_error_message(response))
        return response.json()

    def _build_prompt(self, content_mode: str, length_target: int, source_text: str, planning_mode: str, visual_style: str, character_mode: str) -> str:
        if "FACT MODE INPUT" in source_text:
            return self._build_fact_mode_prompt(length_target, source_text)
        tone = self.MODE_GUIDANCE.get(content_mode, self.MODE_GUIDANCE["General Viral Mode"])
        word_target = self._word_target(length_target)
        resolved_planning_mode = self._resolved_planning_mode(planning_mode, source_text)
        scene_count = self._target_scene_count(length_target)
        character_guidance = self._character_guidance(character_mode)
        dialogue_structure_guidance = self._dialogue_structure_guidance(character_mode, source_text)
        listicle_guidance = self._listicle_guidance(content_mode, length_target, resolved_planning_mode)
        comedy_guidance = self._comedy_guidance(source_text, content_mode)
        self_talk_guidance = self._self_talk_guidance(source_text)
        return f"""
Act as a Viral Meme Creator specializing in "Urban Brainrot" and "Roadman" parodies.

Your task is to generate a multi-scene short-form video script for Veo 3.1 Lite and Flow-style audio cues based on my topic.

Return JSON only with this exact shape:
{{
  "title": "SEO title",
  "description": "SEO description",
  "hashtags": ["#tag1", "#tag2"],
  "call_to_action": "short CTA",
  "spoken_script": "complete narration script",
  "stock_footage_tags": ["tag1", "tag2", "tag3"],
  "scenes": [
    {{
      "scene_number": 1,
      "headline": "short scene heading",
      "action_prompt": "full Veo 3.1 style action prompt for this scene",
      "audio_dialogue_cue": "full spoken dialogue or narrator cue for this scene, with spoken words in quotes",
      "narration_text": "the exact narration beat for this scene",
      "purpose": "hook/setup/action/twist",
      "supporting_text": "one or two short lines for visual direction",
      "visual_keywords": ["keyword", "keyword"],
      "mood": "mood",
      "duration_hint": 2.4,
      "visual_query": "cinematic stock footage or reenactment search phrase for this scene",
      "negative_prompt": "what to avoid",
      "camera_style": "camera framing and movement",
      "style_notes": "visual notes to keep the style consistent",
      "overlay_text": ""
    }}
  ]
}}

Generate:
1. Title
2. Description
3. Hashtags
4. CTA
5. Full Script
6. Scene table in spirit of:
   Scene | Action Prompt (Veo 3.1) | Audio / Dialogue Cue

Hard output rules:
- Tone: aggressive, funny, shameless, slightly nonsensical, quotable.
- Use UK road-slang naturally where it fits: bruv, fam, opps, moving mad, on my life, cap, finesse, washed, vexed.
- Structure: exactly {scene_count} scenes.
- Character consistency: refer to the Uploaded Character in every visual prompt when a character image is present.
- Action-oriented: describe clear physical movement, props, expressions, and camera angles.
- Every scene must feel like the next clip in one continuous skit, not a reset or retelling.
- Scene 1 must set up the bit.
- Scene 2 must escalate it.
- Scene 3 must react, confront, or twist it.
- Scene 4 must land the funniest payoff, reveal, or ending.

CRITICAL SCRIPT DISTRIBUTION RULE — READ THIS CAREFULLY:
- The spoken_script is a LINEAR story. Divide it into {scene_count} consecutive, non-overlapping chunks.
- Scene 1 audio_dialogue_cue = lines 1 to roughly 25% of the script (the opening setup lines).
- Scene 2 audio_dialogue_cue = lines from 25% to 50% of the script (the escalation lines).
- Scene 3 audio_dialogue_cue = lines from 50% to 75% of the script (the reaction or twist lines).
- Scene 4 audio_dialogue_cue = lines from 75% to 100% of the script (the payoff/ending lines).
- Each scene's audio_dialogue_cue must contain ONLY the lines assigned to that quarter. NEVER copy or paraphrase lines from a different scene's chunk.
- DO NOT repeat the same sentence in two different scenes under any circumstances.
- DO NOT paraphrase or echo earlier scene dialogue in a later scene.
- Scene 3 must use completely different lines than Scene 1.
- Scene 4 must use completely different lines than Scene 2.
- If the script is 12 lines, assign lines 1-3 to Scene 1, lines 4-6 to Scene 2, lines 7-9 to Scene 3, and lines 10-12 to Scene 4.

Script rules:
- Length: 20 to 45 seconds spoken unless the selected duration is shorter or longer.
- For this request, target about {length_target} seconds total.
- Spoken script target: {word_target}.
- Style: fast-paced, quotable, visual, funny, high-energy speech.
- Never sound like a news article summary.
- Use short punchy lines.
- Use simple language that is easy to understand instantly.
- Every line must be easy to understand on first listen.
- Prioritize the skit and joke progression over explanation.
- Prioritize retention over completeness.
- Create curiosity or escalation every few seconds.
- Make the viewer want to see the next clip.
- Use pattern interrupts.
- Let some lines breathe.
- No filler.
- No repetition.
- Never use robotic wording.
- Never use long paragraphs.
- Never waste the first sentence.
- The first sentence must hook instantly.
- Dialogue should feel like reckless road-talk parody, not realistic criminal instruction.
- {comedy_guidance}
- {self_talk_guidance}
- {dialogue_structure_guidance}

Output requirements:
- Title and description must be SEO-friendly but not bland.
- Keep description to 1 or 2 short sentences.
- Use no more than 6 hashtags.
- CTA must be a single short sentence.
- spoken_script must be written line by line for voiceover pacing, not as one dense paragraph.
- Include {scene_count} scenes for this output.
- Think in terms of a premium storyboard table, not a blog summary.
- Each scene should represent one clear clip beat.
- Each scene must have a cinematic, detailed action_prompt ready for Veo 3.1 Lite.
- Each scene must have an audio_dialogue_cue that reads like final dialogue, narration, or a sound cue.
- Any words meant to be spoken on-screen must be placed in quotation marks inside audio_dialogue_cue.
- If a character is talking, write it like: Dexter: "Calculations don't lie, bruv."
- This matters because quoted dialogue is what signals the video model that the words should actually be spoken.
- Each scene must include narration_text that matches only that scene's beat and can be voiced as its own short segment.
- narration_text may closely match audio_dialogue_cue, but keep it clean and voice-ready.
- Each scene needs distinct cinematic visual language.
- Favor close-ups, props, reactions, movement, attitude, and dramatic background details over generic explanatory visuals.
- Write every scene so the Uploaded Character appears in all of them when a character image is provided.
- If a character image is provided, treat that uploaded image as the main source of character identity and continuity.
- Do not require a typed character description when the image already defines the character.
- Keep the same character identity, silhouette, clothing logic, and face continuity across all scenes.
- Character direction: {character_guidance}
- Do not rely on on-screen text to tell the story.
- overlay_text should be blank unless visually unavoidable.
- stock_footage_tags should contain 10 to 20 reusable search tags for fillers, transitions, and establishing shots.
- Every scene must include purpose, negative_prompt, camera_style, and style_notes.
- Do not include markdown fences.
- Do not mention these instructions.
- {listicle_guidance}

Formatting preference:
- Write scenes like a premium storyboard table.
- Favor scene names like "The Setup", "The Flex", "The Violation", "The Payoff".
- Keep action prompts vivid, visual, physical, and specific.
- Mention the Uploaded Character directly in visual prompts when relevant.
- Use camera directions like close-up, medium shot, low-angle shot, whip pan, push-in, split-screen, handheld chaos.
- Keep audio/dialogue cues punchy, memorable, and ready to perform.
- For comedy or viral skits, let the dialogue be absurd, characterful, quotable, and slightly brainrotted.
- Use short cocky lines, disrespect, flexing, threats, panic, bragging, and absurd overconfidence.
- Favor slang-heavy one-liners over clean formal narration.
- Keep it funny and performative, not realistic crime instruction.
- If the concept says the character is talking to himself, muttering, spiraling, or going mad, then write the lines as self-directed monologue, muttering, or manic self-conversation. Do not make him sound like he is pitching to the audience or addressing an unseen group unless the concept explicitly says that.

Quality bar:
- Never produce a boring summary.
- Every scene must move the skit forward.
- Scene 3 and Scene 4 MUST NOT repeat or echo Scene 1 and Scene 2 in any form.
- The audio_dialogue_cue for Scene 3 must not share any sentences with Scene 1 or Scene 2.
- The audio_dialogue_cue for Scene 4 must not share any sentences with Scene 1, 2, or 3.
- Each scene's dialogue must be a fresh, forward-moving beat of the story.
- Keep the wording clean, vivid, and easy to picture.
- Strong ending required.
- Make the beats connect smoothly so the 4 clips can be stitched together in order.
- Avoid dead air, long setup beats, or scenes that only repeat information.
- Final check before output: read all 4 audio_dialogue_cues. If any two scenes share a sentence, rewrite the later scene before returning JSON.

Source material:
{source_text[:9000]}
""".strip()

    def _build_fact_mode_prompt(self, length_target: int, source_text: str) -> str:
        word_target = self._word_target(length_target)
        fact_count = len(re.findall(r"(?m)^Fact\s+\d+:", source_text)) or 3
        total_scenes = fact_count + 1
        scene_rules = "\n".join(f"- Scene {index + 2} is Fact {index + 1} only." for index in range(fact_count))
        return f"""
You are an elite viral short-form facts writer creating a Veo-ready talking-character short.

Return JSON only with this exact shape:
{{
  "title": "SEO title",
  "description": "SEO description",
  "hashtags": ["#tag1", "#tag2"],
  "call_to_action": "short CTA",
  "intro_script": "short intro hook spoken by the character",
  "spoken_script": "full combined script",
  "stock_footage_tags": ["tag1", "tag2", "tag3"],
  "scenes": [
    {{
      "scene_number": 1,
      "headline": "Intro Hook",
      "action_prompt": "Veo action prompt",
      "audio_dialogue_cue": "quoted spoken line for this clip",
      "narration_text": "voice-ready line for this clip",
      "purpose": "hook",
      "supporting_text": "visual direction",
      "visual_keywords": ["keyword", "keyword"],
      "mood": "animated",
      "duration_hint": 6.0,
      "visual_query": "reference-driven talking character prompt",
      "negative_prompt": "what to avoid",
      "camera_style": "camera framing and movement",
      "style_notes": "visual notes to keep the style consistent",
      "overlay_text": ""
    }}
  ]
}}

Hard rules:
- You must generate a strong upload-ready title, description, hashtags, and CTA for YouTube Shorts, TikTok, and Instagram Reels.
- The title should feel clickable and platform-ready, not generic.
- The description should be short, clean, and usable as the upload description.
- Hashtags should be ready to paste directly into a short-form upload.
- Exactly {total_scenes} scenes.
- Scene 1 is the intro hook clip only.
{scene_rules}
- One fact per clip after the intro. Never merge facts.
- One recurring on-screen character maximum.
- Every scene must be written as a Veo talking clip with quoted spoken lines for native lip sync.
- Keep the same character identity, face, outfit logic, and speaking style across all clips.
- Each clip should feel like the next part of one clean short.
- Do not use two-character dialogue.
- Keep language simple, punchy, and instantly clear.
- Target about {length_target} seconds total.
- Total script target: {word_target}.
- The intro should tease the facts fast and make the viewer want the next clips.
- Each fact clip should focus on one surprising fact and land cleanly.
- No on-screen text. No subtitles. No logos.

Source facts:
{source_text[:9000]}
""".strip()

    def _extract_text(self, payload: dict[str, Any], allow_non_text_success: bool = False) -> str:
        candidates = payload.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    return text
                if "inlineData" in part:
                    continue
        if allow_non_text_success and candidates:
            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason:
                return str(finish_reason)
        if candidates:
            finish_reason = str(candidates[0].get("finishReason", "")).strip()
            usage_metadata = payload.get("usageMetadata", {})
            thoughts_used = 0
            if isinstance(usage_metadata, dict):
                thoughts_used = int(usage_metadata.get("thoughtsTokenCount", 0) or 0)
            if finish_reason == "MAX_TOKENS" and thoughts_used:
                raise ValueError(
                    "Gemini stopped at MAX_TOKENS before returning visible text. "
                    "This usually means hidden thinking consumed the output budget."
                )
        prompt_feedback = payload.get("promptFeedback", {})
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            raise ValueError(f"Gemini blocked the test prompt: {prompt_feedback.get('blockReason')}")
        raise ValueError(f"Gemini response did not include text output. Raw payload: {json.dumps(payload)[:1200]}")

    def _parse_script_data(self, text: str, api_key: str, candidate_models: list[str]) -> dict:
        stripped = self._strip_code_fences(text)
        attempts = [
            text.strip(),
            stripped,
            self._cleanup_json_like_text(stripped),
            self._linewise_json_fix(stripped),
            *self._heuristic_json_variants(stripped),
        ]
        last_error: Exception | None = None
        for attempt in attempts:
            if not attempt:
                continue
            try:
                return self._load_json_object(attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        repaired = self._repair_json_via_gemini(api_key, candidate_models, text)
        if repaired:
            repaired_attempts = [
                repaired,
                self._cleanup_json_like_text(repaired),
                self._linewise_json_fix(repaired),
                *self._heuristic_json_variants(repaired),
            ]
            for attempt in repaired_attempts:
                if not attempt:
                    continue
                try:
                    return self._load_json_object(attempt)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
        raise ValueError(str(last_error) if last_error else "Unable to parse Gemini script JSON.")

    def _repair_json_via_gemini(self, api_key: str, candidate_models: list[str], broken_text: str) -> str:
        repair_prompt = (
            "Convert the following malformed output into valid JSON only. "
            "Do not add commentary. Preserve the intended fields and values. "
            "Repair missing commas, broken arrays, and malformed quoted strings if needed.\n\n"
            f"{broken_text[:16000]}"
        )
        for candidate_model in candidate_models:
            try:
                payload = self._generate(
                    api_key,
                    candidate_model,
                    repair_prompt,
                    1400,
                    response_mime_type="text/plain",
                )
                return self._extract_text(payload).strip()
            except Exception:
                continue
        return ""

    def _load_json_object(self, text: str) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Expected a JSON object in the Gemini response.")
        return json.loads(text[start : end + 1])

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _cleanup_json_like_text(self, text: str) -> str:
        cleaned = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        return cleaned.strip()

    def _heuristic_json_variants(self, text: str) -> list[str]:
        variants: list[str] = []
        cleaned = self._cleanup_json_like_text(text)
        transforms = [
            lambda value: re.sub(r'([}\]"0-9])(\s*\n\s*")', r'\1,\2', value),
            lambda value: re.sub(r'([}\]"0-9])(\s*"[\w_ -]+"\s*:)', r'\1,\2', value),
            lambda value: re.sub(r'([}\]"0-9])(\s*"[^"]+"\s*:)', r'\1,\2', value),
            lambda value: re.sub(r'(\})(\s*\n\s*\{)', r'\1,\2', value),
            lambda value: re.sub(r'(\])(\s*\n\s*\{)', r'\1,\2', value),
            lambda value: re.sub(r'(\})(\s*\{)', r'\1,\2', value),
            lambda value: re.sub(r'(\])(\s*\{)', r'\1,\2', value),
            lambda value: re.sub(r'([}\]"0-9])(\s*"[^"]+"\s*:\s*\{)', r'\1,\2', value),
            lambda value: re.sub(r'(")(\s*\n\s*")', r'\1,\2', value),
        ]
        current = cleaned
        for transform in transforms:
            current = transform(current)
            normalized = current.strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
        return variants

    def _linewise_json_fix(self, text: str) -> str:
        cleaned = self._cleanup_json_like_text(text)
        lines = cleaned.splitlines()
        if len(lines) < 2:
            return cleaned

        fixed: list[str] = []
        for index, line in enumerate(lines):
            current = line.rstrip()
            stripped = current.strip()
            next_stripped = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if stripped and next_stripped:
                if (
                    not stripped.endswith((",", "{", "[", ":"))
                    and not next_stripped.startswith((",", "}", "]"))
                    and (
                        next_stripped.startswith('"')
                        or next_stripped.startswith("{")
                    )
                    and (
                        stripped.endswith('"')
                        or stripped.endswith("}")
                        or stripped.endswith("]")
                        or bool(re.search(r"[-0-9]$", stripped))
                        or stripped.endswith("true")
                        or stripped.endswith("false")
                        or stripped.endswith("null")
                    )
                ):
                    current = f"{current},"
            fixed.append(current)
        return "\n".join(fixed).strip()

    def _normalize_script_data(self, data: dict[str, Any], content_mode: str, length_target: int, planning_mode: str, visual_style: str) -> dict[str, Any]:
        intro_script = ""
        spoken_script = str(
            data.get("spoken_script")
            or data.get("voiceover_script")
            or data.get("script")
            or data.get("narration")
            or data.get("voiceover")
            or ""
        ).strip()
        description = str(data.get("description") or "").strip()
        title = str(data.get("title") or "Short Video").strip()
        visual_style_value = str(visual_style or "Reference-Driven").strip()
        call_to_action = str(data.get("call_to_action") or data.get("cta") or "").strip()
        planning_mode_value = "General"

        hashtags_raw = data.get("hashtags", [])
        if isinstance(hashtags_raw, str):
            hashtags = [token for token in hashtags_raw.replace(",", " ").split() if token]
        else:
            hashtags = [str(item).strip() for item in hashtags_raw if str(item).strip()]
        hashtags = [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags[:6]]

        stock_tags_raw = data.get("stock_footage_tags") or data.get("stock_tags") or data.get("stock_footage_keywords") or []
        if isinstance(stock_tags_raw, str):
            stock_footage_tags = [token.strip() for token in re.split(r"[,\n]", stock_tags_raw) if token.strip()]
        else:
            stock_footage_tags = [str(item).strip() for item in stock_tags_raw if str(item).strip()]
        stock_footage_tags = stock_footage_tags[:20]

        raw_scenes = (
            data.get("scenes")
            or data.get("scene_prompts")
            or data.get("scene_plan")
            or data.get("visual_plan")
            or data.get("video_prompts")
            or []
        )
        normalized_scenes = self._normalize_scenes(raw_scenes, spoken_script, intro_script, content_mode, length_target)
        normalized_scenes = self._ensure_scene_count(normalized_scenes, spoken_script, intro_script, content_mode, length_target)
        if not spoken_script:
            spoken_script = self._compose_spoken_script(intro_script, normalized_scenes)
        if len(stock_footage_tags) < 6:
            stock_footage_tags = self._fallback_stock_tags(normalized_scenes, spoken_script, content_mode)

        return {
            "title": title,
            "description": description,
            "hashtags": hashtags,
            "call_to_action": call_to_action,
            "intro_script": intro_script,
            "spoken_script": spoken_script,
            "visual_style": visual_style_value,
            "planning_mode": planning_mode_value,
            "stock_footage_tags": stock_footage_tags,
            "scenes": normalized_scenes,
        }

    def _is_complete_script_package(self, data: dict[str, Any], length_target: int) -> bool:
        spoken_script = str(data.get("spoken_script", "")).strip()
        intro_script = str(data.get("intro_script", "")).strip()
        scenes = data.get("scenes", [])
        required_scenes = self._target_scene_count(length_target)
        if not isinstance(scenes, list) or len(scenes) < required_scenes:
            return False
        has_scene_narration = all(str(scene.get("narration_text", "")).strip() for scene in scenes[:required_scenes])
        has_scene_visuals = all(
            (
                str(scene.get("action_prompt", "")).strip()
                or str(scene.get("supporting_text", "")).strip()
            )
            and (
                str(scene.get("audio_dialogue_cue", "")).strip()
                or str(scene.get("narration_text", "")).strip()
            )
            and str(scene.get("visual_query", "")).strip()
            for scene in scenes[:required_scenes]
        )
        total_words = len(spoken_script.split())
        if total_words < 12:
            total_words = len(
                " ".join(
                    [intro_script, *[str(scene.get("narration_text", "")).strip() for scene in scenes[:required_scenes]]]
                ).split()
            )
        minimum_words = 12 if length_target <= 30 else 18
        return total_words >= minimum_words and has_scene_narration and has_scene_visuals

    def _to_script_package(self, script_data: dict[str, Any]) -> ScriptPackage:
        scenes = [SceneSpec(**scene) for scene in script_data.get("scenes", [])]
        if not scenes:
            raise ValueError("Gemini did not return a scene plan.")
        return ScriptPackage(
            title=script_data.get("title", ""),
            description=script_data.get("description", ""),
            hashtags=script_data.get("hashtags", []),
            call_to_action=script_data.get("call_to_action", ""),
            spoken_script=script_data.get("spoken_script", ""),
            visual_style=script_data.get("visual_style", "Sketchbook Storytelling"),
            intro_script=script_data.get("intro_script", ""),
            scenes=scenes,
            stock_footage_tags=script_data.get("stock_footage_tags", []),
            planning_mode=script_data.get("planning_mode", "Auto"),
            reuse_estimate=ReuseEstimate(),
        )

    def _build_retry_prompt(self, content_mode: str, length_target: int, source_text: str, partial_data: dict[str, Any], planning_mode: str, visual_style: str, character_mode: str) -> str:
        return (
            "Your previous response was incomplete because the spoken_script and/or scenes were missing or too short. "
            "Return a complete JSON package only.\n\n"
            f"{self._build_prompt(content_mode, length_target, source_text, planning_mode, visual_style, character_mode)}\n\n"
            f"Previous incomplete data:\n{json.dumps(partial_data, ensure_ascii=False)[:6000]}"
        )

    def _normalize_scenes(
        self,
        raw_scenes: Any,
        spoken_script: str,
        intro_script: str,
        content_mode: str,
        length_target: int,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_scenes, list):
            for index, raw_scene in enumerate(raw_scenes, start=1):
                if isinstance(raw_scene, str):
                    normalized.append(self._scene_from_text(raw_scene, index, content_mode))
                    continue
                if isinstance(raw_scene, dict):
                    headline = str(
                        raw_scene.get("headline")
                        or raw_scene.get("title")
                        or raw_scene.get("scene_title")
                        or raw_scene.get("scene_name")
                        or f"Scene {index}"
                    ).strip()
                    supporting_text = str(
                        raw_scene.get("supporting_text")
                        or raw_scene.get("description")
                        or raw_scene.get("action_prompt")
                        or raw_scene.get("prompt")
                        or raw_scene.get("veo_prompt")
                        or raw_scene.get("scene_prompt")
                        or raw_scene.get("visual_direction")
                        or headline
                    ).strip()
                    narration_text = str(
                        raw_scene.get("narration_text")
                        or raw_scene.get("audio_dialogue_cue")
                        or raw_scene.get("dialogue_cue")
                        or raw_scene.get("flow_audio")
                        or raw_scene.get("voiceover_line")
                        or raw_scene.get("script_line")
                        or raw_scene.get("scene_voiceover")
                        or ""
                    ).strip()
                    action_prompt = str(
                        raw_scene.get("action_prompt")
                        or raw_scene.get("veo_prompt")
                        or raw_scene.get("action")
                        or raw_scene.get("visual_query")
                        or raw_scene.get("prompt")
                        or raw_scene.get("scene_prompt")
                        or supporting_text
                        or headline
                    ).strip()
                    audio_dialogue_cue = str(
                        raw_scene.get("audio_dialogue_cue")
                        or raw_scene.get("dialogue_cue")
                        or raw_scene.get("flow_audio")
                        or raw_scene.get("dialogue")
                        or raw_scene.get("audio_cue")
                        or raw_scene.get("narration_text")
                        or narration_text
                    ).strip()
                    visual_keywords_raw = (
                        raw_scene.get("visual_keywords")
                        or raw_scene.get("keywords")
                        or raw_scene.get("tags")
                        or []
                    )
                    if isinstance(visual_keywords_raw, str):
                        visual_keywords = [part.strip() for part in re.split(r"[,|/]", visual_keywords_raw) if part.strip()]
                    else:
                        visual_keywords = [str(item).strip() for item in visual_keywords_raw if str(item).strip()]
                    mood = str(raw_scene.get("mood") or raw_scene.get("tone") or self._default_mood(content_mode)).strip()
                    purpose = str(raw_scene.get("purpose") or raw_scene.get("beat") or f"Scene {index}").strip()
                    negative_prompt = str(
                        raw_scene.get("negative_prompt")
                        or raw_scene.get("avoid")
                        or raw_scene.get("avoid_list")
                        or ""
                    ).strip()
                    camera_style = str(
                        raw_scene.get("camera_style")
                        or raw_scene.get("camera")
                        or raw_scene.get("framing")
                        or ""
                    ).strip()
                    style_notes = str(
                        raw_scene.get("style_notes")
                        or raw_scene.get("visual_style_notes")
                        or raw_scene.get("look")
                        or ""
                    ).strip()
                    overlay_text = str(raw_scene.get("overlay_text") or "").strip()[:40]
                    visual_query = str(
                        raw_scene.get("visual_query")
                        or raw_scene.get("action_prompt")
                        or raw_scene.get("veo_prompt")
                        or raw_scene.get("prompt")
                        or raw_scene.get("scene_prompt")
                        or " ".join([headline, *visual_keywords[:3]])
                    ).strip()
                    duration_hint = self._safe_duration(
                        raw_scene.get("duration_hint") or raw_scene.get("duration_target"),
                        length_target,
                    )
                    normalized.append(
                        {
                            "headline": headline or f"Scene {index}",
                            "action_prompt": action_prompt,
                            "audio_dialogue_cue": audio_dialogue_cue,
                            "narration_text": narration_text or audio_dialogue_cue,
                            "purpose": purpose,
                            "supporting_text": supporting_text or action_prompt or headline or f"Scene {index}",
                            "visual_keywords": visual_keywords[:4] or self._keywords_from_text(headline),
                            "mood": mood,
                            "duration_hint": duration_hint,
                            "visual_query": visual_query or action_prompt or headline,
                            "negative_prompt": negative_prompt,
                            "camera_style": camera_style,
                            "style_notes": style_notes,
                            "overlay_text": overlay_text,
                        }
                    )
        if normalized:
            self._fill_missing_scene_narration(normalized, spoken_script, intro_script)
            self._repair_scene_visuals(normalized, content_mode)
            self._normalize_scene_dialogue_cues(normalized)
            self._deduplicate_scene_dialogue(normalized, spoken_script)
            return normalized[:10]
        return self._fallback_scenes_from_script(spoken_script, intro_script, content_mode, length_target)

    def _ensure_scene_count(
        self,
        scenes: list[dict[str, Any]],
        spoken_script: str,
        intro_script: str,
        content_mode: str,
        length_target: int,
    ) -> list[dict[str, Any]]:
        target_count = self._target_scene_count(length_target)
        working = list(scenes[:10])
        if len(working) >= target_count:
            self._fill_missing_scene_narration(working, spoken_script, intro_script)
            return working

        remaining_script = self._remaining_script_after_scenes(spoken_script, working)
        fallback_scenes = self._fallback_scenes_from_script(
            remaining_script or spoken_script,
            "",
            content_mode,
            length_target,
        )
        fallback_index = 0
        while len(working) < target_count and fallback_index < len(fallback_scenes):
            candidate = dict(fallback_scenes[fallback_index])
            fallback_index += 1
            duplicate = False
            for existing in working:
                if (
                    str(existing.get("headline", "")).strip().lower() == str(candidate.get("headline", "")).strip().lower()
                    and str(existing.get("narration_text", "")).strip().lower() == str(candidate.get("narration_text", "")).strip().lower()
                ):
                    duplicate = True
                    break
            if not duplicate:
                working.append(candidate)

        while len(working) < target_count:
            index = len(working) + 1
            working.append(self._scene_from_text(f"Scene {index} beat.", index, content_mode))

        self._fill_missing_scene_narration(working, spoken_script, intro_script)
        self._repair_scene_visuals(working, content_mode)
        self._normalize_scene_dialogue_cues(working)
        self._deduplicate_scene_dialogue(working, spoken_script)
        return working[:10]

    def _remaining_script_after_scenes(self, spoken_script: str, scenes: list[dict[str, Any]]) -> str:
        lines = [line.strip() for line in spoken_script.splitlines() if line.strip()]
        if not lines:
            return ""
        consumed = 0
        for scene in scenes:
            narration = self._spoken_content_only(str(scene.get("narration_text", "")).strip())
            if not narration:
                continue
            consumed = max(consumed, self._consume_script_lines(lines, narration, consumed))
        remaining = lines[consumed:]
        return "\n".join(remaining).strip()

    def _consume_script_lines(self, lines: list[str], narration: str, start_index: int) -> int:
        target_tokens = self._line_tokens(narration)
        if not target_tokens:
            return start_index
        best_index = start_index
        collected: list[str] = []
        for index in range(start_index, len(lines)):
            collected.extend(self._line_tokens(lines[index]))
            if self._is_prefix_token_match(target_tokens, collected):
                best_index = index + 1
            if len(collected) >= len(target_tokens) + 6:
                break
        return best_index

    def _line_tokens(self, text: str) -> list[str]:
        normalized = self._normalize_line(text)
        return [token for token in normalized.split() if token]

    def _is_prefix_token_match(self, target_tokens: list[str], collected_tokens: list[str]) -> bool:
        if not target_tokens or not collected_tokens:
            return False
        compare_length = min(len(target_tokens), len(collected_tokens))
        return collected_tokens[:compare_length] == target_tokens[:compare_length]

    def _fallback_scenes_from_script(self, spoken_script: str, intro_script: str, content_mode: str, length_target: int) -> list[dict[str, Any]]:
        cleaned = re.sub(r"\s+", " ", spoken_script).strip()
        chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", cleaned) if chunk.strip()]
        if not chunks:
            chunks = [spoken_script.strip()] if spoken_script.strip() else ["Opening hook.", "Main point.", "Key takeaway."]
        target_count = self._target_scene_count(length_target)
        grouped: list[str] = []
        bucket = ""
        for sentence in chunks:
            candidate = sentence if not bucket else f"{bucket} {sentence}"
            if len(candidate) < 120 and len(grouped) + 1 < target_count:
                bucket = candidate
            else:
                if bucket:
                    grouped.append(bucket)
                bucket = sentence
        if bucket:
            grouped.append(bucket)
        grouped = grouped[:8]
        scenes = []
        for index, text in enumerate(grouped, start=1):
            scenes.append(self._scene_from_text(text, index, content_mode))
        self._fill_missing_scene_narration(scenes, spoken_script, intro_script)
        self._repair_scene_visuals(scenes, content_mode)
        self._normalize_scene_dialogue_cues(scenes)
        return scenes or [self._scene_from_text("Opening hook.", 1, content_mode)]

    def _scene_from_text(self, text: str, index: int, content_mode: str) -> dict[str, Any]:
        headline = self._fallback_headline(text, index)
        keywords = self._keywords_from_text(text)
        visual_prompt = self._infer_visual_prompt(headline, text, keywords, content_mode, index)
        return {
            "headline": headline,
            "action_prompt": visual_prompt,
            "audio_dialogue_cue": text.strip(),
            "narration_text": text.strip(),
            "purpose": self._fallback_purpose(index),
            "supporting_text": visual_prompt,
            "visual_keywords": keywords[:4],
            "mood": self._default_mood(content_mode),
            "duration_hint": 2.4,
            "visual_query": " ".join(keywords[:4]) or headline,
            "negative_prompt": "",
            "camera_style": self._fallback_camera_style(index),
            "style_notes": "Reference-driven animated storytelling, expressive character acting, cinematic vertical composition",
            "overlay_text": "",
        }

    def _repair_scene_visuals(self, scenes: list[dict[str, Any]], content_mode: str) -> None:
        for index, scene in enumerate(scenes, start=1):
            action_prompt = str(scene.get("action_prompt", "")).strip()
            dialogue = str(scene.get("audio_dialogue_cue", "")).strip() or str(scene.get("narration_text", "")).strip()
            if self._looks_like_duplicate_scene_text(action_prompt, dialogue):
                keywords = scene.get("visual_keywords", [])
                if not isinstance(keywords, list):
                    keywords = self._keywords_from_text(dialogue or action_prompt)
                scene["action_prompt"] = self._infer_visual_prompt(
                    str(scene.get("headline", "")).strip() or f"Scene {index}",
                    dialogue or action_prompt,
                    [str(item).strip() for item in keywords if str(item).strip()],
                    content_mode,
                    index,
                )
                scene["supporting_text"] = scene["action_prompt"]
                if not str(scene.get("camera_style", "")).strip():
                    scene["camera_style"] = self._fallback_camera_style(index)
                if not str(scene.get("style_notes", "")).strip():
                    scene["style_notes"] = "Reference-driven animated storytelling, expressive character acting, cinematic vertical composition"

    def _looks_like_duplicate_scene_text(self, action_prompt: str, dialogue: str) -> bool:
        left = re.sub(r"[^a-z0-9]+", " ", action_prompt.lower()).strip()
        right = re.sub(r"[^a-z0-9]+", " ", dialogue.lower()).strip()
        if not left or not right:
            return False
        if left == right:
            return True
        return left in right or right in left

    def _fallback_headline(self, text: str, index: int) -> str:
        presets = {
            1: "The Setup",
            2: "The Flex",
            3: "The Warning",
            4: "The Payoff",
            5: "The Ending",
        }
        cleaned = text.strip()
        if len(cleaned.split()) <= 4 and cleaned:
            return cleaned.title()
        return presets.get(index, f"Scene {index}")

    def _fallback_purpose(self, index: int) -> str:
        presets = {
            1: "hook",
            2: "setup",
            3: "escalation",
            4: "payoff",
            5: "ending",
        }
        return presets.get(index, f"scene_{index}")

    def _fallback_camera_style(self, index: int) -> str:
        presets = {
            1: "close-up hero framing, dramatic push-in",
            2: "medium shot, confident character framing",
            3: "dynamic reaction shot, punch-in movement",
            4: "wide or impact shot, energetic reveal framing",
            5: "lingering ending shot, stylish hold",
        }
        return presets.get(index, "cinematic vertical framing")

    def _infer_visual_prompt(self, headline: str, text: str, keywords: list[str], content_mode: str, index: int) -> str:
        mood = self._default_mood(content_mode)
        keyword_text = ", ".join(keywords[:4]) if keywords else "animated lab details"
        beat_guidance = {
            1: "Introduce the character with a strong visual hook and confident body language.",
            2: "Show the character flexing, reacting, or demonstrating control of the environment.",
            3: "Escalate the tension with movement, attitude, or confrontation energy.",
            4: "Land the payoff with a striking reveal, reaction, or impact moment.",
            5: "End with a memorable stylish finishing image.",
        }.get(index, "Keep the beat visual, clear, and character-driven.")
        return (
            f'MEDIUM OR CLOSE SHOT of the recurring character in a cinematic vertical scene titled "{headline}". '
            f"{beat_guidance} Use props, environment details, and reactions inspired by: {keyword_text}. "
            f"The mood should feel {mood}, animated, expressive, and visually clear. "
            "Keep the background rich with stylized story details, strong lighting, and clean subject focus."
        )

    def _normalize_scene_dialogue_cues(self, scenes: list[dict[str, Any]]) -> None:
        speaker = "Character"
        for scene in scenes:
            cue = str(scene.get("audio_dialogue_cue", "")).strip()
            narration = str(scene.get("narration_text", "")).strip()
            if cue:
                normalized = self._normalize_dialogue_cue(cue, speaker)
                scene["audio_dialogue_cue"] = normalized
                speaker = normalized.split(":", 1)[0].strip() or speaker
            elif narration:
                scene["audio_dialogue_cue"] = self._normalize_dialogue_cue(narration, speaker)

    def _deduplicate_scene_dialogue(self, scenes: list[dict[str, Any]], spoken_script: str) -> None:
        """Force-distribute spoken_script evenly across scenes when duplicate dialogue is detected.

        Gemini sometimes dumps the entire script into scenes 3 and 4 instead of assigning each
        scene its own unique portion.  This method detects that via word-overlap ratio and, when
        duplication is found, splits the script into sentence-level chunks (one per scene) so
        every scene gets a unique, sequential slice of the story.
        """
        if len(scenes) < 2:
            return

        # ── helpers ──────────────────────────────────────────────────────────
        def _spoken(scene: dict) -> str:
            raw = self._spoken_content_only(str(scene.get("audio_dialogue_cue", "")).strip())
            return raw or self._spoken_content_only(str(scene.get("narration_text", "")).strip())

        def _wordset(text: str) -> set[str]:
            return set(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())

        def _overlap_ratio(a: str, b: str) -> float:
            wa, wb = _wordset(a), _wordset(b)
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / max(len(wa), len(wb))

        # ── detect duplication ───────────────────────────────────────────────
        contents = [_spoken(s) for s in scenes]
        has_duplicates = False
        for i in range(1, len(contents)):
            for j in range(i):
                if _overlap_ratio(contents[i], contents[j]) > 0.40:
                    has_duplicates = True
                    break
            if has_duplicates:
                break

        if not has_duplicates:
            return

        # ── split the script into sentences ──────────────────────────────────
        raw_script = spoken_script.strip()
        # Split on sentence-ending punctuation, keeping the delimiter attached
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_script) if s.strip()]
        # Fallback: split on double-space or single newline if no punctuation breaks
        if len(sentences) < len(scenes):
            sentences = [s.strip() for s in re.split(r"\s{2,}|\n", raw_script) if s.strip()]
        if len(sentences) < len(scenes):
            sentences = [raw_script]

        n = len(scenes)
        # Distribute sentences into n equal-ish chunks
        chunk_size = max(1, len(sentences) // n)
        chunks: list[str] = []
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n - 1 else len(sentences)
            chunk = " ".join(sentences[start:end]).strip()
            chunks.append(chunk if chunk else sentences[-1])

        # Pad any short tail
        while len(chunks) < n:
            chunks.append(chunks[-1])

        # ── reassign each scene its unique chunk ─────────────────────────────
        for i, scene in enumerate(scenes):
            cue = str(scene.get("audio_dialogue_cue", "")).strip()
            speaker = cue.split(":", 1)[0].strip() if ":" in cue else "Character"
            new_text = chunks[i]
            scene["audio_dialogue_cue"] = f'{speaker}: "{new_text}"'
            scene["narration_text"] = new_text

    def _normalize_dialogue_cue(self, cue: str, fallback_speaker: str) -> str:
        text = cue.strip().replace("“", '"').replace("”", '"').replace("’", "'")
        if not text:
            return ""
        if ":" in text:
            possible_speaker, remainder = text.split(":", 1)
            speaker = possible_speaker.strip() or fallback_speaker
            spoken = remainder.strip()
        else:
            speaker = fallback_speaker
            spoken = text
        quoted_matches = re.findall(r'"([^"]+)"', spoken)
        if quoted_matches:
            spoken_text = " ".join(match.strip() for match in quoted_matches if match.strip())
        else:
            spoken_text = spoken.strip().strip('"')
        spoken_text = spoken_text.strip()
        if not spoken_text:
            spoken_text = text.strip().strip('"')
        return f'{speaker}: "{spoken_text}"'

    def _normalize_line(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _spoken_content_only(self, text: str) -> str:
        value = text.strip().replace("“", '"').replace("”", '"').replace("’", "'")
        if not value:
            return ""
        if ":" in value:
            _, remainder = value.split(":", 1)
            value = remainder.strip()
        quoted_matches = re.findall(r'"([^"]+)"', value)
        if quoted_matches:
            return " ".join(match.strip() for match in quoted_matches if match.strip())
        return value.strip().strip('"')

    def _keywords_from_text(self, text: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9']+", text)
        stop_words = {
            "the", "and", "that", "with", "from", "this", "have", "your", "into", "about",
            "what", "when", "where", "which", "they", "them", "then", "than", "were", "been",
        }
        filtered = []
        for word in words:
            lower = word.lower()
            if len(lower) < 4 or lower in stop_words:
                continue
            if lower not in filtered:
                filtered.append(lower)
        return filtered[:6] or ["story", "scene", "detail"]

    def _default_mood(self, content_mode: str) -> str:
        if content_mode == "True Crime Mode":
            return "suspenseful"
        if content_mode == "Finance Tips Mode":
            return "confident"
        if content_mode == "Facts / Listicle Mode":
            return "animated"
        return "energetic"

    def _safe_duration(self, value: Any, length_target: int) -> float:
        # Veo generates ~5-8s clips per scene; allow up to 8s so durations aren't
        # artificially clamped below what the video model actually produces.
        try:
            duration = float(value)
            if duration > 0:
                return max(3.0, min(duration, 8.0))
        except Exception:
            pass
        # Default: target length divided evenly across 4 scenes, floored at 5s
        scene_count = self._target_scene_count(length_target)
        return max(5.0, round(length_target / scene_count, 1))

    def _word_target(self, length_target: int) -> str:
        if length_target <= 20:
            return "40-70 words"
        if length_target <= 30:
            return "80-120 words"
        if length_target <= 45:
            return "105-145 words"
        return "130-180 words"

    def _max_output_tokens(self, length_target: int) -> int:
        # Each scene has ~12 fields; 4 scenes of JSON easily reaches 2000-3500 tokens.
        # Previous limits (700-1450) caused truncated JSON and parse failures.
        if length_target <= 20:
            return 2000
        if length_target <= 30:
            return 2800
        if length_target <= 45:
            return 3500
        return 4000

    def _target_scene_count(self, length_target: int) -> int:
        if length_target <= 20:
            return 3
        return 4

    def _fallback_stock_tags(self, scenes: list[dict[str, Any]], spoken_script: str, content_mode: str) -> list[str]:
        tags: list[str] = []
        for scene in scenes:
            for token in scene.get("visual_keywords", [])[:3]:
                cleaned = str(token).strip()
                if cleaned and cleaned not in tags:
                    tags.append(cleaned)
        for token in self._keywords_from_text(spoken_script):
            if token not in tags:
                tags.append(token)
        defaults = {
            "True Crime Mode": ["cinematic", "noir", "rain", "streetlight", "mystery", "shadow", "detective", "evidence", "silhouette", "night"],
            "Finance Tips Mode": ["business", "money", "charts", "office", "city", "market", "warning", "strategy", "close-up", "screens"],
            "Facts / Listicle Mode": ["animated", "wild", "creature", "strange", "surprising", "close-up", "reaction", "detail", "fast", "storytelling"],
            "General Viral Mode": ["cinematic", "animated", "storytelling", "dramatic", "close-up", "reaction", "atmospheric", "detail", "motion", "vertical"],
        }
        for token in defaults.get(content_mode, defaults["General Viral Mode"]):
            if token not in tags:
                tags.append(token)
        return tags[:20]

    def _fill_missing_scene_narration(self, scenes: list[dict[str, Any]], spoken_script: str, intro_script: str) -> None:
        available_lines = [line.strip() for line in spoken_script.splitlines() if line.strip()]
        if intro_script.strip() and available_lines and available_lines[0] == intro_script.strip():
            available_lines = available_lines[1:]
        line_index = 0
        for scene in scenes:
            if str(scene.get("narration_text", "")).strip():
                if not str(scene.get("audio_dialogue_cue", "")).strip():
                    scene["audio_dialogue_cue"] = str(scene.get("narration_text", "")).strip()
                continue
            if line_index < len(available_lines):
                scene["narration_text"] = available_lines[line_index]
                scene["audio_dialogue_cue"] = available_lines[line_index]
                line_index += 1
            else:
                scene["narration_text"] = str(scene.get("supporting_text") or scene.get("headline") or "").strip()
                scene["audio_dialogue_cue"] = scene["narration_text"]
            if not str(scene.get("action_prompt", "")).strip():
                scene["action_prompt"] = str(scene.get("visual_query") or scene.get("supporting_text") or scene.get("headline") or "").strip()

    def _compose_spoken_script(self, intro_script: str, scenes: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        if intro_script.strip():
            parts.append(intro_script.strip())
        for scene in scenes:
            narration = str(scene.get("narration_text", "")).strip()
            if narration:
                parts.append(narration)
        return "\n".join(parts).strip()

    def _resolved_planning_mode(self, selected_mode: str, source_text: str) -> str:
        if selected_mode and selected_mode != "Auto":
            return selected_mode
        lowered = source_text.lower()
        list_markers = ["top ", "best ", "craziest ", "facts", "reasons", "things", "creatures", "list"]
        if any(marker in lowered for marker in list_markers):
            return "List / Topic Mode"
        return "Story Mode"

    def _listicle_guidance(self, content_mode: str, length_target: int, planning_mode: str) -> str:
        if content_mode != "Facts / Listicle Mode" and planning_mode != "List / Topic Mode":
            return "If the source reads like a list, optimize for retention instead of trying to include every point."
        if length_target <= 30:
            return (
                "For fast list or fact shorts, do not try to fit every source item into the video. "
                "Select only the strongest 4 to 6 facts or highlights, open with the wildest one, "
                "and move beat-to-beat with a presenter-friendly rhythm."
            )
        return (
            "For list or fact videos, focus on the strongest highlights rather than exhausting the full list. "
            "Use clean transitions, escalating surprises, and an ending that lands on the most memorable fact."
        )

    def _character_guidance(self, character_mode: str) -> str:
        if character_mode == "Two Character Conversation":
            return (
                "Use the uploaded master image as a two-character reference. Preserve both character identities, their left/right spatial relationship, outfit logic, and face continuity across all scenes. "
                "Write each scene so it is clear which character speaks first and which character reacts or replies."
            )
        return (
            "Use the same on-screen character in every scene. If a character image is provided, keep that exact character identity and continuity across the full short."
        )

    def _comedy_guidance(self, source_text: str, content_mode: str) -> str:
        lowered = source_text.lower()
        comedy_markers = [
            "cartoon",
            "beef",
            "brainrot",
            "anime rivalry",
            "rivalry",
            "meme",
            "parody",
            "funny",
            "skit",
        ]
        if any(marker in lowered for marker in comedy_markers):
            return (
                "If the concept feels comedic, rivalry-based, cartoonish, or meme-heavy, write the dialogue like loud quotable road-talk parody: "
                "cocky, disrespectful, absurd, shameless, and instantly funny. Make it feel like characters are chatting reckless nonsense, "
                "flexing, hating, or threatening each other in short viral lines."
            )
        if content_mode == "General Viral Mode":
            return (
                "If the concept naturally leans comedic, do not be too polished. Let the dialogue get sillier, cockier, and more quotable."
            )
        return "Keep the dialogue emotionally strong and highly watchable."

    def _self_talk_guidance(self, source_text: str) -> str:
        lowered = source_text.lower()
        self_talk_markers = [
            "talking to himself",
            "talks to himself",
            "talking to her self",
            "talking to herself",
            "muttering",
            "monologue",
            "going mad",
            "losing his mind",
            "losing her mind",
            "spiraling",
            "arguing with himself",
            "arguing with herself",
        ]
        if any(marker in lowered for marker in self_talk_markers):
            return (
                "This concept is self-talk driven. The character should sound like he is muttering to himself, hyping himself up, "
                "arguing with himself, or spiraling in his own head. Avoid audience-facing lines like 'you heard it here first', "
                "'watch this', or generic threats to random opps unless the concept explicitly says he is addressing someone else."
            )
        return (
            "Only use direct-to-camera or audience-addressed lines when the concept clearly calls for it. Otherwise keep the speech grounded in the scene itself."
        )

    def _dialogue_structure_guidance(self, character_mode: str, source_text: str) -> str:
        if character_mode == "Two Character Conversation":
            return (
                "This is a two-character conversation. Write the spoken beats as explicit turn-taking dialogue between Character A and Character B. "
                "Use quoted lines for each speaker and explicitly label each line with Character A or Character B. Example: Character A says, \"...\" Then Character B replies, \"...\". "
                "There must be a short natural pause between every speaker change. Never overlap dialogue. Only one character may speak at a time. "
                "Do not collapse both characters into one narrator voice. Keep responses short, clear, and easy for Veo to assign to the correct person."
            )
        lowered = source_text.lower()
        if "talking to himself" in lowered or "talking to herself" in lowered:
            return "This is a solo self-talk concept, so keep the lines as one character speaking to himself or herself, not a dialogue between multiple people."
        return "Treat this as a single-character script unless the concept explicitly calls for a second speaker."

    def _candidate_models(self, preferred_model: str) -> list[str]:
        normalized_preferred = self._normalize_model(preferred_model)
        candidates = [normalized_preferred, self.FLASH_LITE_FALLBACK, self.DEFAULT_MODEL]
        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _normalize_model(self, model: str | None) -> str:
        normalized = (model or "").strip()
        if not normalized:
            return self.DEFAULT_MODEL
        if normalized == "gemini-2.5-pro":
            return self.DEFAULT_MODEL
        return normalized

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", {})
            if isinstance(error, dict):
                message = error.get("message") or error.get("status")
                if message:
                    return str(message)
        except Exception:
            pass
        return f"HTTP {response.status_code}"
