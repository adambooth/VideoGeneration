from __future__ import annotations

import time
from tempfile import mkdtemp
from pathlib import Path


class DeeVidService:
    HOME_URL = "https://deevid.ai/"

    def check_available(self) -> tuple[bool, str]:
        try:
            self._import_playwright()
        except Exception as exc:  # noqa: BLE001
            return False, f"Playwright unavailable: {exc}"
        return True, "Ready"

    def open_login_browser(self, profile_dir: str) -> None:
        sync_playwright = self._import_playwright()
        profile_path = Path(profile_dir).expanduser().resolve()
        profile_path.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=False,
                accept_downloads=True,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(self.HOME_URL, wait_until="domcontentloaded")
            page.bring_to_front()
            page.wait_for_event("close", timeout=0)
            context.close()

    def generate_scene(
        self,
        *,
        profile_dir: str,
        image_path: str,
        prompt: str,
        output_path: str,
        timeout_seconds: int = 420,
    ) -> str:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Scene start image not found: {image_path}")
        if not prompt.strip():
            raise ValueError("DeeVid prompt is empty.")

        sync_playwright = self._import_playwright()
        profile_path = Path(profile_dir).expanduser().resolve()
        profile_path.mkdir(parents=True, exist_ok=True)
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=False,
                accept_downloads=True,
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(self.HOME_URL, wait_until="domcontentloaded")
            page.bring_to_front()
            self._upload_image(page, image_path)
            self._fill_prompt(page, prompt)
            self._run_generation(page)
            downloaded = self._download_result(page, context, timeout_seconds)
            if target_path.exists():
                target_path.unlink()
            downloaded.replace(target_path)
            context.close()
        return str(target_path)

    def _upload_image(self, page, image_path: str) -> None:
        file_input = page.locator("input[type='file']").first
        if file_input.count():
            file_input.set_input_files(image_path)
            return
        add_button = page.get_by_text("Add an image to the prompt", exact=False).first
        if add_button.count():
            with page.expect_file_chooser() as chooser_info:
                add_button.click()
            chooser = chooser_info.value
            chooser.set_files(image_path)
            return
        raise RuntimeError("Could not find DeeVid image upload control.")

    def _fill_prompt(self, page, prompt: str) -> None:
        textarea = page.locator("textarea").first
        if textarea.count():
            textarea.fill(prompt)
            return
        content_box = page.locator("[contenteditable='true']").first
        if content_box.count():
            content_box.fill(prompt)
            return
        raise RuntimeError("Could not find DeeVid prompt input box.")

    def _run_generation(self, page) -> None:
        for label in ("Run", "Create", "Generate"):
            button = page.get_by_role("button", name=label).first
            if button.count():
                button.click()
                return
        raise RuntimeError("Could not find DeeVid generate button.")

    def _download_result(self, page, context, timeout_seconds: int) -> Path:
        download_dir = Path(mkdtemp(prefix="avc-deevid-download-"))
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            for label in ("Download", "Save", "Export"):
                button = page.get_by_role("button", name=label).first
                if button.count():
                    with page.expect_download(timeout=15000) as download_info:
                        button.click()
                    download = download_info.value
                    target = download_dir / download.suggested_filename
                    download.save_as(str(target))
                    return target
            anchor = page.locator("a[download]").first
            if anchor.count():
                with page.expect_download(timeout=15000) as download_info:
                    anchor.click()
                download = download_info.value
                target = download_dir / download.suggested_filename
                download.save_as(str(target))
                return target
            time.sleep(3)
            page.wait_for_timeout(1500)
        raise RuntimeError("Timed out waiting for DeeVid video download.")

    def _import_playwright(self):
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Install Playwright and run 'playwright install chromium' for DeeVid automation.") from exc
        return sync_playwright
