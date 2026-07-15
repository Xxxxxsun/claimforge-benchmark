"""Dependency-free OpenAI-compatible multimodal client."""
from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class RetryableError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact_error_text(text: str) -> str:
    text = re.sub(r"data:[^,]+;base64,[A-Za-z0-9+/=]+", "data:<redacted-base64>", text)
    return re.sub(r"([A-Za-z0-9+/]{120,}=*)", "<redacted-long-token>", text)


def _is_transient_gateway_error(status: int, text: str) -> bool:
    """Some gateway-side timeouts are unfortunately returned as HTTP 400."""
    if status in {408, 409, 425, 429} or status >= 500:
        return True
    normalized = text.lower()
    normalized_unescaped = normalized
    for _ in range(2):
        normalized_unescaped = normalized_unescaped.replace("\\\\", "\\")
    try:
        decoded = bytes(text, "utf-8").decode("unicode_escape").lower()
    except UnicodeDecodeError:
        decoded = normalized
    transient_markers = (
        "connect timed out",
        "connection timed out",
        "request timeout",
        "gateway timeout",
        "service unavailable",
        "pre-004",
        "请求服务异常",
        "调用模型错误",
    )
    escaped_transient_markers = (
        r"\u8bf7\u6c42\u670d\u52a1\u5f02\u5e38",  # 请求服务异常
        r"\u8c03\u7528\u6a21\u578b\u9519\u8bef",  # 调用模型错误
    )
    return any(marker in normalized or marker in normalized_unescaped or marker in decoded for marker in transient_markers) or any(marker in normalized or marker in normalized_unescaped for marker in escaped_transient_markers)


class VisionClient:
    def __init__(self, model: dict[str, Any], timeout: float, image_config: dict[str, Any]):
        self.model, self.provider, self.timeout, self.image_config = model, model["provider"], timeout, image_config

    def image_url(self, path: Path | None, external_url: str | None) -> str:
        transport = self.image_config.get("transport", "base64")
        if transport == "url":
            if external_url:
                if external_url.startswith(("https://", "http://", "oss://", "data:image/")):
                    return external_url
                raise ValueError("url transport requires an http(s), oss, or data:image image_url")
            prefix, local_root = self.image_config.get("urlPrefix"), self.image_config.get("localRoot")
            if path is not None and prefix and local_root:
                try:
                    relative_path = path.resolve().relative_to(Path(local_root).resolve()).as_posix()
                except ValueError as exc:
                    raise ValueError(f"image path is outside image.localRoot: {path}") from exc
                return prefix.rstrip("/") + "/" + urllib.parse.quote(relative_path, safe="/")
            raise ValueError("url transport requires image_url, or image.urlPrefix plus image.localRoot")
        if path is None:
            if not external_url:
                raise ValueError("base64 transport requires image_path or image_url")
            return external_url
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def call(self, system_prompt: str, user_prompt: str, image: str) -> tuple[str, int]:
        if self.model.get("requestFormat") == "gemini_httpstream":
            return self._call_gemini_httpstream(system_prompt, user_prompt, image)
        image_part: dict[str, Any] = {"type": "image_url", "image_url": {"url": image}}
        if self.image_config.get("detail"):
            image_part["image_url"]["detail"] = self.image_config["detail"]
        extra_body = dict(self.provider.get("extraBody", {}))
        for key in self.model.get("omitExtraBodyKeys", []):
            extra_body.pop(key, None)
        payload = {
            "model": self.model["id"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, image_part]},
            ],
            "max_tokens": self.model["maxTokens"],
            **extra_body,
        }
        if not self.model.get("omitTemperature", False):
            payload["temperature"] = self.model["temperature"]
        endpoint = self.provider["apiBase"].rstrip("/") + "/chat/completions"
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {self.provider['apiKey']}", "Content-Type": "application/json", **self.provider.get("headers", {})})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            text = _redact_error_text(exc.read().decode(errors="replace"))[:500]
            if _is_transient_gateway_error(exc.code, text):
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                raise RetryableError(f"HTTP {exc.code}: {text}", retry_after=delay) from exc
            raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetryableError(str(exc)) from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response: {str(body)[:500]}") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            raise RuntimeError("response content is not text")
        return content, int((time.monotonic() - started) * 1000)

    def _gemini_inline_data(self, image: str) -> dict[str, Any]:
        if image.startswith("data:image/") and ";base64," in image:
            header, data = image.split(",", 1)
            mime_type = header.removeprefix("data:").removesuffix(";base64")
            return {"inlineData": {"mimeType": mime_type, "data": data}}
        mime_type = mimetypes.guess_type(image)[0] or self.image_config.get("mimeType") or "image/jpeg"
        return {"inlineData": {"mimeType": mime_type, "data": image}}

    def _extract_gemini_content(self, body: dict[str, Any]) -> str:
        data = body.get("data") if isinstance(body.get("data"), dict) else body
        for key in ("message", "content"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        completion = data.get("completion")
        if isinstance(completion, dict):
            value = completion.get("content")
            if isinstance(value, str) and value:
                return value
            try:
                parts = completion["candidates"][0]["content"]["parts"]
                text = "".join(part.get("text", "") for part in parts if isinstance(part, dict) and not part.get("thought"))
                if text:
                    return text
            except (KeyError, IndexError, TypeError):
                pass
        raise RuntimeError(f"unexpected Gemini response: {_redact_error_text(str(body))[:500]}")

    def _call_gemini_httpstream(self, system_prompt: str, user_prompt: str, image: str) -> tuple[str, int]:
        extra_body = dict(self.provider.get("extraBody", {}))
        for key in self.model.get("omitExtraBodyKeys", []):
            extra_body.pop(key, None)
        params = {
            "use_gemini_httpstream_api": "1",
            "maxOutputTokens": self.model["maxTokens"],
            "includeThoughts": "false",
            "thinkingBudget": 0,
            "responseMimeType": "application/json",
        }
        if not self.model.get("omitTemperature", False):
            params["temperature"] = self.model["temperature"]
        params.update(self.model.get("geminiParams", {}))
        payload = {
            "model": self.model["id"],
            "prompt": [{"role": "user", "parts": [{"text": user_prompt}, self._gemini_inline_data(image)]}],
            "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
            "params": params,
            "passparams": self.model.get("passparams", {}),
            **extra_body,
        }
        endpoint = self.provider["apiBase"].rstrip("/") + "/chat/completions"
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), method="POST", headers={"Authorization": f"Bearer {self.provider['apiKey']}", "Content-Type": "application/json", **self.provider.get("headers", {})})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            text = _redact_error_text(exc.read().decode(errors="replace"))[:500]
            if _is_transient_gateway_error(exc.code, text):
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                raise RetryableError(f"HTTP {exc.code}: {text}", retry_after=delay) from exc
            raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RetryableError(str(exc)) from exc
        return self._extract_gemini_content(body), int((time.monotonic() - started) * 1000)


def retry_delay(attempt: int, configured: list[float]) -> float:
    return float(configured[min(attempt, len(configured) - 1)]) + random.uniform(0, 1)
