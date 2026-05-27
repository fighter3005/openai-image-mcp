#!/usr/bin/env python3
"""
MCP Server for OpenAI-Compatible Image API (ComfyUI Wrapper)

Supports image generation and editing via the comfyui-openai-api-wrapper endpoints.
Designed to be consumed by LibreChat as an MCP tool provider.

Tools:
  - generate_image: Generate images from text prompts (via /v1/images/generations)
  - edit_image:     Edit/modify existing images (via /v1/images/edits), supports multiple input images

Configuration via environment variables:
  IMAGE_API_BASE_URL    — Base URL of the OpenAI-compatible API (default: http://127.0.0.1:5000)
  IMAGE_API_TIMEOUT     — Per-request timeout in seconds (default: 600, matches slow generation)
  IMAGE_API_KEY         — Optional Bearer token for the API
  IMAGE_API_MODEL       — Default model to use (default: flux-2-klein-4b)
"""

import base64
import os
import sys

from mcp.server.fastmcp import FastMCP
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("IMAGE_API_BASE_URL", "http://127.0.0.1:5000")
API_TIMEOUT = float(os.environ.get("IMAGE_API_TIMEOUT", "600"))
API_KEY = os.environ.get("IMAGE_API_KEY", "")
API_MODEL = os.environ.get("IMAGE_API_MODEL", "flux-2-klein-4b")

# Internal cache for generated image bytes (keyed by tool call id)
_image_cache: dict[str, tuple[bytes, str]] = {}

client = httpx.Client(
    timeout=httpx.Timeout(API_TIMEOUT, connect=30.0),
    follow_redirects=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    """Build standard request headers."""
    h: dict[str, str] = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _b64_from_url(url: str) -> str:
    """Fetch an image from a URL and return base64-encoded PNG data."""
    resp = client.get(url)
    resp.raise_for_status()
    b64 = base64.b64encode(resp.content).decode("utf-8")
    return b64


def _make_cache_key(tool_call_id: str, image_index: int = 0) -> str:
    """Create a deterministic cache key."""
    return f"{tool_call_id}:{image_index}"


def _cache_image(
    tool_call_id: str, data_item: dict, index: int = 0, output_format: str = "png"
) -> dict:
    """Store raw image bytes in the cache and return a reference dict."""
    b64_json = data_item.get("b64_json", "")
    if not b64_json:
        url = data_item.get("url")
        if url:
            try:
                b64_json = _b64_from_url(url)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to fetch image from URL {url}: {e}",
                }

    raw = base64.b64decode(b64_json)
    cache_key = _make_cache_key(tool_call_id, index)
    _image_cache[cache_key] = (raw, output_format)

    return {
        "status": "success",
        "cache_key": cache_key,
        "format": output_format,
        "index": index + 1,
    }


def _parse_images(images_b64: list[str]) -> list[str]:
    """Strip data-url prefixes from a list of base64 image strings."""
    cleaned = []
    for i, img in enumerate(images_b64):
        clean = img.strip()
        if not clean:
            continue
        if clean.startswith("data:"):
            try:
                clean = clean.split(",", 1)[1]
            except IndexError:
                return [f"Malformed data URL at image index {i}."]
        cleaned.append(clean)

    missing = len(images_b64) - len(cleaned)
    if missing:
        return [*cleaned, f"{missing} image(s) were empty or invalid and will be skipped."]

    return cleaned


def _api_error_response(message: str) -> list[dict]:
    """Return a consistent error response format."""
    return [{"status": "error", "message": message}]


def _process_api_response(
    resp: httpx.Response, tool_call_id: str, output_format: str = "png"
) -> tuple[list[dict], int]:
    """Process an API JSON response and cache image bytes. Returns (results, count)."""
    results: list[dict] = []
    data = resp.json()
    created = data.get("created")
    items = data.get("data", [])

    for idx, item in enumerate(items):
        result = _cache_image(tool_call_id, item, idx, output_format)
        results.append(result)

    return results, len(items)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="openai-image-mcp",
    instructions=(
        "Image generation and editing via an OpenAI-compatible API backed by ComfyUI. "
        f"Default model: {API_MODEL}. Images are returned as base64-encoded data."
    ),
)


@mcp.tool()
def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    width: int = 1024,
    height: int = 1024,
    response_format: str = "b64_json",
    n: int = 1,
    seed: int | None = None,
    output_format: str = "png",
) -> list[dict]:
    """
    Generate images from a text prompt using an OpenAI-compatible API.

    Args:
        prompt:           The text description of the image to generate.
        negative_prompt:  Optional things to avoid in the image.
        steps:            Number of inference steps (API default if omitted).
        cfg_scale:        Guidance scale for classifier-free guidance.
        width:            Image width in pixels (default: 1024).
        height:           Image height in pixels (default: 1024).
        response_format:  'b64_json' for base64 data, 'url' for temporary URLs.
        n:                Number of images to generate (1–10).
        seed:             Optional random seed for reproducible results (-1 or None = random).
        output_format:    Output format: 'png', 'jpeg', or 'webp'.

    Returns:
        A list of result dicts, each containing:
          - status: "success" | "error"
          - cache_key: reference to cached raw bytes
          - format: output_format used
          - index: ordinal position
    """
    if not prompt or not prompt.strip():
        return _api_error_response("prompt is required and must be non-empty.")

    model = API_MODEL  # env var only, no per-request override

    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "n": max(1, min(n, 10)),
        "size": f"{width}x{height}",
        "response_format": response_format,
        "output_format": output_format,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt.strip()
    if steps is not None:
        payload["steps"] = steps
    if cfg_scale is not None:
        payload["cfg_scale"] = float(cfg_scale)
    if seed is not None and seed >= 0:
        payload["seed"] = seed

    try:
        resp = client.post(
            f"{API_BASE_URL}/v1/images/generations",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        results, count = _process_api_response(resp, str(resp.headers.get("x-request-id", "unknown")), output_format)
    except httpx.HTTPStatusError as e:
        error_body = {}
        try:
            error_body = e.response.json()
        except Exception:
            pass
        err_msg = error_body.get("error", {}).get(
            "message", f"API request failed with status {e.response.status_code}"
        )
        return _api_error_response(err_msg)
    except httpx.RequestError as e:
        return _api_error_response(f"Request to API failed: {e}")

    return results


@mcp.tool()
def edit_image(
    images_b64: list[str],
    prompt: str,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    mask_b64: str | None = None,
    response_format: str = "b64_json",
    n: int = 1,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    output_format: str = "png",
) -> list[dict]:
    """
    Edit or inpaint one or more images using a text prompt.

    Accepts multiple input images as a list of base64-encoded strings. Each string
    may optionally include a data URL prefix (e.g. 'data:image/png;base64,...').

    Args:
        images_b64:       List of base64-encoded source images (at least 1 required).
        prompt:           Text description of the edit/inpaint.
        negative_prompt:  Optional things to avoid in the edited result.
        steps:            Number of inference steps (API default if omitted).
        cfg_scale:        Guidance scale for classifier-free guidance.
        mask_b64:         Optional base64-encoded mask for inpainting regions.
        response_format:  'b64_json' for base64 data, 'url' for temporary URLs.
        n:                Number of edited images to generate (1–10).
        width:            Output image width (optional — inferred if omitted).
        height:           Output image height (optional — inferred if omitted).
        seed:             Optional random seed for reproducible results (-1 or None = random).
        output_format:    Output format: 'png', 'jpeg', or 'webp'.

    Returns:
        A list of result dicts with status, cache_key, format, and index.
    """
    if not images_b64 or all(not img.strip() for img in images_b64):
        return _api_error_response("images_b64 must contain at least one non-empty base64 image.")

    # Clean data URL prefixes
    cleaned = _parse_images(images_b64)
    if any(msg.startswith(("Malformed",)) for msg in cleaned):
        return _api_error_response(cleaned[0])

    # Any skip warnings
    skip_msgs = [msg for msg in cleaned if msg.startswith("image(s) were empty")]
    images_to_send = [img for img in cleaned if not img.startswith(("Malformed", "image(s)"))]

    if not images_to_send:
        return _api_error_response(f"All image(s) were empty or invalid. {skip_msgs[0] if skip_msgs else 'Check input.'}")

    # Strip mask prefix too
    clean_mask = None
    if mask_b64 and mask_b64.strip():
        clean_mask = mask_b64.strip()
        if clean_mask.startswith("data:"):
            try:
                clean_mask = clean_mask.split(",", 1)[1]
            except IndexError:
                return _api_error_response("Malformed data URL for mask_b64.")

    model = API_MODEL  # env var only, no per-request override

    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "images": images_to_send,  # use 'images' plural key for multi-image support
        "n": max(1, min(n, 10)),
        "response_format": response_format,
        "output_format": output_format,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt.strip()
    if steps is not None:
        payload["steps"] = steps
    if cfg_scale is not None:
        payload["cfg_scale"] = float(cfg_scale)
    if width and height:
        payload["size"] = f"{width}x{height}"
    elif width or height:
        w = width or 1024
        h = height or 1024
        payload["size"] = f"{w}x{h}"
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    if clean_mask:
        payload["mask"] = clean_mask

    try:
        resp = client.post(
            f"{API_BASE_URL}/v1/images/edits",
            json=payload,
            headers=_headers(),
        )
        resp.raise_for_status()
        results, count = _process_api_response(resp, str(resp.headers.get("x-request-id", "unknown")), output_format)
    except httpx.HTTPStatusError as e:
        error_body = {}
        try:
            error_body = e.response.json()
        except Exception:
            pass
        err_msg = error_body.get("error", {}).get(
            "message", f"API request failed with status {e.response.status_code}"
        )
        return _api_error_response(err_msg)
    except httpx.RequestError as e:
        return _api_error_response(f"Request to API failed: {e}")

    # Include any skip warnings in results
    if skip_msgs:
        results.insert(0, {"status": "warning", "message": skip_msgs[0]})

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server over stdio."""
    print(
        f"[openai-image-mcp] Connecting to API: {API_BASE_URL}",
        file=sys.stderr, flush=True,
    )
    if API_KEY:
        print("[openai-image-mcp] API key configured", file=sys.stderr, flush=True)
    print(f"[openai-image-mcp] Model: {API_MODEL}", file=sys.stderr, flush=True)
    print(f"[openai-image-mcp] Timeout: {API_TIMEOUT}s", file=sys.stderr, flush=True)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
