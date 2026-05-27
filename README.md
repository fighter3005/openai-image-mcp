# openai-image-mcp — MCP Server for Image Generation & Editing

A Model Context Protocol (MCP) server that bridges [LibreChat](https://github.com/danny-avila/LibreChat) with the [comfyui-openai-api-wrapper](https://github.com/fighter3005/comfyui-openai-api-wrapper) via OpenAI-compatible image endpoints.

**Default model:** `flux-2-klein-4b` (set via env var, not per-request).

## Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text prompts via `/v1/images/generations` |
| `edit_image`     | Edit/inpaint one or more images with a prompt via `/v1/images/edits` |

## Quick Start

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
# Required: point to your ComfyUI OpenAI API wrapper
export IMAGE_API_BASE_URL=http://127.0.0.1:5000

# Default model (overrides per-request model param)
export IMAGE_API_MODEL=flux-2-klein-4b

# Optional: Bearer token for the API
# export IMAGE_API_KEY=sk-...

# Optional: timeout per request (default 600s — image gen is slow)
export IMAGE_API_TIMEOUT=600
```

### 3. Run the server

```bash
python server.py
```

The server communicates over **stdio** (JSON-RPC), which LibreChat natively supports for MCP tool integration.

## LibreChat Configuration

In your `config.yaml` (LibreChat MCP settings), add:

```yaml
mcpServers:
  openai-image-mcp:
    command: "python3"
    args:
      - "/home/youruser/CODING/openai-image-mcp/server.py"
    env:
      IMAGE_API_BASE_URL: "http://127.0.0.1:5000"
      IMAGE_API_MODEL: "flux-2-klein-4b"
      # IMAGE_API_KEY: "sk-your-key-here"
      IMAGE_API_TIMEOUT: "600"
```

## API Parameters Reference

### generate_image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of the image |
| `negative_prompt` | string | — | Things to avoid |
| `steps` | int | API default | Inference steps |
| `cfg_scale` | float | API default | Guidance scale |
| `width` | int | 1024 | Image width |
| `height` | int | 1024 | Image height |
| `response_format` | string | `b64_json` | `b64_json` or `url` |
| `n` | int | 1 | Number of images (1–10) |
| `seed` | int | random | Seed for reproducibility |
| `output_format` | string | `png` | `png`, `jpeg`, or `webp` |

> **Note:** `model` is not a parameter — it's set via the `IMAGE_API_MODEL` env var.

### edit_image

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images_b64` | list\<string\> | *required* | **List** of base64-encoded source images |
| `prompt` | string | *required* | Text description of the edit |
| `negative_prompt` | string | — | Things to avoid in edit |
| `steps` | int | API default | Inference steps |
| `cfg_scale` | float | API default | Guidance scale |
| `mask_b64` | string | — | Optional inpainting mask (base64) |
| `response_format` | string | `b64_json` | `b64_json` or `url` |
| `n` | int | 1 | Number of results (1–10) |
| `width/height` | int | inferred | Output dimensions |
| `seed` | int | random | Seed for reproducibility |
| `output_format` | string | `png` | `png`, `jpeg`, or `webp` |

> **Note:** `images_b64` accepts a list — pass one or more base64 strings. Data URL prefixes (e.g. `data:image/png;base64,...`) are stripped automatically.
> **Note:** `model` is not a parameter — it's set via the `IMAGE_API_MODEL` env var.

## Supported Models (from ComfyUI workflows)

- `flux-2-dev-turbo`
- `flux-2-dev`
- `flux-2-klein-4b` ← default
- `flux-dev-checkpoint`
- `flux-kontext-dev`
- `flux-krea-dev`
- `flux-schnell`
- `qwen-image-2025`
- `z-image-turbo`
- `z-image`

## Notes

- Images are returned as base64-encoded data in the tool response (MCP stdio transport is JSON-only).
- The API timeout defaults to 600 seconds since flux generation can take several minutes.
- If the API requires authentication, set `IMAGE_API_KEY`.
- The model is configured once via env var `IMAGE_API_MODEL` — no per-request model switching.
