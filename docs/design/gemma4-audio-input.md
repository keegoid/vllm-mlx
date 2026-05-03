# Design: Audio Input for Chat Completions

## Problem

Gemma 4 is Google's omnimodal model family supporting text, image, video, and audio input. vllm-mlx already supports text, image, and video input plus tool calling. Audio is the last missing input modality. This document captures the design decisions, model capability findings, and architecture tradeoffs for adding audio input to vllm-mlx chat completions.

## Scope

This feature adds **audio as an input modality to `/v1/chat/completions`** via the OpenAI `input_audio` and `audio_url` content types. It is NOT:

- **STT (speech-to-text)**: vllm-mlx already has `/v1/audio/transcriptions` using dedicated models (Whisper, Parakeet). These are optimized for transcription accuracy, handle long audio (hours), and produce structured text with timestamps.
- **TTS (text-to-speech)**: vllm-mlx already has `/v1/audio/speech` using dedicated models (Kokoro, Chatterbox). These generate speech audio output with voice selection and prosody control.

What we built is the omnimodal case: the LLM itself "hears" the audio and responds, similar to how it "sees" images. The model processes audio through its native audio encoder (Conformer) rather than delegating to a specialized transcription model.

### When to use each

| Capability | Tool | Use case |
|-----------|------|----------|
| Transcribe audio to text | Whisper STT (`/v1/audio/transcriptions`) | Meeting transcription, captioning, voice-to-text input |
| Generate speech from text | Kokoro TTS (`/v1/audio/speech`) | Reading text aloud, voice assistants, accessibility |
| Understand audio content | Gemma 4 E4B chat (`/v1/chat/completions` with `input_audio`) | "What language is this?", "Is the speaker angry?", "Summarize and suggest a reply" |

The key distinction: Whisper turns audio into text. Gemma 4 E4B turns audio into *understanding*. For a voice pipeline (e.g., OpenClaw), the architecture would chain: Whisper STT (reliable transcription of user speech) -> LLM reasoning (Gemma 4 or Qwen3) -> Kokoro TTS (speak the response back). Gemma 4 E4B's audio input is for when you need the model to reason about audio content directly, not just transcribe it.

## Model Capabilities

### Gemma 4 31B IT (text + vision only)

- **Zero audio weights** in both `google/gemma-4-31b-it` and `mlx-community/gemma-4-31b-it-8bit`
- No `audio_config` in config.json
- `audio_tower = None`, `embed_audio = None`
- This is a Google design choice, not a quantization issue -- the original Google model also has no audio weights
- Audio weights cannot be loaded separately; the model architecture was trained without them
- Suitable for: text + vision + tool calling

### Gemma 4 E4B IT (text + vision + audio)

- **754 audio weight keys** in `mlx-community/gemma-4-e4b-it-8bit`
- Full `AudioEncoder` (Conformer, 12 layers), `MultimodalEmbedder`, `Gemma4AudioFeatureExtractor`
- Audio processing pipeline: WAV -> mel spectrogram (128 bins) -> Conformer encoder -> projection -> scatter into embeddings at `<|audio|>` token positions
- Max audio duration: 30 seconds at 16kHz (480,000 samples)
- 4B parameters -- much smaller than 31B, less capable for coding/reasoning/tool use
- Suitable for: audio understanding, multimodal reasoning about sound

### Implications for OpenClaw

Gemma 4 31B cannot replace Qwen3-Coder for OpenClaw with audio support. The 31B model handles text + vision + tool calling but has no audio capability. If audio understanding is needed:

- Keep Qwen3-Coder (or Gemma 4 31B) on port 8000 for primary coding/reasoning
- Run Gemma 4 E4B on a separate port as a dedicated audio understanding service
- For reliable STT, use Whisper (faster, handles long audio, structured output)

Additional blockers for Gemma 4 31B replacing Qwen3:
- Tool parser PR (#254) not merged upstream -- manual install required
- Continuous batching mode untested with Gemma 4 vision + tool calling
- Model size: 31B 8-bit (~31GB) vs Qwen3-Coder 4-bit (~15GB)

## Architecture

### Data Flow

```
HTTP Request (JSON with audio_url or input_audio content parts)
  -> server.py: validate audio capability (reject non-MLLM, reject batched)
  -> server.py: detect media, build chat_kwargs with audio
  -> SimpleEngine.chat() / stream_chat() via **kwargs
  -> MLLM.chat() / stream_chat():
       - _collect_audio_inputs(): extract URLs, decode base64 to temp files
       - Build chat_messages with {"type": "audio"} markers
       - get_chat_template() renders <audio> placeholder tokens
       - mlx_vlm.generate(audio=paths) -> prepare_inputs() -> processor()
       - Gemma4Processor: expand <|audio|> tokens, extract mel features
       - Gemma4 model: audio_tower encodes features, scatter into embeddings
  -> Response
```

### API Format

Two content types supported:

```json
{"type": "audio_url", "audio_url": {"url": "https://example.com/audio.wav"}}
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
```

Supported audio formats: WAV, MP3, M4A, OGG, FLAC (anything miniaudio/librosa supports).

### Key Design Decisions

**1. Audio validation at server boundary**

Audio requests are rejected with HTTP 400 before reaching engines that can't handle them:
- Non-MLLM (text-only) engines: would cause TypeError since LLM models don't accept audio kwargs
- Batched engines: the batch pipeline (MLLMScheduler, MultimodalProcessor, MLLMBatchGenerator) only threads images and videos; threading audio through would require changes to all three components

The batched engine also has defense-in-depth ValueError guards in case of direct engine calls.

**2. No base64 decoding in `extract_multimodal_content()`**

`extract_multimodal_content()` in api/utils.py only extracts `audio_url` URLs, not `input_audio` base64 data. Base64 decoding happens in `MLLM._collect_audio_inputs()` where `_temp_manager` provides tracked cleanup. This prevents unmanaged temp files in the LLM path where audio isn't used.

**3. Audio markers in chat messages**

`chat()` and `stream_chat()` insert `{"type": "audio"}` into the `chat_messages` content array, following the same pattern as `{"type": "image"}`. mlx-vlm's `get_chat_template()` -> `_flatten_content()` renders these as `<audio>` placeholder tokens. The Gemma4Processor then expands each `<audio>` to `<|boa|><|audio|>*N<|eoa|>` based on the actual waveform duration.

**4. Prefix cache disabled for audio**

The MLLM prefix cache keys on `(all_images, formatted_prompt)` but does not include audio content. Rather than extending the cache key (which would require hashing audio files), we disable prefix cache lookup and storage when audio is present. This is correct but means audio requests don't benefit from prefix caching.

Future improvement: include a stable hash of audio file paths/content in the cache key.

**5. Native video path skipped for mixed video+audio**

The native video pipeline (Qwen temporal 3D conv + M-RoPE) doesn't handle audio inputs. When a request contains both video and audio, we skip the native video path and fall through to the standard path which decomposes video into image frames. This is slightly less efficient for video but correctly processes both modalities.

Gemma 4 models don't use the native video path anyway (`video_token_id` not set), so this only affects Qwen-family models in the hypothetical video+audio case.

**6. Per-request temp file cleanup**

`input_audio` base64 payloads are decoded to `NamedTemporaryFile(delete=False)` and registered with `_temp_manager`. A `try/finally` block around `generate()` and `stream_generate()` in both `chat()` and `stream_chat()` ensures cleanup after success, failure, timeout, or client disconnect. A 10MB size guard prevents oversized payloads.

**7. Log redaction**

Request logging redacts multimodal content, showing `[input_audio]`, `[image]`, `[text:43ch]` instead of raw base64 payloads or media URLs.

## Testing

### Unit Tests
- `test_multimodal_with_audio_url`: audio_url extraction from messages
- `test_multimodal_with_audio_url_string`: string-format audio_url
- `test_multimodal_with_audio_and_images`: mixed media extraction
- `test_input_audio_not_extracted_by_utils`: base64 NOT decoded in utils (deferred to MLLM)
- All existing `extract_multimodal_content` tests updated for 4-tuple return

### End-to-End Tests (Gemma 4 E4B on M5 Max 128GB)

| Test | Input | Result |
|------|-------|--------|
| Speech transcription | macOS `say` "Hello, my name is Alice and I live in San Francisco" | Perfect word-for-word transcription |
| Comprehension | Same clip, "What city? What name?" | "San Francisco", "Alice" |
| Weather | "Rain tomorrow, 65 degrees" | Correct extraction, converted "sixty five" to "65" |
| Streaming | Same audio, `stream: true` | Tokens stream correctly |
| Sine wave (440Hz) | Synthetic test tone | Model responded (hallucinated "dog barking" but proved audio processing) |

### Not Tested
- Audio with continuous batching (rejected by design)
- Audio with Gemma 4 31B (no audio weights)
- Audio longer than 30 seconds
- Multiple audio files in one request (mlx-vlm warns and uses only first)
- Audio + video in same request on Qwen models
- Audio via `audio_url` (URL fetch, not base64)

## Files Changed

| File | Change |
|------|--------|
| `api/models.py` | `InputAudio` model, `input_audio` field on `ContentPart` |
| `api/__init__.py` | Export `InputAudio` |
| `api/utils.py` | `extract_multimodal_content()` returns 4-tuple with audio, handles `audio_url` |
| `server.py` | Audio capability validation, media detection, chat_kwargs, log redaction, JSONResponse import |
| `engine/simple.py` | `"input_audio"` in `_MEDIA_TYPES` |
| `engine/batched.py` | Defense-in-depth audio rejection with `input_audio` message scanning |
| `models/mllm.py` | `_collect_audio_inputs()`, audio markers in chat_messages, audio in generate/stream_generate/chat/stream_chat, prefix cache disabled for audio, native video skip, temp file cleanup |
| `tests/test_api_utils.py` | 4 new audio tests, all existing tests updated for 4-tuple |
| `tests/test_native_tool_format.py` | Updated for 4-tuple |
| `tests/test_server.py` | Updated for 4-tuple |
