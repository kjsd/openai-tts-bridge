# 🦀 openai-tts-bridge

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance, modular **OpenAI-compatible & Gradio 4 (Chatterbox) TTS proxy bridge** written in Rust.

It seamlessly bridges **SkyrimNet** (both the native **OpenAI-Compatible TTS** client in `vbeta27+` and the legacy **Chatterbox / Gradio** client) as well as standard **OpenAI TTS clients** (Open WebUI, AnythingLLM, Elixir clients, etc.) with state-of-the-art TTS engines using a plugin-based architecture.

Supported engines:
- **Aratako Irodori-TTS**: Expressive Japanese TTS controlled via inline Unicode style emojis (`👂`, `😮‍💨`, `🤭`, `😊`, `😡`, `😭`, `😰`, `🫶`, `😏`, etc.) and zero-shot reference audio cloning.
- **Zyphra Zonos 2**: High-dimensional emotion vectors, speaking-rate conditioning, zero-shot voice cloning, and ultra-fast 32-bit float to 16-bit PCM WAV conversion.

---

## ⚡ Features

- **Modular Plugin Engine Architecture**:
  - Switch TTS backends seamlessly via environment variable (`TTS_ENGINE=irodori` or `TTS_ENGINE=zonos2`).
  - Extensible `TtsEngine` trait allows adding new speech synthesis engines easily.
- **Full SkyrimNet OpenAI-Compatible TTS Support (`vbeta27+`)**:
  - **Zero-Shot Voice Auto-Enrollment (`POST /v1/audio/voices`)**: Accepts `multipart/form-data` uploads (`title`, `ref_audio`, `ref_text`) from SkyrimNet's `OpenAICompatibleTTSInterface`, automatically registering local NPC `.wav` files (e.g. `femaleeventoned.wav`) without manual pre-registration on the server.
  - **Automatic Re-Enrollment on Restart (`404 voice_not_found`)**: When the bridge restarts and receives a synthesis request for a previously cached `snv1-<hash>` voice ID, it returns `HTTP 404` with `{"error": {"code": "voice_not_found"}}`, prompting SkyrimNet to automatically re-upload the reference `.wav` and retry synthesis transparently.
- **Dual API Compatibility**:
  - **OpenAI Speech & Voice API**: Implements `POST /v1/audio/speech`, `POST /v1/audio/voices`, and `GET /v1/models`.
  - **Gradio 4 (Chatterbox) API**: Emulates Gradio 4 endpoints used by legacy SkyrimNet Chatterbox (`POST /gradio_api/upload`, `POST /gradio_api/call/generate_audio`, SSE event stream, and `/gradio_api/file={path}`).
  - **XTTS / SkyrimNet Routes**: Provides `/speakers`, `/speakers_list`, `/get_speakers`, and `/languages` endpoints.
- **Smart Emotion & Audio Tag Transformation (138+ SkyrimNet Carry Tags)**:
  - Supports all **11 `carry_tags_*.yaml` categories (138 tags)** from SkyrimNet `vbeta27+` (`anger`, `body`, `composure`, `fear`, `joy`, `manner`, `pace`, `sorrow`, `style`, `urgency`, `volume`), `0650_audio_tags.prompt` vocalizations, Higgs-style tags (`emotion-*`, `style-*`, `prosody-*`, `sfx-*`), and `<SE>...</SE>` wrapper stripping.
  - Resilient parser rescues SkyrimNet sentence-boundary split anomalies (e.g. `chuckle] Hello` or `[whisper Hello`).
  - **Engine-specific conversion**:
    - **Irodori-TTS**: Transforms tags directly into all **36 official Irodori Unicode style emojis** inline; safely strips unknown or non-vocalizable tags.
    - **Zonos 2**: Converts tags into multidimensional emotion slider vectors, CFG scales, and pacing multipliers; safely strips tags from speakable text.
- **Dynamic Speaking-Rate Conditioning**:
  - Calibrates pacing dynamically based on emotion and configuration (`DEFAULT_SPEED=1.15` for Zonos2, customizable per request).
- **Zero-Loss Ultra-Fast Audio Conversion**:
  - Converts raw 32-bit float PCM into standard **16-bit signed integer RIFF WAVE** (`Format 1`, 44.1kHz Mono) with clipping protection.
- **Concurrency & Resource Safety**:
  - Built-in async mutex queue serializes backend inference requests, preventing GPU VRAM spikes and collisions.
  - Enforces body size limits (`MAX_BODY_SIZE_MB`, default 100MB) for reference audio uploads.
- **Self-Contained Binary**: Zero runtime dependencies (no Python or Node.js required to run the bridge).

---

## 🎭 Emotion & Audio Tag Mapping Matrix

The bridge parses inline tags in brackets `[...]`, `【...】`, `(...)`, or `*...*` (and strips `<SE>` / `</SE>` wrappers).

| Category / Tags (Examples) | Expression | Zonos 2 Mapping | Irodori-TTS Emoji |
| :--- | :--- | :--- | :--- |
| `[whisper]`, `[whispering]`, `[hushed]`, `[quietly]`, `[murmuring]`, `[conspiratorial]` | Whisper / Soft Ear-level | `happy: 0.2`, `cfg: 1.1`, speed 0.88x | `👂` |
| `[gentle]`, `[gently]`, `[tender]`, `[warmly]`, `[seductive]`, `[sultry]`, `[flirtatious]` | Gentle / Tender / Warm | `happy: 0.2`, `cfg: 1.1`, speed 0.88x | `🫶` |
| `[sigh]`, `[tired]`, `[exhausted]`, `[weary]`, `[resigned]`, `[bored]` | Sigh / Weary / Resigned | `sad: 0.6`, `cfg: 1.15`, speed 0.90x | `😮‍💨` |
| `[pant]`, `[breathless]`, `[breathlessly]` | Shortness of breath / Panting | `surprised: 0.6`, `fear: 0.3` | `🌬️` |
| `[chuckle]`, `[giggle]`, `[snicker]`, `【くすくす】` | Chuckle / Suppressed Laugh | `happy: 0.5`, `cfg: 1.15` | `🤭` |
| `[happy]`, `[happily]`, `[cheerful]`, `[joyful]`, `[hopeful]`, `[proud]`, `[confident]` | Joy / Happy / Proud | `happy: 0.7`, `cfg: 1.2` | `😊` |
| `[laugh]`, `[laughter]`, `[excited]`, `[excitedly]` | Excited / Loud Laughter | `happy: 0.8`, `cfg: 1.2`, speed 1.10x | `😆` |
| `[angry]`, `[furious]`, `[frustrated]`, `[impatient]`, `[shout]`, `[yell]`, `[menacing]`, `[stern]` | Anger / Shout / Commanding | `angry: 0.75`, `cfg: 1.2`, speed 1.15x | `😡` |
| `[sad]`, `[sorrowful]`, `[mournful]`, `[grieving]`, `[melancholy]`, `[tearful]`, `[crying]`, `[sob]` | Sad / Crying / Sorrow | `sad: 0.7`, `cfg: 1.2`, speed 0.90x | `😭` |
| `[trembling]`, `[shaky]`, `[timid]`, `[weak]`, `[hesitant]` | Trembling / Timid / Weak | `fear: 0.7`, `cfg: 1.2` | `🥺` |
| `[fear]`, `[scared]`, `[terrified]`, `[panicked]`, `[nervous]`, `[anxious]`, `[urgent]`, `[desperate]` | Fear / Panicked / Urgent | `fear: 0.7`, `cfg: 1.2` | `😰` |
| `[surprised]`, `[shocked]`, `[gasp]` | Surprise / Shock / Gasp | `surprised: 0.7`, `cfg: 1.2` | `😲` |
| `[pained]`, `[strained]`, `[hoarse]`, `[distress]` | Distressed / In Pain | `sad: 0.5`, `angry: 0.3`, speed 0.90x | `😖` |
| `[worried]`, `[uneasy]` | Worried / Uneasy | `fear: 0.7`, `cfg: 1.2` | `😟` |
| `[annoyed]`, `[exasperated]`, `[cold]`, `[grim]`, `[deadpan]`, `[monotone]`, `[dismissive]` | Annoyed / Cold / Dismissive | `angry: 0.3`, `sad: 0.2`, speed 0.95x | `🙄` |
| `[sarcastic]`, `[mocking]`, `[smug]`, `[playful]`, `[teasing]` | Sarcasm / Teasing / Smirk | `angry: 0.3`, `sad: 0.2`, speed 0.95x | `😏` |
| `[shy]`, `[shyly]`, `[embarrassed]` | Shy / Bashful | `fear: 0.7`, `cfg: 1.2` | `🫣` |
| `[relieved]`, `[calm]`, `[serious]`, `[solemn]`, `[determined]`, `[narration]` | Calm / Relieved / Solemn | Neutral (`cfg: 1.0`, speed 1.00x) | `😌` |
| `[sleepy]`, `[drowsy]` | Sleepy / Sluggish | `sad: 0.6`, speed 0.90x | `😴` |
| `[yawn]` | Yawn | Stripped from spoken text | `🥱` |
| `[pleading]`, `[plead]`, `[beg]` | Pleading / Begging | `sad: 0.7`, speed 0.90x | `🙏` |
| `[drunk]`, `[drunkenly]` | Drunk | `sad: 0.5`, speed 0.90x | `🥴` |
| `[thoughtful]`, `[curious]`, `[confused]`, `[suspicious]` | Thoughtful / Curious | `surprised: 0.7` / Neutral | `🤔` |
| `[groan]`, `[moan]` | Groan / Heavy Breathing | `sad: 0.5`, `angry: 0.3`, speed 0.90x | `🥵` |
| `[clear throat]`, `[cough]`, `[sniff]`, `[sneeze]` | Throat-clearing / Cough / Sniff | Stripped / `sad: 0.5` | `🤧` |
| `[dramatic]`, `[dramatic tone]`, `[advertisement]` | Dramatic / Loudspeaker | `surprised: 0.3`, `sad: 0.2` | `📢` |
| `[sings]`, `[singing]`, `[humming]` | Singing / Humming | `happy: 0.4`, speed 1.05x | `🎵` |
| `[pause]` | Silence / Beat | Stripped from spoken text | `⏸️` |
| `[slowly]`, `[deliberate]`, `[deliberately]` | Slow pacing | Speed 0.75x | `🐢` |
| `[quickly]`, `[rushed]`, `[hurried]`, `[hurriedly]` | Fast pacing | Speed 1.20x | `⏩` |

---

## ⚙️ Configuration

Options can be configured via command-line arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tts-engine` | `TTS_ENGINE` | `zonos2` | Engine to use: `irodori` or `zonos2` |
| `--host` | `HOST` | `0.0.0.0` | Host address to bind bridge server to |
| `-p, --port` | `PORT` | `8080` | Port to listen on |
| `--irodori-url` | `IRODORI_URL` | `http://127.0.0.1:8088` | Base URL of Irodori-TTS server |
| `--irodori-model` | `IRODORI_MODEL` | `irodori-tts` | Model identifier for Irodori-TTS |
| `--zonos-url` | `ZONOS_URL` | `http://127.0.0.1:1919` | Base URL of Zonos 2 FastAPI server |
| `--default-voice` | `DEFAULT_VOICE` | `default` | Fallback speaker identifier |
| `--default-model` | `DEFAULT_MODEL` | `zonos2` | Fallback model name |
| `--default-speed` | `DEFAULT_SPEED` | `1.15` | Base speech speed multiplier for Zonos 2 |
| `--max-body-size-mb` | `MAX_BODY_SIZE_MB` | `100` | Max request body size in MB |
| `--log-level` | `LOG_LEVEL` | `info` | Log level (`error`, `warn`, `info`, `debug`, `trace`) |

---

## 🚀 Quick Start

### 1. Build from Source

```bash
git clone git@github.com:kjsd/openai-tts-bridge.git
cd openai-tts-bridge
cargo build --release
```

The optimized binary will be produced at `target/release/openai-tts-bridge` (~8MB).

### 2. Run with Irodori-TTS Backend (Recommended for SkyrimNet)

```bash
TTS_ENGINE=irodori \
IRODORI_URL=http://127.0.0.1:8088 \
DEFAULT_VOICE=default \
./target/release/openai-tts-bridge
```

### 3. Run with Zonos 2 Backend

```bash
TTS_ENGINE=zonos2 \
ZONOS_URL=http://127.0.0.1:1919 \
DEFAULT_VOICE=default \
./target/release/openai-tts-bridge
```

---

## 🖥️ Running as a systemd Service (Linux)

To run `openai-tts-bridge` as a background daemon on a Linux server:

1. Create a service file at `/etc/systemd/system/openai-tts-bridge.service`:

```ini
[Unit]
Description=OpenAI & Gradio TTS Bridge Server
After=network.target

[Service]
Type=simple
User=minoru
WorkingDirectory=/home/minoru/work/openai-tts-bridge
ExecStart=/home/minoru/work/openai-tts-bridge/target/release/openai-tts-bridge
Restart=always
RestartSec=3
Environment=HOST=0.0.0.0
Environment=PORT=8080
Environment=TTS_ENGINE=irodori
Environment=IRODORI_URL=http://127.0.0.1:8088
Environment=ZONOS_URL=http://127.0.0.1:1919
Environment=DEFAULT_VOICE=default
Environment=DEFAULT_SPEED=1.15
Environment=LOG_LEVEL=info

[Install]
WantedBy=multi-user.target
```

2. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openai-tts-bridge
```

3. Check logs:

```bash
journalctl -u openai-tts-bridge -f
```

---

## 📡 Client Setup & Integration

### 1. SkyrimNet (`vbeta27+` OpenAI-Compatible TTS - Recommended)

In SkyrimNet configuration:
- **TTS Method**: `OpenAI TTS` (`openai`)
- **Endpoint URL**: `http://<server-ip>:8080` (or `/v1/audio/speech`)
- **Service Profile**: `self_hosted` (skips unnecessary reference audio STT)
- **Zero-Shot Voice Cloning**: SkyrimNet automatically uploads local NPC `.wav` files (such as `femaleeventoned.wav` or custom voices) to `POST /v1/audio/voices` and synthesizes speech via `POST /v1/audio/speech` with `snv1-<sha256>` voice IDs. If `openai-tts-bridge` restarts, it responds with `404 voice_not_found`, and SkyrimNet automatically re-enrolls the `.wav` on the fly.

### 2. SkyrimNet (Legacy Chatterbox / Gradio 4 Client)

In SkyrimNet configuration:
- **TTS Engine**: `Chatterbox` (or `Gradio 4`)
- **Server Address**: `http://<server-ip>:8080`
- Character reference voices uploaded by SkyrimNet via Gradio (`/gradio_api/upload`) are automatically processed, converted, and synthesized.

### 3. Standard OpenAI Client (cURL)

```bash
# 1. Optional: Enroll a custom zero-shot reference voice
curl -X POST http://localhost:8080/v1/audio/voices \
  -F "title=snv1-mycustomvoice" \
  -F "ref_audio=@/path/to/sample.wav;type=audio/wav"

# 2. Synthesize speech
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "[conspiratorial] パパ、寒くない？ [sigh]",
    "voice": "snv1-mycustomvoice",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### 4. Health & Models API

```bash
# Health check (includes backend engine status)
curl http://localhost:8080/health

# List available models
curl http://localhost:8080/v1/models
```

---

## 🧪 Testing

Run all unit tests and mock integration tests:

```bash
cargo test
```

Includes comprehensive tests for:
- Float32 to PCM16 WAV audio conversion and header validation.
- Emotion parser and malformed tag recovery (missing brackets, punctuation split anomalies, `<SE>` wrapper stripping).
- Irodori emoji mapper conversions covering all 36 official styles and 138+ SkyrimNet `carry_tags_*.yaml` tags.
- SkyrimNet OpenAI-compatible zero-shot voice enrollment (`POST /v1/audio/voices`) and automatic `404 voice_not_found` re-enrollment flows for both Irodori and Zonos 2.
- Full end-to-end OpenAI API speech synthesis and Gradio 4 upload/generation flows with mocked backends.

---

## 📄 License

MIT License.
