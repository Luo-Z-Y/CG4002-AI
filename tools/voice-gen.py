"""
Pokemon Voice Data Generator v3 — Gemini TTS + Cloud TTS Hybrid
=================================================================
Uses TWO engines for maximum realism:

  1) GEMINI TTS (primary) — Google's newest model with natural-language style
     prompts. You can literally say "speak with a Singaporean accent, excitedly"
     and it obeys. Much more expressive than old Cloud TTS.

  2) CLOUD TTS (fallback) — The en-SG Standard/Wavenet voices from your
     original script, with SSML + post-processing. Used when Gemini TTS
     is unavailable or hits rate limits.

Generates 200 WAV files per Pokemon for 6 Pokemon (1,200 total).

═══════════════════════════════════════════════════════════════════
SETUP — choose ONE of the two authentication methods:
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │  OPTION A: Gemini API key (simplest, free tier)         │
  │                                                         │
  │  1. Go to https://aistudio.google.com/apikey            │
  │  2. Create an API key                                   │
  │  3. Set it:                                             │
  │       export GEMINI_API_KEY="AIza..."                   │
  │                                                         │
  │  pip install google-genai pydub numpy scipy requests    │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  OPTION B: Vertex AI (GCP project, more quota)          │
  │                                                         │
  │  1. Enable "Vertex AI API" in your GCP project          │
  │  2. Authenticate:                                       │
  │       gcloud auth application-default login             │
  │  3. Set env vars:                                       │
  │       export GCP_PROJECT="my-project-id"                │
  │       export GCP_LOCATION="us-central1"                 │
  │       export USE_VERTEX=1                               │
  │                                                         │
  │  pip install google-genai pydub numpy scipy requests    │
  └─────────────────────────────────────────────────────────┘

You also need ffmpeg:
    macOS:   brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg

Usage:
    python generate_pokemon_voices_v3.py
    python generate_pokemon_voices_v3.py -n 50          # 50 per Pokemon (test run)
    python generate_pokemon_voices_v3.py --engine gemini # Gemini TTS only
    python generate_pokemon_voices_v3.py --engine cloud  # Cloud TTS only
    python generate_pokemon_voices_v3.py --engine hybrid # both (default)
"""

import os
import io
import wave
import base64
import random
import time
import argparse
import struct
import json
import re
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import requests

VERBOSE = False
REQUEST_TIMEOUT_SEC = 30
GEMINI_MAX_RETRIES = 1
SLEEP_BETWEEN_SAMPLES_SEC = 0.0


def debug_log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def _extract_retry_delay_seconds(error_text: str) -> Optional[float]:
    patterns = [
        r"Please retry in ([0-9.]+)s",
        r"'retryDelay': '([0-9.]+)s'",
        r'"retryDelay": "([0-9.]+)s"',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text)
        if match:
            return float(match.group(1))
    return None


def parse_pokemon_selection(selection: str | None) -> list[str]:
    if not selection:
        return list(POKEMONS)
    requested = [item.strip() for item in selection.split(",") if item.strip()]
    by_key = {name.lower(): name for name in POKEMONS}
    resolved: list[str] = []
    for item in requested:
        key = item.lower()
        if key not in by_key:
            raise ValueError(f"Unsupported Pokemon: {item}. Choose from: {', '.join(POKEMONS)}")
        resolved.append(by_key[key])
    return resolved

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

POKEMONS = ["Bulbasaur", "Charizard", "Pikachu", "Mewtwo", "Lugia", "Greninja"]
ACTIVE_POKEMONS = POKEMONS
SAMPLES_PER_POKEMON = 200

# Cloud TTS fallback key (from your original script)
CLOUD_TTS_API_KEY = os.environ.get(
    "CLOUD_TTS_API_KEY",
    "AIzaSyBb6ze6U_AB9HQdzg4_J6H2toBlmDzagBM"
)
CLOUD_TTS_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={CLOUD_TTS_API_KEY}"


# ─────────────────────────────────────────────────────────────
# GEMINI TTS — announcer-style prompts
# ─────────────────────────────────────────────────────────────

# Available Gemini TTS voices (30 speakers)
GEMINI_VOICES = [
    "Aoede", "Charon", "Elara", "Fenrir", "Kore",
    "Leda", "Orus", "Puck", "Zephyr", "Achernar",
    "Gacrux", "Pulcherrima", "Vindemiatrix", "Sadachbia",
    "Sadaltager", "Sulafat", "Laomedeia", "Autonoe",
    "Callirrhoe", "Umbriel", "Enceladus", "Iapetus",
    "Dione", "Oberon", "Despina", "Erinome",
    "Algieba", "Rasalgethi", "Schedar", "Zubenelgenubi",
]

ANNOUNCER_DELIVERY_PROMPTS = [
    "battle announcer, short and punchy",
    "game announcer, energetic and direct",
    "live arena announcer, bright and forceful",
    "street-event announcer, fast and excited",
]

ANNOUNCER_ACCENT_PROMPTS = [
    "Singapore English with a noticeable local Singaporean accent",
    "Singapore English with noticeable Malay Singaporean influence",
    "Singapore English with noticeable Chinese Singaporean influence",
    "Singapore English with noticeable Indian Singaporean influence",
]

ANNOUNCER_TEXTURE_PROMPTS = [
    "slightly rough live microphone sound, not studio clean",
    "outdoor PA system sound, a bit gritty and compressed",
    "small event speaker sound, slightly noisy and imperfect",
    "handheld mic sound, mildly distorted and less polished",
]

GEMINI_STYLE_PROMPTS = {
    pokemon: [
        f"{delivery}, {accent}, {texture}, keep the keyword understandable"
        for delivery in ANNOUNCER_DELIVERY_PROMPTS
        for accent in ANNOUNCER_ACCENT_PROMPTS
        for texture in ANNOUNCER_TEXTURE_PROMPTS
    ]
    for pokemon in POKEMONS
}

# Spoken summon keywords only. No punctuation, no extra wording.
POKEMON_NAME_TEXTS = {pokemon: [pokemon] for pokemon in POKEMONS}


# ─────────────────────────────────────────────────────────────
# GEMINI TTS ENGINE
# ─────────────────────────────────────────────────────────────

def init_gemini_client():
    """
    Initialise the google-genai client.
    Supports both Gemini API (API key) and Vertex AI (GCP project).
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("[WARN] google-genai not installed. Run: pip install google-genai")
        return None, None

    use_vertex = os.environ.get("USE_VERTEX", "").strip()

    if use_vertex == "1":
        project = os.environ.get("GCP_PROJECT", "")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if not project:
            print("[WARN] USE_VERTEX=1 but GCP_PROJECT not set.")
            return None, None
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        print(f"[OK] Vertex AI client initialised (project={project}, location={location})")
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set. Gemini TTS unavailable.")
            print("       Get one free at https://aistudio.google.com/apikey")
            return None, None
        client = genai.Client(api_key=api_key)
        print("[OK] Gemini API client initialised")

    return client, genai_types


def gemini_synthesise(
    client,
    genai_types,
    text: str,
    style_prompt: str,
    voice_name: str,
    temperature: float = 1.0,
) -> Optional[bytes]:
    """
    Generate speech using Gemini TTS.
    Returns raw PCM audio bytes (24kHz, 16-bit, mono) or None on failure.
    """
    full_prompt = (
        "Generate speech audio that says exactly one Pokemon summon keyword and nothing else. "
        "Do not speak the instructions. Do not add any extra words, intro, outro, commentary, or explanation. "
        f"Speaking style: {style_prompt}. "
        f'The exact keyword to speak is: "{text}"'
    )
    fallback_prompt = (
        "Say exactly this one keyword and nothing else. "
        "Use a Singapore English accent with a slightly rough live microphone sound. "
        f'Keyword: "{text}"'
    )
    active_prompt = full_prompt

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=active_prompt,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                    temperature=temperature,
                ),
            )

            # Extract raw audio bytes
            data = response.candidates[0].content.parts[0].inline_data.data
            return data

        except Exception as e:
            message = str(e)
            print(f"    [WARN] Gemini TTS error: {message}")
            if ("400" in message or "INVALID_ARGUMENT" in message) and active_prompt != fallback_prompt:
                print("    [INFO] Retrying Gemini with a simpler fallback prompt...", flush=True)
                active_prompt = fallback_prompt
                continue
            retry_delay = _extract_retry_delay_seconds(message)
            should_retry = (
                "429" in message
                and retry_delay is not None
                and attempt < GEMINI_MAX_RETRIES
            )
            if should_retry:
                wait_sec = max(1.0, retry_delay + 1.0)
                print(f"    [INFO] Waiting {wait_sec:.1f}s for Gemini quota reset before retrying...", flush=True)
                time.sleep(wait_sec)
                continue
            return None

    return None


def pcm_to_audiosegment(pcm_data: bytes, sample_rate: int = 24000) -> AudioSegment:
    """Convert raw PCM bytes (16-bit mono) to a pydub AudioSegment."""
    return AudioSegment(
        data=pcm_data,
        sample_width=2,       # 16-bit
        frame_rate=sample_rate,
        channels=1,
    )


# ─────────────────────────────────────────────────────────────
# CLOUD TTS ENGINE (fallback — your original approach, improved)
# ─────────────────────────────────────────────────────────────

SG_VOICES = [
    ("en-SG", "en-SG-Standard-A"), ("en-SG", "en-SG-Standard-B"),
    ("en-SG", "en-SG-Standard-C"), ("en-SG", "en-SG-Standard-D"),
    ("en-SG", "en-SG-Wavenet-A"),  ("en-SG", "en-SG-Wavenet-B"),
    ("en-SG", "en-SG-Wavenet-C"),  ("en-SG", "en-SG-Wavenet-D"),
]

CLOUD_NAME_SSML_TEMPLATES = [
    '<speak><prosody rate="{rate}" pitch="{pitch}"><emphasis level="strong">{name}</emphasis></prosody></speak>',
    '<speak><prosody rate="{rate}" pitch="{pitch}"><break time="80ms"/>{name}<break time="80ms"/></prosody></speak>',
    '<speak><prosody rate="{rate}" pitch="{pitch}" volume="{volume}">{name}</prosody></speak>',
    '<speak><prosody rate="{rate}" pitch="{pitch}"><emphasis level="moderate">{name}!</emphasis></prosody></speak>',
]


def cloud_tts_synthesise(ssml: str, lang_code: str, voice_name: str,
                         pitch: float, rate: float) -> Optional[bytes]:
    """Call Cloud TTS and return MP3 bytes."""
    payload = {
        "input": {"ssml": ssml},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "MP3",
            "pitch": pitch,
            "speakingRate": rate,
        },
    }
    try:
        debug_log(f"    [cloud] requesting {voice_name} ({lang_code}) pitch={pitch} rate={rate}")
        resp = requests.post(CLOUD_TTS_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        if resp.status_code == 200:
            return base64.b64decode(resp.json()["audioContent"])
        debug_log(f"    [cloud] primary voice failed: HTTP {resp.status_code} {resp.text[:300]}")
        # Fallback voice
        payload["voice"] = {"languageCode": "en-SG", "name": "en-SG-Standard-B"}
        retry = requests.post(CLOUD_TTS_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        if retry.status_code == 200:
            return base64.b64decode(retry.json()["audioContent"])
        debug_log(f"    [cloud] fallback voice failed: HTTP {retry.status_code} {retry.text[:300]}")
        return None
    except Exception as exc:
        debug_log(f"    [cloud] request error: {type(exc).__name__}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# AUDIO POST-PROCESSING (shared by both engines)
# ─────────────────────────────────────────────────────────────

def add_background_noise(audio: AudioSegment, intensity_db: float = -30) -> AudioSegment:
    """Pink noise overlay to simulate real recording conditions."""
    n_samples = int(len(audio) * audio.frame_rate / 1000)
    white = np.random.randn(n_samples).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = scipy_signal.lfilter(b, a, white).astype(np.float32)
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    pink = (pink * 32767 * (10 ** (intensity_db / 20))).astype(np.int16)
    noise = AudioSegment(data=pink.tobytes(), sample_width=2,
                         frame_rate=audio.frame_rate, channels=1)
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    elif len(noise) < len(audio):
        noise = noise + AudioSegment.silent(duration=len(audio) - len(noise))
    return audio.overlay(noise)


def add_echo(audio: AudioSegment, delay_ms: int = 150, decay_db: float = -8) -> AudioSegment:
    echo = AudioSegment.silent(duration=delay_ms) + audio + AudioSegment.silent(duration=delay_ms)
    echo = echo[:len(audio) + delay_ms].apply_gain(decay_db)
    padded = audio + AudioSegment.silent(duration=delay_ms)
    return padded.overlay(echo)


def add_reverb(audio: AudioSegment, num_reflections: int = 4) -> AudioSegment:
    result = audio
    for i in range(1, num_reflections + 1):
        delay = int(30 * i * random.uniform(0.8, 1.2))
        decay = -4 * i + random.uniform(-1, 1)
        delayed = AudioSegment.silent(duration=delay) + audio.apply_gain(decay)
        if len(delayed) > len(result):
            result = result + AudioSegment.silent(duration=len(delayed) - len(result))
        elif len(delayed) < len(result):
            delayed = delayed + AudioSegment.silent(duration=len(result) - len(delayed))
        result = result.overlay(delayed)
    return result


def add_distortion(audio: AudioSegment, gain_db: float = 8) -> AudioSegment:
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples * (10 ** (gain_db / 20))
    samples = np.tanh(samples / 32768) * 32768
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return AudioSegment(data=samples.tobytes(), sample_width=2,
                        frame_rate=audio.frame_rate, channels=audio.channels)


def add_lofi_colour(audio: AudioSegment) -> AudioSegment:
    original_rate = audio.frame_rate
    degraded_rate = random.choice([11025, 12000, 14000, 16000])
    return audio.set_frame_rate(degraded_rate).set_frame_rate(original_rate)


def apply_random_effects(audio: AudioSegment, pokemon: str) -> AudioSegment:
    """Apply rough live-capture variation while preserving the spoken keyword."""

    if random.random() < 0.60:
        audio = add_background_noise(audio, intensity_db=random.uniform(-32, -20))

    if random.random() < 0.35:
        audio = add_echo(audio, delay_ms=random.randint(50, 160),
                         decay_db=random.uniform(-18, -10))

    if random.random() < 0.35:
        audio = add_reverb(audio, num_reflections=random.randint(2, 5))

    if random.random() < 0.45:
        audio = add_lofi_colour(audio)

    if random.random() < 0.28:
        audio = add_distortion(audio, gain_db=random.uniform(2, 7))

    eq_roll = random.random()
    if eq_roll < 0.35:
        audio = low_pass_filter(audio, cutoff=random.randint(1800, 3400))
    elif eq_roll < 0.60:
        audio = high_pass_filter(audio, cutoff=random.randint(280, 900))

    audio = audio.apply_gain(random.uniform(-3, 2))

    if random.random() < 0.30:
        audio = audio.fade_in(random.randint(10, 90)).fade_out(random.randint(20, 140))

    pad_before = random.randint(0, 220)
    pad_after = random.randint(0, 180)
    audio = AudioSegment.silent(duration=pad_before) + audio + AudioSegment.silent(duration=pad_after)

    return audio


# ─────────────────────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────────────────────

def generate_one_gemini(client, genai_types, pokemon: str) -> Optional[AudioSegment]:
    """Generate one spoken summon-name sample using Gemini TTS."""
    voice = random.choice(GEMINI_VOICES)
    style = random.choice(GEMINI_STYLE_PROMPTS[pokemon])
    spoken_text = random.choice(POKEMON_NAME_TEXTS[pokemon])
    temperature = random.uniform(0.8, 2.0)

    pcm = gemini_synthesise(client, genai_types, spoken_text, style, voice, temperature)
    if pcm is None:
        return None
    return pcm_to_audiosegment(pcm, sample_rate=24000)


def generate_one_cloud(pokemon: str) -> Optional[AudioSegment]:
    """Generate one spoken summon-name sample using Cloud TTS."""
    lang_code, voice_name = random.choice(SG_VOICES)
    template = random.choice(CLOUD_NAME_SSML_TEMPLATES)
    ssml = template.format(
        name=pokemon,
        rate=random.choice(["x-slow", "slow", "medium", "fast", "x-fast"]),
        pitch=f"{random.uniform(-8, 8):+.1f}st",
        volume=random.choice(["medium", "loud", "x-loud"]),
    )
    pitch = round(random.uniform(-8, 8), 1)
    rate = round(random.uniform(0.6, 1.5), 2)

    mp3_bytes = cloud_tts_synthesise(ssml, lang_code, voice_name, pitch, rate)
    if mp3_bytes is None:
        return None
    return AudioSegment.from_mp3(io.BytesIO(mp3_bytes))


def generate_all(output_root: str = "pokemon_voices", engine: str = "hybrid"):
    """
    Main loop: generate SAMPLES_PER_POKEMON audio files for each Pokemon.

    engine: "gemini" | "cloud" | "hybrid"
      - gemini: Gemini TTS only
      - cloud:  Cloud TTS only (en-SG voices + SSML)
      - hybrid: 70% Gemini, 30% Cloud (best variety)
    """

    os.makedirs(output_root, exist_ok=True)

    # Init Gemini if needed
    gemini_client, genai_types = None, None
    if engine in ("gemini", "hybrid"):
        gemini_client, genai_types = init_gemini_client()
        if gemini_client is None and engine == "gemini":
            print("[FATAL] Gemini TTS unavailable and engine='gemini'. Aborting.")
            return
        if gemini_client is None and engine == "hybrid":
            print("[INFO] Gemini unavailable — falling back to Cloud TTS only.")
            engine = "cloud"

    if engine == "gemini" and SLEEP_BETWEEN_SAMPLES_SEC <= 0 and SAMPLES_PER_POKEMON * len(ACTIVE_POKEMONS) > 3:
        print("[WARN] Gemini free tier is very rate-limited. Expect 429s after roughly 3 requests per minute.")
        print("       Use --sleep-between 20 or fewer samples / fewer Pokemon for a clean run.")

    for pokemon in ACTIVE_POKEMONS:
        folder = os.path.join(output_root, pokemon.lower())
        os.makedirs(folder, exist_ok=True)

        success = 0
        fail = 0

        print(f"\n{'='*55}")
        print(f"  {pokemon} — generating {SAMPLES_PER_POKEMON} samples ({engine} engine)")
        print(f"{'='*55}")

        for i in range(1, SAMPLES_PER_POKEMON + 1):
            audio = None

            # Decide which engine to use for this sample
            use_gemini = False
            if engine == "gemini":
                use_gemini = True
            elif engine == "cloud":
                use_gemini = False
            else:  # hybrid
                use_gemini = random.random() < 0.70

            debug_log(
                f"  [{pokemon}] sample {i}/{SAMPLES_PER_POKEMON} "
                f"engine={'gemini' if use_gemini else 'cloud'}"
            )

            # Generate raw audio
            try:
                if use_gemini:
                    audio = generate_one_gemini(gemini_client, genai_types, pokemon)
                    # Fallback to cloud if Gemini fails
                    if audio is None and engine == "hybrid":
                        audio = generate_one_cloud(pokemon)
                else:
                    audio = generate_one_cloud(pokemon)
            except Exception as e:
                print(f"    [ERR] Sample #{i}: {e}")

            if audio is None:
                if VERBOSE:
                    print(f"    [WARN] Sample #{i}: synthesis returned no audio", flush=True)
                fail += 1
                continue

            # Post-process
            try:
                audio = apply_random_effects(audio, pokemon)

                filename = f"{pokemon.lower()}_{i:03d}.wav"
                filepath = os.path.join(folder, filename)
                audio.export(filepath, format="wav")
                success += 1
            except Exception as e:
                print(f"    [ERR] Post-processing #{i}: {e}")
                fail += 1
                continue

            if VERBOSE:
                print(f"    [OK] Saved {filepath}", flush=True)

            if i % 25 == 0:
                print(f"  {pokemon}: {i}/{SAMPLES_PER_POKEMON}  "
                      f"({success} ok, {fail} failed)")

            if SLEEP_BETWEEN_SAMPLES_SEC > 0:
                time.sleep(SLEEP_BETWEEN_SAMPLES_SEC)
            elif i % 30 == 0:
                time.sleep(0.5)

        print(f"\n  >> {pokemon}: {success} saved, {fail} failed")

    print(f"\n{'='*55}")
    print(f"  DONE — output in ./{output_root}/")
    print(f"  Folders: {', '.join(p.lower() for p in ACTIVE_POKEMONS)}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pokemon Voice Data Generator v3 (Gemini TTS + Cloud TTS)"
    )
    parser.add_argument("-o", "--output", default="pokemon_voices",
                        help="Output directory (default: pokemon_voices)")
    parser.add_argument("-n", "--count", type=int, default=200,
                        help="Samples per Pokemon (default: 200)")
    parser.add_argument("--engine", choices=["gemini", "cloud", "hybrid"],
                        default="hybrid",
                        help="TTS engine: gemini, cloud, or hybrid (default: hybrid)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="HTTP timeout in seconds for Cloud TTS requests (default: 30)")
    parser.add_argument("--pokemon", default=None,
                        help="Comma-separated Pokemon subset, e.g. Pikachu or Bulbasaur,Pikachu")
    parser.add_argument("--sleep-between", type=float, default=0.0,
                        help="Seconds to sleep after each saved sample (useful for Gemini rate limits)")
    parser.add_argument("--gemini-retries", type=int, default=1,
                        help="Number of Gemini 429 retries using the returned retry delay (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample engine and HTTP failure details")
    args = parser.parse_args()

    ACTIVE_POKEMONS = parse_pokemon_selection(args.pokemon)
    VERBOSE = args.verbose
    REQUEST_TIMEOUT_SEC = args.timeout
    GEMINI_MAX_RETRIES = max(0, args.gemini_retries)
    SLEEP_BETWEEN_SAMPLES_SEC = max(0.0, args.sleep_between)
    SAMPLES_PER_POKEMON = args.count
    generate_all(output_root=args.output, engine=args.engine)
