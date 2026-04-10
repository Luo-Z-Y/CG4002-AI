"""
pokemon_tts_generate.py
────────────────────────
Generates many synthetic audio samples of 6 Pokémon names using
Microsoft Edge TTS (free, no API key needed, 400+ voices).

Each sample = one voice × one rate × one pitch variant, so you get
large, controlled variation — good for CNN training augmentation.

Requirements:
    pip install edge-tts asyncio

Usage:
    python pokemon_tts_generate.py

Output structure:
    tts_audio/
        bulbasaur/
            bulbasaur_en-US-AriaNeural_r-10_p+0.mp3
            bulbasaur_en-US-GuyNeural_r+0_p-5.mp3
            ...
        charizard/
            ...
"""

import asyncio
import itertools
import json
import os
from pathlib import Path

import edge_tts
from edge_tts.exceptions import NoAudioReceived

# ── CONFIG ────────────────────────────────────────────────────────────────────

POKEMON_NAMES = [
    "bulbasaur",
    "charizard",
    "greninja",
    "lugia",
    "mewtwo",
    "pikachu",
]

# English voices (mix of genders / accents for variety without going too wild)
# Full list:  edge-tts --list-voices  (run in terminal)
VOICES = [
    # US English
    "en-US-AriaNeural",        # Female, friendly
    "en-US-GuyNeural",         # Male, neutral
    "en-US-JennyNeural",       # Female, conversational
    "en-US-DavisNeural",       # Male, casual
    "en-US-AmberNeural",       # Female, warm
    "en-US-AnaNeural",         # Female (child-like)
    "en-US-BrandonNeural",     # Male
    "en-US-ChristopherNeural", # Male, professional
    "en-US-EricNeural",        # Male
    "en-US-JacobNeural",       # Male
    "en-US-MichelleNeural",    # Female
    "en-US-MonicaNeural",      # Female
    "en-US-RogerNeural",       # Male
    "en-US-SteffanNeural",     # Male, narrative
    # UK English
    "en-GB-LibbyNeural",       # Female
    "en-GB-RyanNeural",        # Male
    "en-GB-SoniaNeural",       # Female
    "en-GB-ThomasNeural",      # Male
    # Australian English
    "en-AU-NatashaNeural",     # Female
    "en-AU-WilliamNeural",     # Male
    # Canadian English
    "en-CA-ClaraNeural",       # Female
    "en-CA-LiamNeural",        # Male
    # Indian English
    "en-IN-NeerjaNeural",      # Female
    "en-IN-PrabhatNeural",     # Male
]

# Speaking rate adjustments (+/- %)  — keep range small for CNN stability
RATES = ["-10%", "-5%", "+0%", "+5%", "+10%"]

# Pitch adjustments (Hz)  — subtle shifts only
PITCHES = ["-5Hz", "+0Hz", "+5Hz"]

OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "data" / "voice" / "tts_audio")
FAILURE_LOG = "tts_failures.jsonl"
MAX_RETRIES = 3
BATCH_SIZE = 4

# ── GENERATE ──────────────────────────────────────────────────────────────────

async def generate_sample(
    name: str,
    voice: str,
    rate: str,
    pitch: str,
    out_dir: str,
) -> str:
    """Synthesise one audio sample and save it."""
    # Build a safe filename
    rate_tag = rate.replace("+", "p").replace("-", "m").replace("%", "pct")
    pitch_tag = pitch.replace("+", "p").replace("-", "m").replace("Hz", "hz")
    filename = f"{name}_{voice}_r{rate_tag}_pi{pitch_tag}.mp3"
    dest = os.path.join(out_dir, filename)

    if os.path.exists(dest):
        return "skipped"  # Skip if already generated (resume-friendly)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(
                text=name,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(dest)
            return "saved"
        except NoAudioReceived:
            if attempt == MAX_RETRIES:
                break
            await asyncio.sleep(0.75 * attempt)

    failure = {
        "name": name,
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "dest": dest,
        "error": "NoAudioReceived",
    }
    with Path(FAILURE_LOG).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure) + "\n")
    return "failed"


async def generate_all() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Path(FAILURE_LOG).unlink(missing_ok=True)

    total = 0
    total_failed = 0
    for name in POKEMON_NAMES:
        name_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(name_dir, exist_ok=True)

        combos = list(itertools.product(VOICES, RATES, PITCHES))
        print(f"\n── {name.upper()} — generating {len(combos)} samples ──")

        name_failed = 0
        for i in range(0, len(combos), BATCH_SIZE):
            batch = combos[i : i + BATCH_SIZE]
            tasks = [
                generate_sample(name, voice, rate, pitch, name_dir)
                for voice, rate, pitch in batch
            ]
            results = await asyncio.gather(*tasks)
            name_failed += sum(result == "failed" for result in results)
            done = min(i + BATCH_SIZE, len(combos))
            print(f"  {done}/{len(combos)} done... failures: {name_failed}", end="\r")

        count = len(os.listdir(name_dir))
        total += count
        total_failed += name_failed
        print(f"  ✓ {count} files saved to {name_dir}/ | failures: {name_failed}          ")

    print(f"\n✓ All done. {total} total samples in '{OUTPUT_DIR}/'")
    if total_failed:
        print(f"Failed requests logged to {FAILURE_LOG} ({total_failed} total failures)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # Quick dependency check
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("edge-tts not installed. Run:\n  pip install edge-tts")
        return

    expected = len(POKEMON_NAMES) * len(VOICES) * len(RATES) * len(PITCHES)
    print(f"Will generate up to {expected} samples")
    print(f"  {len(POKEMON_NAMES)} Pokémon × {len(VOICES)} voices × "
          f"{len(RATES)} rates × {len(PITCHES)} pitches\n")

    asyncio.run(generate_all())


if __name__ == "__main__":
    main()
