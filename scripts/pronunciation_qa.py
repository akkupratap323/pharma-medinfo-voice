"""Automated drug-name pronunciation QA — TTS -> STT round trip.

SynthioLabs checks pronunciation manually against spectrograms. This
automates the check: if our TTS speaks "dupilumab" clearly enough that
Deepgram transcribes it back, it passes; if the transcript comes back as
something else, it fails and gets a phonetic alias in the pre-TTS map.

Usage:
  python scripts/pronunciation_qa.py            # run all terms
  python scripts/pronunciation_qa.py --term dupilumab

Env: ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, DEEPGRAM_API_KEY
Output: data/pronunciation_report.md (+ .json), wavs in data/pron_audio/
"""

import argparse
import json
import os
import pathlib

import httpx
from rapidfuzz import fuzz

TERMS = [
    "Dupixent", "dupilumab", "atopic dermatitis", "eosinophilic esophagitis",
    "chronic rhinosinusitis", "nasal polyps", "prurigo nodularis",
    "conjunctivitis", "keratitis", "blepharitis", "arthralgia",
    "hypersensitivity", "anaphylaxis", "subcutaneous", "interleukin",
    "corticosteroids", "eosinophilia", "vasculitis", "helminth",
    "immunogenicity", "pre-filled syringe", "autoinjector",
]

CARRIERS = [
    "The prescribing information for {term} covers this in detail.",
    "Patients taking {term} should discuss this with their doctor.",
]

# Pre-TTS phonetic aliases for terms that fail the round trip.
# Wire into TextFilterProcessor.clean_text_for_speech AFTER a failing run.
ALIASES: dict[str, str] = {
    # "dupilumab": "doo-PIL-you-mab",   # example — add only proven failures
}

AUDIO_DIR = pathlib.Path("data/pron_audio")
PASS_THRESHOLD = 85  # rapidfuzz partial ratio


def tts(text: str, out_path: pathlib.Path) -> None:
    voice = os.environ["ELEVENLABS_VOICE_ID"]
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json={"text": text, "model_id": "eleven_turbo_v2_5"},
        timeout=60,
    )
    r.raise_for_status()
    out_path.write_bytes(r.content)


def stt(audio_path: pathlib.Path) -> str:
    r = httpx.post(
        "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true",
        headers={
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": "audio/mpeg",
        },
        content=audio_path.read_bytes(),
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


def check_term(term: str) -> dict:
    spoken = ALIASES.get(term.lower(), term)
    scores, transcripts = [], []
    for i, carrier in enumerate(CARRIERS):
        sentence = carrier.format(term=spoken)
        wav = AUDIO_DIR / f"{term.replace(' ', '_')}_{i}.mp3"
        tts(sentence, wav)
        transcript = stt(wav)
        transcripts.append(transcript)
        scores.append(fuzz.partial_ratio(term.lower(), transcript.lower()))
    best = max(scores)
    return {
        "term": term,
        "aliased": term.lower() in ALIASES,
        "score": best,
        "pass": best >= PASS_THRESHOLD,
        "heard": transcripts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--term")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    terms = [args.term] if args.term else TERMS

    results = []
    for t in terms:
        res = check_term(t)
        flag = "PASS" if res["pass"] else "FAIL"
        print(f"{flag}  {t:32s} score={res['score']}  heard={res['heard'][0][:60]!r}")
        results.append(res)

    passed = sum(r["pass"] for r in results)
    lines = [
        "# Pronunciation QA Report",
        "",
        f"TTS: ElevenLabs turbo v2.5 | STT: Deepgram nova-3 | "
        f"threshold: partial-ratio >= {PASS_THRESHOLD}",
        "",
        f"**{passed}/{len(results)} terms pass.**",
        "",
        "| term | score | pass | aliased | heard (first carrier) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['term']} | {r['score']} | {'yes' if r['pass'] else 'NO'} "
            f"| {'yes' if r['aliased'] else ''} | {r['heard'][0][:60]} |"
        )
    lines += [
        "",
        "Failures: add a phonetic alias to ALIASES, wire the alias map into "
        "TextFilterProcessor.clean_text_for_speech, and re-run until green.",
    ]
    pathlib.Path("data/pronunciation_report.md").write_text("\n".join(lines))
    pathlib.Path("data/pronunciation_report.json").write_text(json.dumps(results, indent=2))
    print(f"\n{passed}/{len(results)} pass -> data/pronunciation_report.md")


if __name__ == "__main__":
    main()
