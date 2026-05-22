# Scarlatti555 🎵
### Identify classical music by humming (which SoundHound, Shazam, and Google can't do)

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Named for Domenico Scarlatti, who wrote 555 keyboard sonatas — most of which no existing music identification tool can recognise from a hum. This project is built for the classical music lover with thousands of motives and themes in their head--and who can't always remember the piece associated with each one!

---

## The problem

Shazam and SoundHound identify music by fingerprinting exact recordings. This makes them for identifying everything but the most popular classical music. Scarlatti555 chucks fingerprinting (hum->recording) in favour of contour matching (hum-> score), an approach that inherently advantageous for identifying a classical piece.

---

## How it works

```
Your hum → pitch extraction (CREPE) → interval sequence → vector similarity search → match
```

Rather than fingerprinting audio, Scarlatti555 extracts the **melodic contour** of your hum — the sequence of relative intervals between notes — and matches it against a curated database of classical themes. Because matching is done on intervals rather than raw pitch, it is transposition-invariant: you can hum in any key and still get the right result.

The database is built from MIDI corpora (Kunstderfuge, IMSLP) supplemented by an Optical Music Recognition (OMR) pipeline that converts scanned scores to MIDI, dramatically expanding coverage of less-represented repertoire — including obscure Scarlatti sonatas and Romantic-era works like the Field nocturnes that are absent from most MIDI databases.

---

## Features

- [x] Project scaffolding and architecture
- [ ] Real-time pitch extraction from browser mic (TORCHCREPE)
- [ ] Tempo-invariant interval sequence matching
- [ ] MIDI corpus ingestion (Kunstderfuge + IMSLP)
- [ ] OMR pipeline for PDF scores (Audiveris)
- [ ] FastAPI backend with `/identify` endpoint
- [ ] PostgreSQL + pgvector similarity search
- [ ] Streamlit prototype UI
- [ ] React web frontend
- [ ] React Native mobile app (iOS + Android)

Possible features:
- [ ] YouTube link for matched piece
- [ ] Spotify/Apple Music compatibility: prompt to open recording in given streaming app?
- [ ] Include everything you need to know about music to understand how this works in docs (e.g., what is an interval)
---

## Stack

| Layer | Technology |
|---|---|
| Pitch extraction | torchcrepe (neural pitch tracker) |
| Backend | Python 3.13.5, FastAPI |
| Similarity search | pgvector (PostgreSQL extension) |
| MIDI processing | pretty_midi, music21 |
| OMR | Audiveris |
| Prototype UI | Streamlit |
| Web frontend | React + Web Audio API |
| Mobile | React Native |
| Deployment | Railway (backend), Vercel (frontend) |

---

## Why this is hard

**Pitch tracking from humming** is noisy. Human voices waver, slide between notes, and don't produce clean discrete pitches. CREPE — a purpose-built neural network for monophonic pitch tracking — handles this significantly better than traditional FFT approaches.

**Rhythm elasticity.** People hum the same theme at wildly different tempos and with imprecise note durations. Matching uses relative interval sequences rather than absolute rhythms, with dynamic time warping for alignment.

**Database coverage.** Classical music has an enormous thematic vocabulary — Scarlatti alone wrote 555 keyboard sonatas. Achieving useful coverage requires both existing MIDI corpora and automated conversion of scanned scores through OMR.

**Idiomatic keyboard writing.** Some classical themes (particularly Baroque keyboard music) aren't cleanly reducible to a single singable melody. The database stores multiple hummable representations per piece where needed.

---

## Corpus sources

- [Kunstderfuge](https://www.kunstderfuge.com/) — ~19,000 curated classical MIDI files
- [IMSLP](https://imslp.org/) — sheet music and MIDI, supplemented by OMR pipeline
- [Mutopia Project](https://www.mutopiaproject.org/) — community-encoded scores

---

## Dev log

→ [Read the dev log](devlog.md)

---

## Getting started

```bash
git clone https://github.com/yourusername/scarlatti555
cd scarlatti555
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

*Full setup instructions will be added as the project develops. Eventually this will be a React Native app.*

---

## Project status

Active development. See the [issues](../../issues) for current work in progress.

---

## License

MIT — see [LICENSE](LICENSE) for details.
