# Scarlatti555
### Identify classical music by humming (which SoundHound, Shazam, and Google can't do)

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/Python-3.13.5-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Named for Domenico Scarlatti, who wrote 555 keyboard sonatas — most of which no existing music identification tool can recognise from a hum. Built for the classical music lover with thousands of motives and themes in their head, who can't always remember which piece each one belongs to.

**Live:** [Backend](https://web-production-b5bb4.up.railway.app) · [Frontend](https://scarlatti555.netlify.app/)

---

## The problem

Shazam and SoundHound identify music by fingerprinting exact recordings. Classical music has hundreds of recorded performances per piece and no single canonical audio signature to fingerprint against — and is thematically far more complex than the popular music these QbH tools were designed around. Scarlatti555 drops fingerprinting (hum → recording) in favor of contour matching (hum → score), matching the *melodic shape* of what someone hums against a structured representation of the piece, independent of tempo, key, or performance.

The largest part of this project is the data engineering, which has turned out to be about modelling musical memory: with 10 or so different instruments playing at once, how do you know which one will be "catchiest"? I'm currently building this corpus up from a small hand-verified corpus of extracted melodies toward an ML-extracted corpus spanning the >1 million pieces on MuseScore, using a siamese network trained on Romantic-era Lieder to model what a hummer actually remembers as "the tune" — see **Phidyle**, below. This works because 19th century German and French song (like Schubert or early Strauss) specifically features one instrument (the voice) which always carries the melody. Once successfully isolated, it can be used as a label against a MIDI file with all parts.

---

## How it works

```
User hums into browser
        ↓
MediaRecorder captures audio (React, browser-native, no install)
        ↓
torchcrepe extracts pitch (CNN-based F0 detection)
        ↓
Frame collapsing → discrete notes
        ↓
Hz → MIDI → semitone intervals (tempo and key invariant)
        ↓
Octave folding (normalizes large pitch jumps)
        ↓
Dynamic Time Warping match against corpus
        ↓
Ranked results returned as JSON
```

The core insight: **interval sequences, not absolute pitches**, are what make matching tempo- and key-invariant. Someone humming Fur Elise a third higher and twice as slow as another person should still match — intervals between consecutive notes stay the same; absolute pitch and timing don't.

---

## Features

- [x] Project scaffolding and architecture
- [x] Real-time pitch extraction from browser mic (torchcrepe)
- [x] Tempo-invariant interval sequence matching (Dynamic Time Warping)
- [x] MIDI corpus ingestion (Kunstderfuge)
- [x] FastAPI backend with `/identify` endpoint
- [x] Docker containerization
- [x] React web frontend with real-time audio recording
- [x] Deployed (Railway + Netlify)
- [ ] ML-extracted corpus at scale (Phidyle — in progress, see below)
- [ ] PostgreSQL + pgvector similarity search
- [ ] Embedding-based matching (siamese network) for scale beyond brute-force DTW
- [ ] OMR pipeline for scanned PDF scores (Audiveris) — expand coverage beyond existing MIDI corpora
- [ ] React Native mobile app

Possible features:
- [ ] YouTube link for matched piece
- [ ] Spotify/Apple Music compatibility — prompt to open recording in given streaming app
- [ ] Music theory primer in docs (what is an interval, why DTW, etc.) for non-musician readers

---

## Stack

| Layer | Technology |
|---|---|
| Pitch extraction | torchcrepe (neural pitch tracker, CNN-based F0 detection) |
| Backend | Python 3.13, FastAPI, Docker |
| Matching | Dynamic Time Warping (`fastdtw`) |
| MIDI / score processing | `music21`, `pretty_midi` |
| Corpus source (current) | Kunstderfuge MIDI, OpenScore Lieder Corpus (MusicXML) |
| Web frontend | React, MediaRecorder API |
| Deployment | Railway (backend), Netlify (frontend) |
| Planned | PostgreSQL + pgvector, FAISS, Audiveris (OMR), React Native |

---

## Why this is hard

**Pitch tracking from humming is noisy.** Human voices waver, slide between notes, and don't produce clean discrete pitches. torchcrepe — a neural pitch tracker — handles this substantially better than traditional FFT-based approaches.

**Rhythm is elastic.** People hum the same theme at different tempos with imprecise note durations. Matching uses relative interval sequences rather than absolute pitch/rhythm, with Dynamic Time Warping for tempo-invariant alignment.

**Database coverage is enormous and uneven.** Classical music has a vast thematic vocabulary — Scarlatti alone wrote 555 keyboard sonatas. Existing MIDI corpora cover only a fraction of it. Closing this gap is the motivation for both the Phidyle ML-extraction work and the planned OMR pipeline for scanned scores.

**Idiomatic keyboard writing doesn't always reduce to one singable line.** Some themes (particularly Baroque keyboard music) resist a single clean melodic reduction. Where needed, the database stores multiple hummable representations per piece.

**Extracting the *right* melody — not just *a* sequence of notes — turned out to be the hardest part of the whole project.** See the corpus iteration history below.

---

## The corpus problem — and why it was the hard part

Getting a pipeline to extract *a* sequence of notes from a score is comparatively easy. Getting it to extract the *right* sequence — the one a human would actually recognize and hum back — is not, and this took several real iterations to get right.

### Iteration 1: Skyline on raw MIDI
Took the highest pitch at each timestep across the full MIDI file. Failed on polyphonic pieces (Vivaldi's *Spring*) where the highest note at a given moment is sometimes a harmony note, not the melody, and on piano pieces where the left hand occasionally plays higher than the melody line, causing "left-hand bleed."

### Iteration 2: Basic Pitch (audio-based extraction)
Tried Spotify's open-source audio-to-MIDI model as an alternative to working from raw MIDI. Hit real dependency conflicts on Python 3.13 (TensorFlow version incompatibility) — resolved by forcing the ONNX backend instead. Extraction quality without additional filtering was poor: 1,333 raw notes for a 3-minute piece, including clear bass and harmony bleed. Adding a pitch-range filter and a skyline pass on top improved results substantially but was still a heuristic stacked on a heuristic.

### Iteration 3: Demucs (source separation)
Attempted to isolate the melody stem from a full mix before running pitch extraction — the same general strategy professional audio separation tools use. Blocked by a `torchcodec` dependency issue in the installed Demucs version on Windows. Documented as a viable direction if revisited with a cleaner environment.

### Iteration 4: music21 voice separation — the working solution
Switched from treating MIDI as a flat list of notes to using `music21`, which understands score *structure* — parts, voices, instruments — rather than just timestamps and pitches. For piano pieces with a single part containing multiple simultaneous notes (no separate voice tracks), applied skyline *within* `music21`'s structured representation with a minimum-duration filter to exclude ornaments and very short notes. This correctly reproduced Fur Elise's famous opening (`E5, D#5, E5, D#5, E5...`) and worked acceptably across 5 of 6 test pieces.

### The matching algorithm also needed iteration
Initial sliding-window DTW (searching every position within a long stored piece, not just the opening) was theoretically the right idea — themes aren't always hummed from the very start — but produced compressed, indistinguishable scores (all matches landing in a narrow 36–47% band) because the window size was too small relative to query length to capture distinctive pattern. Simplified to direct full-sequence DTW with trimmed, ~20-interval corpus entries, which produced clearer separation between correct and incorrect matches.

---

## Phidyle — perceptually-grounded melody extraction (active research direction)

This is the most technically substantive part of the project and the part most worth discussing in depth.

**The problem with melody extraction as commonly practiced:** tools like Melodia and Goto's PreFEst extract melody by computing *acoustic salience* — which pitch, at each moment, is most harmonically and perceptually prominent given loudness-weighting tuned to human hearing. This is a good engineering proxy for melody, but it isn't the same thing as *what a human would actually hum after hearing the piece*. Salience is a property of the acoustic signal; "the tune" is a property of human auditory perception and memory — closer to what Bregman's *Auditory Scene Analysis* describes as stream segregation, which combines salience with timbral grouping, continuity, and learned musical expectation.

**The proposed approach:** rather than inferring melody indirectly from acoustic heuristics, train a model directly on cases where melody is *unambiguous by genre convention* — vocal classical music (art songs/Lieder), where the sung line *is* the melody by definition, and the surrounding piano accompaniment is structurally separate. If such a model generalizes, it would extract melody based on something closer to *what stands apart perceptually*, rather than *what is loudest* — and could eventually run across the much larger, unlabeled MuseScore corpus to give Scarlatti real thematic coverage, including obscure repertoire (rare Scarlatti sonatas, Field nocturnes, etc.) absent from existing MIDI databases.

**Data source:** the [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder) (Gotham & Jonas, 2022) — 1,462 crowdsourced, professionally-engraved MusicXML art song scores, CC0 licensed. Unlike hand-sequenced MIDI files (which frequently carry no part-naming metadata at all — confirmed directly during this project, see below), MusicXML produced via notation software requires the engraver to explicitly label each staff, since the format exists to represent human-readable scores, not just playback instructions. This makes part identification ("Singstimme" / voice vs. "Pianoforte" / piano) a reliable string-match problem rather than a per-file listening exercise.

**Validated so far:**
- Confirmed by direct inspection that hand-downloaded hobbyist MIDI files carry no usable instrument/part metadata (`music21`'s `getInstrument()` returns empty across the board) — motivating the move to a properly-engraved source
- Confirmed voice/piano separation is correct by ear on Schubert's *Gute Nacht* (Winterreise No. 1) — isolated vocal line is clean, complete, and recognizable
- Ran the extraction pipeline across the full corpus: **1,401 of 1,462 pieces (95.8%) yield a confidently identified, sufficiently long voice part** suitable as a training label, via part-name keyword matching (`singstimme`, `voice`, `voce`, `gesang`, plus named voice types)

**Open question, not yet resolved:** whether a model trained to find the voice line in vocal pieces generalizes to melody extraction in *purely instrumental* music, where no voice-specific timbral cue exists to anchor the prediction. This is the key unresolved risk — the model may learn "what a human voice sounds/behaves like" specifically, rather than the more general "what stands apart as the melody," and those are not guaranteed to be the same thing. Testing this transfer (training on Lieder, evaluating on instrumental pieces like Fur Elise or Vivaldi's *Spring*) is the next concrete step.

**Matching at scale — a separate problem from extraction:** brute-force DTW against every corpus entry is fine at the current scale (dozens of pieces) but does not scale to a corpus of thousands, let alone the >1 million pieces on MuseScore. The standard solution used in production-scale audio matching systems is a two-stage approach: a learned embedding model (trained via a siamese network) maps any interval sequence to a fixed-length vector, approximate nearest-neighbor search (e.g. FAISS) finds the top-k candidates near-instantly even at large scale, and DTW is reserved as an accurate confirmation step on that short candidate list rather than the first-pass filter. This is deliberately scoped as future work — premature to build before the corpus is large enough to need it.

---

## What's deployed right now vs. what's research

| | Status |
|---|---|
| Audio capture, pitch extraction, basic matching | **Deployed, working** |
| music21-based corpus (6 verified pieces) | **Deployed, working** |
| Docker containerization | **Done** |
| Phidyle data validation (OpenScore Lieder) | **Validated, not yet trained** |
| Phidyle model training | **Not started** |
| Generalization testing (vocal → instrumental) | **Not started** |
| Embedding + FAISS matching at scale | **Not started — correctly deferred** |
| OMR pipeline (scanned scores) | **Not started** |

---

## Corpus sources

- [Kunstderfuge](https://www.kunstderfuge.com/) — curated classical MIDI files
- [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder) — 1,462 engraved MusicXML art songs, CC0, used for Phidyle's voice/piano training pairs
- [IMSLP](https://imslp.org/) — sheet music, planned OMR source for expanding instrumental coverage
- [Mutopia Project](https://www.mutopiaproject.org/) — community-encoded scores

---

## Getting started

```bash
git clone https://github.com/echakrabarti/Scarlatti555
cd scarlatti555
python -m venv venv
source venv/bin/activate   # or venv/Scripts/activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm start
```

---

## Dev log

→ [Read the dev log](devlog.md)

## References

- Bregman, A. (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound.*
- Salamon, J. & Gómez, E. (2012). *Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics.*
- Goto, M. (2004). *A real-time music-scene-description system: predominant-F0 estimation for detecting melody and bass lines in real-world audio signals.*
- Gotham, M. & Jonas, P. (2022). *The OpenScore Lieder Corpus.*

---

## Project status

Active development.

## License

MIT — see [LICENSE](LICENSE) for details.