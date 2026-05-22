# Dev log - Scarlatti555
Newest entries at the top.

---
## 2026-05-22 - [Day 1: Project architecture, pitch pipeline (pitch.py)]
- Set up virtual environment
- Fine-grained outline of stages and stack
    - 8 week timeline: (1) foundations, (2)matching engine, (3) React web app, (4) React Native mobile
    - Stack: Python/FastAPI backend, torchcrepe for pitch extraction, pgvector similarity search, Streamlit (Pythom) prototype before React
- Project architecture:
    - Hum -> main -> pitch (humming is now intervals) -> main -> match (intervals matched to database intervals) -> main -> result (React frontend abstracted for now)
- pitch.py
    - CREPE is incompatible with recent Python versions (no pkg_resources module)
        - Explored options: penn and torchcrepe. Chose torchcrepe: penn will be slow w/out GPU, but torchcrepe is just a pytorch version of CREPE, which the project is already build around
    - learned audio engineering math to get optimized pitch sampling
    - Wrote pitch.py
        - load_audio (read, resample, convert),
        - extract_pitch (torchcrepe NN to retrieve pitches as tensor),
        - filter_voiced (remove dead air),
        - hz_to_midi (convert HZ representation to db's MIDI representation,
        - to_intervals (notes -> intervals),
        - process (one fxn to orchestrate full pipeline)

