"""
verify_gute_nacht.py — export the isolated voice part as a 
playable MIDI to verify separation quality by ear
"""

import os
from music21 import converter

INPUT_PATH = "data/openscore_lieder/scores/Schubert,_Franz/Winterreise,_D.911/1_Gute_Nacht/lc5015378.mxl"
OUTPUT_DIR = "data/PhidyleIsolVocals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

score = converter.parse(INPUT_PATH)

print(f"Number of parts: {len(score.parts)}")
for i, part in enumerate(score.parts):
    print(f"  Part {i}: partName={part.partName!r}")

# part 0 confirmed as 'Singstimme' (voice) from earlier inspection
voice_part = score.parts[0]
notes = voice_part.flatten().notes

# print pitch range for sanity check
pitches = [n.pitch.midi for n in notes if n.isNote]
print(f"\nVoice part: {len(notes)} notes")
print(f"Pitch range: {min(pitches)} - {max(pitches)} (MIDI numbers)")
print(f"Pitch range: {converter.parse(INPUT_PATH).parts[0].lowestOffset}")

# export voice part alone
voice_output = os.path.join(OUTPUT_DIR, "gute_nacht_voice_only.mid")
voice_part.write('midi', fp=voice_output)
print(f"\nSaved: {voice_output}")

# export full piece for comparison
full_output = os.path.join(OUTPUT_DIR, "gute_nacht_FULL.mid")
score.write('midi', fp=full_output)
print(f"Saved: {full_output}")

# export piano parts separately too, for completeness
if len(score.parts) >= 3:
    score.parts[1].write('midi', fp=os.path.join(OUTPUT_DIR, "gute_nacht_piano_RH.mid"))
    score.parts[2].write('midi', fp=os.path.join(OUTPUT_DIR, "gute_nacht_piano_LH.mid"))
    print(f"Saved: gute_nacht_piano_RH.mid and gute_nacht_piano_LH.mid")

print("\nDone. Listen to gute_nacht_voice_only.mid and compare against gute_nacht_FULL.mid")