import torch
import torchcrepe
import librosa
import numpy as np

VOICED_THRESHOLD = 0.5 # more than 50% certainty this is a pitch
SAMPLE_RATE = 16000 # 16000 samples/second
HOP_LENGTH = 160 # window size for sampling 
FMIN = 50.0 # min freq (< lowest note of bass singer)
FMAX = 1000.0 # max freq (> highest note of soprano)

def load_audio(file_path: str):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)
    return audio, sr

def extract_pitch(audio: torch.Tensor, sr: int):
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        model = 'tiny',
        return_periodicity = True,
        device = 'cpu'
    )
    pitch = pitch.squeeze().numpy()
    periodicity = periodicity.squeeze().numpy()
    return pitch, periodicity

def filter_voiced(pitch: np.ndarray, periodicity: np.ndarray, threshold: float = 0.5):
    voiced = periodicity > threshold
    return pitch[voiced]

def frames_to_notes(pitch_frames: np.ndarray, hop_length: int, sr: int) -> np.ndarray:
    if len(pitch_frames) == 0:
        return np.array([])
    notes = []
    current_note_frames = [pitch_frames[0]]
    for i in range(1, len(pitch_frames)):
        current_frame = pitch_frames[i]
        current_median = np.median(current_note_frames)
        midi_current = 12 * np.log2(max(current_frame, 1e-6) / 440.0) + 69
        midi_median = 12 * np.log2(max(current_median, 1e-6) / 440.0) + 69
        if abs(midi_current - midi_median) <= 1.0:
            current_note_frames.append(current_frame)
        else:
            notes.append(np.median(current_note_frames))
            current_note_frames = [current_frame]
    notes.append(np.median(current_note_frames))
    return np.array(notes)

def hz_to_midi(pitch_hz: np.ndarray):
    pitch_hz = np.clip(pitch_hz, 1e-6, None)
    midi_notes = 12*np.log2(pitch_hz/440.0) + 69
    return np.round(midi_notes).astype(int)

def to_intervals(midi_notes: np.ndarray):
    if len(midi_notes) < 2:
        return np.array([])
    return np.diff(midi_notes)

def process(file_path: str):
    audio, sr = load_audio(file_path)
    pitch, periodicity = extract_pitch(audio, sr)
    voiced_pitch = filter_voiced(pitch, periodicity)
    print(f"voiced frames: {len(voiced_pitch)}")
    note_events = frames_to_notes(voiced_pitch, HOP_LENGTH, sr)
    print(f"note events: {len(note_events)}")
    midi_notes = hz_to_midi(note_events)
    print(f"midi notes: {midi_notes[:20]}")
    intervals = to_intervals(midi_notes)
    print(f"intervals: {len(intervals)}, first 20: {intervals[:20]}")
    return intervals