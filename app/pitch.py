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
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    audio = torch.tensor(audio).unsqueeze(0)
    return audio, sr

def extract_pitch(audio: torch.Tensor, sr: int):
    pitch, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length=160,
        fmin=50.0,
        fmax=1000.0,
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

def hz_to_midi(pitch_hz: np.ndarray):
    pitch_hz = np.clip(pitch_hz, 1e-6, None)
    midi_notes = 12*np.log2(pitch_hz/440.0) + 69
    return np.round(midi_notes).astype(int)

def to_intervals(midi_notes: np.ndarray):
    if len(midi_notes) < 2:
        return np.array([])
    return np.diff(midi_notes)

def process(file_path: str):
    audio, sr = load_audio(file_path) # load audio in
    pitch, periodicity = extract_pitch(audio, sr) # save pitches in HZ with confidences
    voiced_pitch = filter_voiced(pitch, periodicity) # remove low confidence pitches
    midi_notes = hz_to_midi(voiced_pitch) # converts cleaned pitchesfrom HZ to MIDI
    intervals = to_intervals(midi_notes) # represents MIDI notes as intervals
    return intervals