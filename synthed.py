import curses
import numpy as np
import sounddevice as sd
import random
import time
from queue import Queue, Empty
from typing import Any, Callable
from dataclasses import dataclass, field

# --- CONFIGURATION ---
SAMPLE_RATE: int = 44100
BLOCK_SIZE: int = 1024
VIS_FPS: int = 30

# Synth Parameters
SMOOTHING_FACTOR_ATTACK: float = 0.08
SMOOTHING_FACTOR_RELEASE: float = 0.04
LFO_FREQ: float = 5.0
LFO_AMP: float = 0.15
OSC_AMP: float = 0.2

# Staggered onset (milliseconds)
STAGGER_MIN_MS: float = 10.0
STAGGER_MAX_MS: float = 40.0

# Bass emphasis factor (1.0 = no boost)
BASS_BOOST: float = 1.35

# Chord transition probability for preferring "primary" transition
PRIMARY_TRANSITION_WEIGHT: float = 0.3

# Per-voice amplitude variation range (±)
VOICE_VARIATION: float = 0.1

# Frequency jitter for stagger ordering (Hz)
STAGGER_FREQ_JITTER: float = 50.0


# --- DATA STRUCTURES ---


@dataclass
class RhythmStep:
    """A single step in a rhythm pattern."""

    duration: float  # Multiplier for base tempo
    velocity: float  # Amplitude multiplier (0.0 - 1.0)


@dataclass
class RhythmPattern:
    """A named rhythm pattern with duration and velocity steps."""

    name: str
    steps: list[RhythmStep]


@dataclass
class Scale:
    """A musical scale with frequencies, chords, and transitions."""

    name: str
    frequencies: list[float]
    chords: dict[str, list[int]]
    transitions: dict[str, list[str]]


@dataclass
class Voice:
    """A single oscillator voice with envelope tracking."""

    freq: float
    phase: float = 0.0
    current_amp: float = 0.0
    target_amp: float = 0.0
    onset_delay_samples: int = 0
    pending_target_amp: float = 0.0


# --- RHYTHM PATTERNS ---

RHYTHM_PATTERNS: list[RhythmPattern] = [
    RhythmPattern(
        "Steady",
        [RhythmStep(1.0, 0.9), RhythmStep(1.0, 0.7), RhythmStep(1.0, 0.8), RhythmStep(1.0, 0.6)],
    ),
    RhythmPattern(
        "Swing",
        [RhythmStep(1.5, 1.0), RhythmStep(0.5, 0.5), RhythmStep(1.0, 0.8), RhythmStep(1.0, 0.7)],
    ),
    RhythmPattern(
        "Syncopated",
        [RhythmStep(1.0, 0.9), RhythmStep(0.5, 0.6), RhythmStep(0.5, 0.7), RhythmStep(1.0, 0.85), RhythmStep(1.0, 0.65)],
    ),
    RhythmPattern(
        "Contemplative",
        [RhythmStep(2.0, 1.0), RhythmStep(1.0, 0.6), RhythmStep(1.0, 0.75)],
    ),
    RhythmPattern(
        "Chaotic",
        [RhythmStep(0.25, 0.9), RhythmStep(0.25, 0.6), RhythmStep(0.5, 0.8), RhythmStep(0.25, 0.7), RhythmStep(1.5, 0.9)],
    ),
    RhythmPattern(
        "Driving",
        [RhythmStep(0.5, 1.0), RhythmStep(0.5, 0.6), RhythmStep(0.5, 0.9), RhythmStep(0.5, 0.7)],
    ),
    RhythmPattern(
        "Waltz",
        [RhythmStep(1.0, 1.0), RhythmStep(1.0, 0.6), RhythmStep(1.0, 0.5)],
    ),
]


# --- SCALES ---

SCALES: list[Scale] = [
    # ═══════════════════════════════════════════════════════════════════════════
    # WARM & FAMILIAR - Entry points, comfortable, "home" feeling
    # ═══════════════════════════════════════════════════════════════════════════
    Scale(
        name="A Minor Pentatonic",
        # A C D E G A (classic minor pentatonic - warm, bluesy)
        frequencies=[110.0, 130.8, 146.8, 164.8, 196.0, 220.0],
        chords={
            # Stable home chords
            "Am": [0, 2, 4],              # A D G - open, stable
            "Am(add9)": [0, 1, 4],        # A C G - lush minor
            # Movement chords
            "C6": [1, 2, 4],              # C D G - bright, uplifting
            "Em7": [3, 4, 5],             # E G A - gentle tension
            # Tension/color
            "Gsus2": [4, 5, 0],           # G A A(oct) - suspended, airy
            "Dsus4": [2, 3, 4],           # D E G - gentle pull
            # Atmospheric
            "Drift": [0, 4, 5],           # A G A(oct) - open fifth, spacious
        },
        transitions={
            "Am": ["C6", "Em7", "Am(add9)", "Drift"],
            "Am(add9)": ["Am", "Dsus4", "Gsus2"],
            "C6": ["Am", "Em7", "Dsus4"],
            "Em7": ["Am", "C6", "Gsus2"],
            "Gsus2": ["Am", "Em7", "Drift"],
            "Dsus4": ["Am", "C6", "Em7"],
            "Drift": ["Am", "Am(add9)", "Gsus2"],
        },
    ),
    Scale(
        name="D Dorian",
        # D E F G A B D (dorian mode - jazzy, sophisticated minor)
        frequencies=[146.8, 164.8, 174.6, 196.0, 220.0, 246.9],
        chords={
            # Home base
            "Dm7": [0, 2, 4, 5],          # D F A B - rich minor 7th
            "Dm9": [0, 1, 4, 5],          # D E A B - lush, jazzy
            # Movement
            "G7sus4": [3, 4, 0],          # G A D - dominant feel
            "Am7": [4, 5, 0, 1],          # A B D E - ii chord
            "Em7b5": [1, 2, 4],           # E F A - half-dim color
            # Resolution & color
            "Cmaj7": [5, 0, 1, 3],        # B D E G - borrowed brightness
            "Floating": [0, 3, 5],        # D G B - open, dreamy
        },
        transitions={
            "Dm7": ["G7sus4", "Am7", "Floating", "Dm9"],
            "Dm9": ["Dm7", "Cmaj7", "Em7b5"],
            "G7sus4": ["Dm7", "Cmaj7", "Am7"],
            "Am7": ["Dm7", "G7sus4", "Em7b5"],
            "Em7b5": ["Dm7", "Am7", "Floating"],
            "Cmaj7": ["Dm7", "G7sus4", "Dm9"],
            "Floating": ["Dm7", "Am7", "Dm9"],
        },
    ),
    Scale(
        name="G Major Pentatonic",
        # G A B D E G (bright, optimistic, pastoral)
        frequencies=[196.0, 220.0, 246.9, 293.7, 329.6, 392.0],
        chords={
            # Bright home chords
            "G6": [0, 2, 4],              # G B E - sparkling major
            "Gmaj7": [0, 2, 3, 5],        # G B D G - lush resolution
            # Movement
            "Em7": [4, 5, 0, 2],          # E G G B - relative minor
            "D6": [3, 4, 1],              # D E A - dominant character
            "Am7": [1, 5, 0, 4],          # A G G E - ii chord feel
            # Color & space
            "Bm7": [2, 3, 4, 1],          # B D E A - iii chord
            "Sunlight": [0, 3, 5],        # G D G - open, radiant
        },
        transitions={
            "G6": ["Em7", "D6", "Gmaj7", "Sunlight"],
            "Gmaj7": ["G6", "Em7", "Am7"],
            "Em7": ["G6", "Am7", "Bm7"],
            "D6": ["G6", "Gmaj7", "Bm7"],
            "Am7": ["D6", "G6", "Em7"],
            "Bm7": ["Em7", "Am7", "D6"],
            "Sunlight": ["G6", "Em7", "Gmaj7"],
        },
    ),

    # ═══════════════════════════════════════════════════════════════════════════
    # MYSTICAL & ETHEREAL - Otherworldly, floating quality
    # ═══════════════════════════════════════════════════════════════════════════
    Scale(
        name="E Phrygian Dominant",
        # E F G# A B C E (Spanish/Middle Eastern - dramatic, exotic)
        frequencies=[164.8, 174.6, 207.7, 220.0, 246.9, 261.6],
        chords={
            # Dark home
            "E": [0, 2, 4],               # E G# B - major but dark context
            "E7sus4": [0, 3, 4],          # E A B - suspense
            # Movement & tension
            "F": [1, 3, 5],               # F A C - Neapolitan color
            "Am": [3, 5, 0],              # A C E - minor iv
            "Bdim": [4, 1, 3],            # B F A - diminished drama
            # Exotic color
            "Gaze": [2, 4, 0],            # G# B E - first inversion
            "Mirage": [1, 2, 5],          # F G# C - cluster tension
        },
        transitions={
            "E": ["F", "Am", "E7sus4", "Gaze"],
            "E7sus4": ["E", "F", "Bdim"],
            "F": ["E", "Am", "Mirage"],
            "Am": ["E", "Bdim", "F"],
            "Bdim": ["E", "Am", "E7sus4"],
            "Gaze": ["E", "F", "Am"],
            "Mirage": ["E", "Am", "F"],
        },
    ),
    Scale(
        name="A Hirajoshi",
        # A B C E F A (Japanese - melancholic, mysterious)
        frequencies=[220.0, 246.9, 261.6, 329.6, 349.2, 440.0],
        chords={
            # Eastern home
            "Am": [0, 2, 3],              # A C E - minor base
            "Am(b6)": [0, 3, 4],          # A E F - dark, unresolved
            # Movement
            "FMaj7": [4, 5, 2, 3],        # F A C E - warm resolution
            "Esus4": [3, 5, 1],           # E A B - suspended
            # Atmosphere
            "Koto": [0, 1, 3],            # A B E - open fourth
            "Zen": [1, 3, 5],             # B E A - spacious
            "Mist": [2, 4, 0],            # C F A - minor feel
        },
        transitions={
            "Am": ["FMaj7", "Esus4", "Am(b6)", "Koto"],
            "Am(b6)": ["Am", "FMaj7", "Mist"],
            "FMaj7": ["Am", "Esus4", "Zen"],
            "Esus4": ["Am", "FMaj7", "Koto"],
            "Koto": ["Am", "Zen", "Am(b6)"],
            "Zen": ["Am", "Koto", "FMaj7"],
            "Mist": ["Am", "FMaj7", "Esus4"],
        },
    ),
    Scale(
        name="C Whole Tone",
        # C D E F# G# A# (dreamlike, floating, no gravity)
        frequencies=[130.8, 146.8, 164.8, 185.0, 207.7, 233.1],
        chords={
            # All augmented/suspended - no resolution, pure drift
            "Caug": [0, 2, 4],            # C E G# - augmented float
            "Daug": [1, 3, 5],            # D F# A# - symmetrical
            "Eaug": [2, 4, 0],            # E G# C - cycling
            # Suspended clusters
            "Shimmer": [0, 1, 4],         # C D G# - bright cluster
            "Drift": [1, 2, 5],           # D E A# - wandering
            "Glow": [3, 4, 1],            # F# G# D - upper cluster
            "Void": [0, 3, 5],            # C F# A# - tritone based
        },
        transitions={
            "Caug": ["Shimmer", "Drift", "Eaug"],
            "Daug": ["Caug", "Glow", "Void"],
            "Eaug": ["Daug", "Shimmer", "Drift"],
            "Shimmer": ["Caug", "Drift", "Glow"],
            "Drift": ["Daug", "Eaug", "Void"],
            "Glow": ["Caug", "Shimmer", "Daug"],
            "Void": ["Caug", "Daug", "Eaug"],
        },
    ),

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEMPLATIVE & AMBIENT - Slow evolution, rich harmonics
    # ═══════════════════════════════════════════════════════════════════════════
    Scale(
        name="F Lydian",
        # F G A B C D F (bright, dreamy, cinematic)
        frequencies=[174.6, 196.0, 220.0, 246.9, 261.6, 293.7],
        chords={
            # Lydian brightness
            "Fmaj7#11": [0, 2, 3, 5],     # F A B D - quintessential lydian
            "Fmaj9": [0, 1, 2, 4],        # F G A C - lush, open
            # Movement
            "G/F": [1, 3, 5, 0],          # G B D F - dominant feel
            "Am7": [2, 4, 5, 1],          # A C D G - ii chord
            "Dm7": [5, 0, 2, 4],          # D F A C - vi chord
            # Atmosphere
            "Horizon": [0, 4, 1],         # F C G - stacked 5ths
            "Dawn": [2, 3, 5],            # A B D - open voicing
        },
        transitions={
            "Fmaj7#11": ["G/F", "Am7", "Horizon", "Fmaj9"],
            "Fmaj9": ["Fmaj7#11", "Dawn", "Dm7"],
            "G/F": ["Fmaj7#11", "Am7", "Dawn"],
            "Am7": ["Dm7", "Fmaj7#11", "G/F"],
            "Dm7": ["G/F", "Fmaj7#11", "Am7"],
            "Horizon": ["Fmaj7#11", "Dawn", "Am7"],
            "Dawn": ["Fmaj7#11", "Fmaj9", "Horizon"],
        },
    ),
    Scale(
        name="Eb Major 7",
        # Eb F G Bb C D (warm, lush, neo-soul)
        frequencies=[155.6, 174.6, 196.0, 233.1, 261.6, 293.7],
        chords={
            # Rich voicings
            "Ebmaj9": [0, 1, 2, 4],       # Eb F G C - extended major
            "Gm7": [2, 3, 5, 0],          # G Bb D Eb - iii chord
            "Cm9": [4, 5, 0, 2],          # C D Eb G - vi chord
            # Movement
            "Fmadd9": [1, 4, 3, 0],       # F C Bb Eb - sus feel
            "Bb6": [3, 5, 2, 0],          # Bb D G Eb - V chord color
            # Space
            "Velvet": [0, 2, 4],          # Eb G C - simple triad
            "Nocturne": [3, 0, 5],        # Bb Eb D - open voicing
        },
        transitions={
            "Ebmaj9": ["Gm7", "Cm9", "Nocturne", "Velvet"],
            "Gm7": ["Cm9", "Ebmaj9", "Fmadd9"],
            "Cm9": ["Fmadd9", "Bb6", "Ebmaj9"],
            "Fmadd9": ["Ebmaj9", "Bb6", "Gm7"],
            "Bb6": ["Ebmaj9", "Cm9", "Gm7"],
            "Velvet": ["Ebmaj9", "Gm7", "Nocturne"],
            "Nocturne": ["Velvet", "Ebmaj9", "Cm9"],
        },
    ),

    # ═══════════════════════════════════════════════════════════════════════════
    # TENSION & RELEASE - Emotional depth, satisfying resolutions
    # ═══════════════════════════════════════════════════════════════════════════
    Scale(
        name="B Minor Natural",
        # B C# D E F# G B (dark, yearning, cinematic)
        frequencies=[123.5, 138.6, 146.8, 164.8, 185.0, 196.0],
        chords={
            # Minor home
            "Bm": [0, 2, 4],              # B D F# - stable minor
            "Bm7": [0, 2, 4, 5],          # B D F# G - extended
            # Emotional movement
            "D": [2, 4, 0],               # D F# B - relative major
            "Em7": [3, 5, 0, 1],          # E G B C# - iv chord
            "F#7sus4": [4, 5, 0],         # F# G B - dominant tension
            # Color
            "Gmaj7": [5, 0, 1, 2],        # G B C# D - borrowed VI
            "Longing": [1, 3, 5],         # C# E G - diminished feel
        },
        transitions={
            "Bm": ["D", "Em7", "Bm7", "Gmaj7"],
            "Bm7": ["Bm", "F#7sus4", "Longing"],
            "D": ["Bm", "Em7", "Gmaj7"],
            "Em7": ["F#7sus4", "Bm", "D"],
            "F#7sus4": ["Bm", "Bm7", "Em7"],
            "Gmaj7": ["D", "Bm", "Em7"],
            "Longing": ["Bm", "F#7sus4", "Gmaj7"],
        },
    ),
    Scale(
        name="A Blues",
        # A C D Eb E G (blues scale - gritty, soulful)
        frequencies=[220.0, 261.6, 293.7, 311.1, 329.6, 392.0],
        chords={
            # Blues fundamentals
            "A7": [0, 4, 5, 1],           # A E G C - dominant 7th
            "A9": [0, 1, 4, 5],           # A C E G - 9th voicing
            # IV and V
            "D7": [2, 0, 1, 5],           # D A C G - IV chord
            "E7sus4": [4, 5, 0],          # E G A - V chord sus
            # Blue notes & tension
            "Gritty": [3, 4, 0],          # Eb E A - blue note cluster
            "Power": [0, 4, 5],           # A E G - open power
            "Resolve": [0, 2, 4],         # A D E - resolution
        },
        transitions={
            "A7": ["D7", "Power", "A9", "Resolve"],
            "A9": ["A7", "E7sus4", "Gritty"],
            "D7": ["A7", "E7sus4", "Gritty"],
            "E7sus4": ["A7", "D7", "Resolve"],
            "Gritty": ["A7", "Power", "D7"],
            "Power": ["A7", "Gritty", "E7sus4"],
            "Resolve": ["A7", "D7", "A9"],
        },
    ),

    # ═══════════════════════════════════════════════════════════════════════════
    # HIGHER REGISTER - Crystalline, bell-like, ethereal
    # ═══════════════════════════════════════════════════════════════════════════
    Scale(
        name="E Minor Aeolian (High)",
        # E F# G A B C E (natural minor - pure, melancholic)
        frequencies=[329.6, 370.0, 392.0, 440.0, 493.9, 523.3],
        chords={
            # Minor home
            "Em": [0, 2, 4],              # E G B - pure minor
            "Em9": [0, 1, 4, 5],          # E F# B C - extended
            # Movement
            "Am7": [3, 5, 0, 2],          # A C E G - iv chord
            "G6": [2, 4, 1],              # G B F# - III chord
            "Bm7": [4, 0, 1, 3],          # B E F# A - v chord
            # Atmosphere
            "Crystal": [0, 4, 5],         # E B C - bell-like
            "Starlight": [1, 3, 5],       # F# A C - diminished color
        },
        transitions={
            "Em": ["Am7", "G6", "Em9", "Crystal"],
            "Em9": ["Em", "Bm7", "Starlight"],
            "Am7": ["Em", "G6", "Bm7"],
            "G6": ["Em", "Am7", "Crystal"],
            "Bm7": ["Em", "Am7", "G6"],
            "Crystal": ["Em", "Em9", "Starlight"],
            "Starlight": ["Em", "Am7", "Crystal"],
        },
    ),
    Scale(
        name="D Mixolydian (Bright)",
        # D E F# G A B C D (dominant mode - uplifting, celebratory)
        frequencies=[293.7, 329.6, 370.0, 392.0, 440.0, 493.9],
        chords={
            # Bright home
            "D7": [0, 2, 4, 5],           # D F# A C - dominant 7
            "D6": [0, 2, 4, 5],           # D F# A B - 6th voicing
            # Movement
            "G": [3, 5, 0],               # G B D - IV chord
            "Am7": [4, 5, 0, 1],          # A B D E - v chord
            "Em7": [1, 3, 5, 0],          # E G B D - ii chord
            # Energy
            "Jubilant": [0, 3, 4],        # D G A - sus4 energy
            "Radiant": [2, 4, 0],         # F# A D - first inversion
        },
        transitions={
            "D7": ["G", "Am7", "D6", "Jubilant"],
            "D6": ["D7", "Em7", "Radiant"],
            "G": ["D7", "Am7", "Em7"],
            "Am7": ["D7", "G", "Em7"],
            "Em7": ["Am7", "D7", "G"],
            "Jubilant": ["D7", "G", "Radiant"],
            "Radiant": ["D7", "Jubilant", "Am7"],
        },
    ),
]


# --- WAVEFORM GENERATORS ---


def generate_sine(phases: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * phases)


def generate_square(phases: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(2 * np.pi * phases))


def generate_sawtooth(phases: np.ndarray) -> np.ndarray:
    return 2.0 * (phases % 1.0) - 1.0


def generate_triangle(phases: np.ndarray) -> np.ndarray:
    return np.abs((phases % 1.0) * 4.0 - 2.0) - 1.0


def generate_organ(phases: np.ndarray) -> np.ndarray:
    """Sine wave with added harmonics."""
    return 0.5 * np.sin(2 * np.pi * phases) + \
           0.25 * np.sin(4 * np.pi * phases) + \
           0.125 * np.sin(6 * np.pi * phases) + \
           0.06 * np.sin(8 * np.pi * phases)


WAVEFORM_GENERATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Sine": generate_sine,
    "Square": generate_square,
    "Sawtooth": generate_sawtooth,
    "Triangle": generate_triangle,
    "Organ": generate_organ,
}


# --- SYNTHESIZER ---


class Synthesizer:
    def __init__(self) -> None:
        self.current_scale_index: int = 0
        self.scales: list[Scale] = SCALES

        self.voices: list[Voice] = [Voice(freq=f) for f in self.current_scale.frequencies]

        self.lfo_phase: float = 0.0
        self.random_mode: bool = False
        self.current_chord_name: str = ""
        self.waveform_names: list[str] = list(WAVEFORM_GENERATORS.keys())
        self.current_waveform_index: int = 0

        self.q: Queue[np.ndarray] = Queue(maxsize=10)

        # Rhythm state
        self.current_rhythm_pattern: int = 0
        self.rhythm_step: int = 0
        self.current_velocity: float = 1.0
        self.base_tempo: float = 0.5

    @property
    def current_scale(self) -> Scale:
        return self.scales[self.current_scale_index]

    @property
    def current_rhythm(self) -> RhythmPattern:
        return RHYTHM_PATTERNS[self.current_rhythm_pattern]

    @property
    def current_waveform(self) -> str:
        return self.waveform_names[self.current_waveform_index]

    def next_scale(self) -> None:
        """Switch to the next scale."""
        self.current_scale_index = (self.current_scale_index + 1) % len(self.scales)

        for i, freq in enumerate(self.current_scale.frequencies):
            if i < len(self.voices):
                self.voices[i].freq = freq

        self._silence_all_voices()
        self.current_chord_name = ""

    def next_waveform(self) -> None:
        """Switch to the next waveform type."""
        self.current_waveform_index = (self.current_waveform_index + 1) % len(self.waveform_names)

    def toggle_voice(self, index: int) -> None:
        """Toggle a specific voice on or off."""
        if 0 <= index < len(self.voices):
            voice = self.voices[index]
            voice.target_amp = 0.0 if voice.target_amp > 0.0 else 1.0

    def set_random_mode(self, enabled: bool) -> None:
        """Enable or disable generative mode."""
        self.random_mode = enabled
        if not enabled:
            self._silence_all_voices()

    def _silence_all_voices(self) -> None:
        """Set all voices to silent."""
        for v in self.voices:
            v.target_amp = 0.0
            v.pending_target_amp = 0.0
            v.onset_delay_samples = 0

    def _select_next_chord(self) -> str:
        """Select the next chord based on transition rules."""
        scale = self.current_scale

        # Check if we can follow a transition from the current chord
        if self.current_chord_name and self.current_chord_name in scale.transitions:
            candidates = [c for c in scale.transitions[self.current_chord_name] if c in scale.chords]
            if candidates:
                # Prefer primary transition with configured probability
                if len(candidates) > 1 and random.random() < PRIMARY_TRANSITION_WEIGHT:
                    return candidates[0]
                return random.choice(candidates)

        return random.choice(list(scale.chords.keys()))

    def _calculate_bass_factor(self, freq: float) -> float:
        """Calculate amplitude boost based on frequency (lower = more boost)."""
        freqs = [v.freq for v in self.voices]
        min_freq, max_freq = min(freqs), max(freqs)
        freq_range = max_freq - min_freq if max_freq > min_freq else 1.0
        freq_position = (freq - min_freq) / freq_range
        return BASS_BOOST - (BASS_BOOST - 1.0) * freq_position

    def _calculate_stagger_order(self, chord_indices: list[int]) -> list[tuple[int, int]]:
        """Calculate stagger delays for chord voices (bass first with jitter)."""
        chord_voices = [(i, self.voices[i].freq) for i in chord_indices]
        chord_voices.sort(key=lambda x: x[1] + random.uniform(-STAGGER_FREQ_JITTER, STAGGER_FREQ_JITTER))

        result = []
        num_voices = len(chord_voices)
        for position, (voice_idx, _) in enumerate(chord_voices):
            stagger_ratio = position / max(1, num_voices - 1)
            stagger_ms = STAGGER_MIN_MS + (STAGGER_MAX_MS - STAGGER_MIN_MS) * stagger_ratio
            delay_samples = int(stagger_ms * SAMPLE_RATE / 1000)
            result.append((voice_idx, delay_samples))

        return result

    def play_random_chord(self) -> None:
        """Trigger a chord change in generative mode."""
        if not self.random_mode:
            return

        chord_name = self._select_next_chord()
        chord_indices = self.current_scale.chords[chord_name]

        # Get velocity from current rhythm step
        step = self.current_rhythm.steps[self.rhythm_step % len(self.current_rhythm.steps)]
        self.current_velocity = step.velocity

        # Calculate stagger order
        stagger_map = dict(self._calculate_stagger_order(chord_indices))

        # Apply to voices
        for i, voice in enumerate(self.voices):
            if i in chord_indices:
                bass_factor = self._calculate_bass_factor(voice.freq)
                variation = 1.0 - VOICE_VARIATION + random.random() * (2 * VOICE_VARIATION)
                final_amp = step.velocity * bass_factor * variation

                voice.pending_target_amp = final_amp
                voice.onset_delay_samples = stagger_map.get(i, 0)
            else:
                voice.target_amp = 0.0
                voice.onset_delay_samples = 0
                voice.pending_target_amp = 0.0

        self.rhythm_step += 1
        self.current_chord_name = chord_name

    def next_rhythm(self) -> None:
        """Switch to the next rhythm pattern."""
        self.current_rhythm_pattern = (self.current_rhythm_pattern + 1) % len(RHYTHM_PATTERNS)
        self.rhythm_step = 0

    def increase_tempo(self) -> None:
        """Increase tempo (shorter durations)."""
        self.base_tempo = max(0.125, 0.9 * self.base_tempo)

    def decrease_tempo(self) -> None:
        """Decrease tempo (longer durations)."""
        self.base_tempo = min(3.0, 1.1 * self.base_tempo)

    def get_next_chord_duration(self) -> float:
        """Get duration until next chord change."""
        step = self.current_rhythm.steps[self.rhythm_step % len(self.current_rhythm.steps)]
        return self.base_tempo * step.duration

    def callback(self, outdata: np.ndarray, frame_cnt: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """Audio callback for sounddevice."""
        final_signal = np.zeros((frame_cnt, 1))

        # Process voices
        for voice in self.voices:
            self._process_voice_envelope(voice, frame_cnt)

            if voice.current_amp < 0.001 and voice.target_amp == 0.0:
                voice.current_amp = 0.0
                continue

            signal = self._generate_voice_signal(voice, frame_cnt)
            final_signal += signal

        # Add LFO texture
        final_signal += self._generate_lfo(frame_cnt)

        # Soft clip to prevent harsh distortion
        final_signal = np.tanh(final_signal)

        outdata[:] = final_signal

        if not self.q.full():
            self.q.put(final_signal.copy())

    def _process_voice_envelope(self, voice: Voice, frame_cnt: int) -> None:
        """Process stagger delay and amplitude smoothing for a voice."""
        if voice.onset_delay_samples > 0:
            voice.onset_delay_samples -= frame_cnt
            if voice.onset_delay_samples <= 0:
                voice.target_amp = voice.pending_target_amp
                voice.onset_delay_samples = 0

        if voice.current_amp != voice.target_amp:
            diff = voice.target_amp - voice.current_amp
            base_smoothing = SMOOTHING_FACTOR_ATTACK if diff > 0 else SMOOTHING_FACTOR_RELEASE
            # Scale smoothing inversely with tempo so faster tempos have snappier envelopes
            tempo_factor = 1.0 / self.base_tempo
            smoothing = min(0.5, base_smoothing * tempo_factor)  # Cap at 0.5 to avoid overshoot
            voice.current_amp += diff * smoothing

    def _generate_voice_signal(self, voice: Voice, frame_cnt: int) -> np.ndarray:
        """Generate audio signal for a single voice."""
        inc = voice.freq / SAMPLE_RATE
        phases = (voice.phase + np.arange(frame_cnt) * inc).reshape(-1, 1)

        generator = WAVEFORM_GENERATORS.get(self.current_waveform, generate_sine)
        raw_wave = generator(phases)

        voice.phase = (voice.phase + frame_cnt * inc) % 1.0

        return OSC_AMP * voice.current_amp * raw_wave

    def _generate_lfo(self, frame_cnt: int) -> np.ndarray:
        """Generate LFO signal."""
        inc = LFO_FREQ / SAMPLE_RATE
        phases = (self.lfo_phase + np.arange(frame_cnt) * inc).reshape(-1, 1)
        lfo = LFO_AMP * np.sin(2 * np.pi * phases)

        self.lfo_phase = (self.lfo_phase + frame_cnt * inc) % 1.0

        return lfo


# --- UI ---


class Visualizer:
    """Terminal UI for the synthesizer."""

    HELP_TEXT = "Synth (1-6: Note, R: Random, S: Scale, W: Wave, T: Rhythm, ↑↓: Tempo, C: Clear, Q: Quit)"

    def __init__(self, stdscr: curses.window, synth: Synthesizer) -> None:
        self.stdscr = stdscr
        self.synth = synth
        self.max_y, self.max_x = 0, 0
        self.center_y = 0
        self.last_change_time = time.time()
        self.next_chord_duration = 1.0

        self._setup_curses()

    def _setup_curses(self) -> None:
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(int(1000 / VIS_FPS))

        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)

        self._update_dimensions()

    def _update_dimensions(self) -> None:
        self.max_y, self.max_x = self.stdscr.getmaxyx()
        self.center_y = self.max_y // 2

    def run(self) -> None:
        """Main UI loop."""
        while True:
            self._update_generative_mode()

            if self._handle_input():
                break

            self._render()

    def _update_generative_mode(self) -> None:
        if self.synth.random_mode and time.time() - self.last_change_time > self.next_chord_duration:
            self.synth.play_random_chord()
            self.next_chord_duration = self.synth.get_next_chord_duration()
            self.last_change_time = time.time()

    def _handle_input(self) -> bool:
        """Handle keyboard input. Returns True if should quit."""
        try:
            key = self.stdscr.getch()
        except curses.error:
            return False

        if key == ord("q"):
            return True
        elif key == curses.KEY_RESIZE:
            self._update_dimensions()
            self.stdscr.clear()
        elif ord("1") <= key <= ord("6"):
            self.synth.random_mode = False
            self.synth.toggle_voice(key - ord("1"))
            self.last_change_time = time.time()
        elif key in (ord("r"), ord("R")):
            self.synth.set_random_mode(True)
            self.last_change_time = time.time()
        elif key in (ord("w"), ord("W")):
            self.synth.next_waveform()
        elif key in (ord("s"), ord("S")):
            self.synth.next_scale()
            if self.synth.random_mode:
                self.synth.play_random_chord()
                self.last_change_time = time.time()
        elif key in (ord("c"), ord("C")):
            self.synth.set_random_mode(False)
        elif key in (ord("t"), ord("T")):
            self.synth.next_rhythm()
        elif key == curses.KEY_UP:
            self.synth.increase_tempo()
        elif key == curses.KEY_DOWN:
            self.synth.decrease_tempo()

        return False

    def _render(self) -> None:
        try:
            data = self.synth.q.get_nowait()
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        except Empty:
            return

        try:
            self.stdscr.erase()
            self._draw_header()
            self._draw_waveform(data)
            self.stdscr.refresh()
        except curses.error:
            pass

    def _draw_header(self) -> None:
        self.stdscr.addstr(0, 0, self.HELP_TEXT, curses.color_pair(2))

        scale_wave = f"Scale: {self.synth.current_scale.name} | Wave: {self.synth.current_waveform}"
        self.stdscr.addstr(1, 0, scale_wave, curses.color_pair(1))

        # Active notes
        parts = ["Active: "]
        for i, v in enumerate(self.synth.voices):
            active = v.target_amp > 0.0 or v.pending_target_amp > 0.0
            parts.append(f"{i+1}:[{'X' if active else ' '}] ")

        if self.synth.random_mode and self.synth.current_chord_name:
            parts.append(f" | Chord: {self.synth.current_chord_name}")

        self.stdscr.addstr(2, 0, "".join(parts), curses.color_pair(2))

        # Rhythm info (random mode only)
        if self.synth.random_mode:
            vel_blocks = int(self.synth.current_velocity * 10)
            velocity_bar = "█" * vel_blocks + "░" * (10 - vel_blocks)
            # Convert base_tempo to BPM-like display (lower base_tempo = faster)
            speed_bps = 1.0 / self.synth.base_tempo
            rhythm_str = (
                f"Rhythm: {self.synth.current_rhythm.name:<12} | "
                f"Velocity: [{velocity_bar}] | "
                f"Speed: {speed_bps:.2f} BPS"
            )
            self.stdscr.addstr(3, 0, rhythm_str, curses.color_pair(1))

    def _draw_waveform(self, data: np.ndarray) -> None:
        if self.max_x <= 0 or len(data) == 0 or self.max_y <= 1:
            return

        step = max(1, len(data) // self.max_x)

        for x in range(self.max_x - 1):
            idx = x * step
            if idx >= len(data):
                break

            sample = float(data[idx][0])
            height_offset = int(sample * (self.max_y / 2) * 0.8)
            y_pos = max(0, min(self.max_y - 1, self.center_y - height_offset))

            try:
                self.stdscr.addch(y_pos, x, "◉", curses.color_pair(2))
            except curses.error:
                pass


# --- ENTRY POINT ---


def main() -> None:
    synth = Synthesizer()
    stream = sd.OutputStream(
        channels=1,
        callback=synth.callback,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
    )

    try:
        with stream:
            curses.wrapper(lambda stdscr: Visualizer(stdscr, synth).run())
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
