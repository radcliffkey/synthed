import curses
import numpy as np
import sounddevice as sd
import random
import time
from queue import Queue, Empty
from typing import Any
from dataclasses import dataclass

# --- CONFIGURATION ---
SAMPLE_RATE: int = 44100
BLOCK_SIZE: int = 1024
VIS_FPS: int = 30

# Synth Parameters
SMOOTHING_FACTOR: float = 0.1
LFO_FREQ: float = 5.0
LFO_AMP: float = 0.15
OSC_AMP: float = 0.2  # Lower amplitude to prevent clipping with polyphony


@dataclass
class Scale:
    name: str
    frequencies: list[float]
    chords: dict[str, list[int]]


# Define Scales
# All scales have 6 notes (5 distinct + octave) to map to keys 1-6
SCALES = [
    Scale(
        name="A Minor Pentatonic",
        frequencies=[110.0, 130.8, 146.8, 164.8, 196.0, 220.0],
        chords={
            "Am": [0, 1, 3],
            "C": [1, 3, 4],
            "Asus4": [0, 2, 3],
            "G6": [3, 4, 2],
            "Em7": [3, 4, 2, 5],
            "Cluster": [0, 1, 2],
        },
    ),
    Scale(
        name="C Major Pentatonic",
        frequencies=[130.8, 146.8, 164.8, 196.0, 220.0, 261.6],
        chords={
            "C": [0, 2, 3],
            "Am": [4, 0, 2],
            "Gsus2": [3, 4, 1],
            "Em": [2, 3, 4],
            "Dm7": [1, 4, 0],
            "C6": [0, 2, 4],
        },
    ),
    Scale(
        name="G Major Pentatonic",
        frequencies=[98.0, 110.0, 123.5, 146.8, 164.8, 196.0],
        chords={
            "G": [0, 2, 3],
            "Em": [4, 0, 2],
            "Em/G": [5, 4, 2],
            "G6": [0, 2, 4],
            "Em7": [4, 5, 2, 3],
            "Dsus4": [3, 5, 1],
            "Bm7": [2, 3, 4, 1],
            "Am": [1, 5, 4],
            "Open": [0, 3, 5],
            "Cluster": [1, 2, 3],
        },
    ),
    Scale(
        name="E Minor Pentatonic",
        frequencies=[82.4, 98.0, 110.0, 123.5, 146.8, 164.8],
        chords={
            "Em": [0, 1, 4],
            "G": [1, 3, 4],
            "A7": [2, 5, 3, 1],
            "G/B": [3, 4, 1],
            "Dsus4": [4, 1, 2],
            "Dsus2": [4, 5, 2],
            "Em7": [0, 1, 4, 2],
        },
    ),
    Scale(
        name="D Minor Pentatonic",
        frequencies=[146.8, 174.6, 196.0, 220.0, 261.6, 293.7],
        chords={
            "Dm": [0, 1, 3],
            "F": [1, 3, 4],
            "Gsus4": [2, 4, 5],
            "Am7": [3, 4, 0, 2],
            "Csus4": [4, 1, 2],
            "Csus2": [4, 5, 2],
        },
    ),
]


@dataclass
class Voice:
    freq: float
    phase: float = 0.0
    current_amp: float = 0.0
    target_amp: float = 0.0


class Synthesizer:
    def __init__(self):
        self.current_scale_index: int = 0
        self.scales: list[Scale] = SCALES

        # Voices: List of Voice objects to track phase and amplitude per scale index
        self.voices: list[Voice] = []
        current_scale = self.scales[self.current_scale_index]
        for freq in current_scale.frequencies:
            self.voices.append(Voice(freq=freq))

        self.lfo_phase: float = 0.0
        self.random_mode: bool = False
        self.current_chord_name: str = ""
        # Thread-safe queue to pass audio data to the visualizer
        self.q: Queue[np.ndarray] = Queue(maxsize=10)

    def next_scale(self) -> None:
        """Switch to the next scale in the list."""
        self.current_scale_index = (self.current_scale_index + 1) % len(self.scales)
        new_scale = self.scales[self.current_scale_index]

        # Update frequencies for existing voices
        # We assume all scales have 6 notes
        for i, freq in enumerate(new_scale.frequencies):
            if i < len(self.voices):
                self.voices[i].freq = freq

        # Optional: Silence voices when switching scales?
        # Let's silence them to avoid dissonance with held notes from previous key
        for v in self.voices:
            v.target_amp = 0.0

        # Reset chord name
        self.current_chord_name = ""

    def toggle_voice_state(self, index: int) -> None:
        """Toggle a specific voice on or off."""
        if 0 <= index < len(self.voices):
            # If target is 0, turn it on (1.0). If > 0, turn it off (0.0).
            if self.voices[index].target_amp > 0.0:
                self.voices[index].target_amp = 0.0
            else:
                self.voices[index].target_amp = 1.0

    def set_random_mode(self, enabled: bool) -> None:
        self.random_mode = enabled
        # If disabling random mode, maybe we want to clear all notes?
        # Or keep them? Let's clear them for a fresh start.
        if not enabled:
            for v in self.voices:
                v.target_amp = 0.0

    def play_random_chord(self) -> None:
        """Pick a random chord to play."""
        if not self.random_mode:
            return

        current_scale = self.scales[self.current_scale_index]

        # Pick a random chord from our dictionary
        chord_name = random.choice(list(current_scale.chords.keys()))
        indices = current_scale.chords[chord_name]

        # Set target amp: 1.0 if index is in chord, 0.0 otherwise
        for i, v in enumerate(self.voices):
            if i in indices:
                v.target_amp = 1.0
            else:
                v.target_amp = 0.0

        # We might want to display the current chord name in the UI later
        self.current_chord_name = chord_name

    def callback(self, outdata: np.ndarray, frame_cnt: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """
        Audio callback for sounddevice.
        """
        if status:
            pass

        # Initialize signal buffer
        final_signal = np.zeros((frame_cnt, 1))

        # --- DSP Logic ---
        # Sum all active voices
        for v in self.voices:
            # Smooth amplitude envelope
            if v.current_amp != v.target_amp:
                diff = v.target_amp - v.current_amp
                v.current_amp += diff * SMOOTHING_FACTOR

            # Skip if silent
            if v.current_amp < 0.001 and v.target_amp == 0.0:
                v.current_amp = 0.0
                continue

            # Calculate phase increment
            inc: float = v.freq / SAMPLE_RATE
            phases: np.ndarray = v.phase + np.arange(frame_cnt) * inc
            phases = phases.reshape(-1, 1)

            # Generate sine wave
            signal: np.ndarray = OSC_AMP * v.current_amp * np.sin(2 * np.pi * phases)
            final_signal += signal

            # Update phase
            v.phase += frame_cnt * inc
            v.phase -= np.floor(v.phase)

        # 2. LFO (Global Vibrato/Tremolo? Let's apply to everything for texture)
        # Or just add LFO as a separate drone?
        # The original code added LFO to the signal. Let's keep it as a subtle texture.
        lfo_inc: float = LFO_FREQ / SAMPLE_RATE
        lfo_phases: np.ndarray = self.lfo_phase + np.arange(frame_cnt) * lfo_inc
        lfo_phases = lfo_phases.reshape(-1, 1)
        lfo: np.ndarray = LFO_AMP * np.sin(2 * np.pi * lfo_phases)

        # Apply LFO only if there is sound, to avoid weird drone when silent?
        # Original code just added it. Let's add it.
        final_signal += lfo

        # Update LFO phase
        self.lfo_phase += frame_cnt * lfo_inc
        self.lfo_phase -= np.floor(self.lfo_phase)

        # 3. Output
        outdata[:] = final_signal

        # 4. Visualization
        if not self.q.full():
            self.q.put(final_signal.copy())


def draw_visualizer(stdscr: curses.window, synth: Synthesizer) -> None:
    """
    The main loop for the Terminal UI.
    """
    # Curses setup
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking input
    stdscr.timeout(int(1000 / VIS_FPS))

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)

    max_y: int
    max_x: int
    max_y, max_x = stdscr.getmaxyx()
    center_y: int = max_y // 2

    last_change_time: float = time.time()

    while True:
        # 1. Generative Logic: Change note every 1.0 seconds
        if synth.random_mode and time.time() - last_change_time > 1.0:
            synth.play_random_chord()
            last_change_time = time.time()

        # 2. Handle Quit and Control
        try:
            key: int = stdscr.getch()
            if key == ord("q"):
                break
            elif key == curses.KEY_RESIZE:
                max_y, max_x = stdscr.getmaxyx()
                center_y = max_y // 2
                stdscr.clear()
            elif ord("1") <= key <= ord("6"):
                synth.random_mode = False
                idx = key - ord("1")
                synth.toggle_voice_state(idx)
                last_change_time = time.time()
            elif key in (ord("r"), ord("R")):
                synth.set_random_mode(True)
                last_change_time = time.time()
            elif key in (ord("s"), ord("S")):
                synth.next_scale()
                # If in random mode, trigger immediate update
                if synth.random_mode:
                    synth.play_random_chord()
                    last_change_time = time.time()
            elif key in (ord("c"), ord("C")):
                synth.set_random_mode(False)
                for v in synth.voices:
                    v.target_amp = 0.0
        except curses.error:
            pass

        # 3. Visualization Logic
        try:
            # Get latest audio chunk
            data: np.ndarray = synth.q.get_nowait()
            # assert data.ndim == 2 and data.shape[1] == 1, f"visualizer data shape mismatch: {data.shape}"

            # Clear screen
            stdscr.erase()

            # Draw Info
            stdscr.addstr(
                0, 0, "Polyphonic Synth (1-6: Note, R: Random, S: Scale, C: Clear, Q: Quit)", curses.color_pair(2)
            )

            # Draw Scale Name
            scale_name = synth.scales[synth.current_scale_index].name
            stdscr.addstr(1, 0, f"Scale: {scale_name}", curses.color_pair(1))

            # Draw Active Notes
            status_str = "Active: "
            for i, v in enumerate(synth.voices):
                state = "[X]" if v.target_amp > 0.0 else "[ ]"
                status_str += f"{i+1}:{state} "

            if synth.random_mode and synth.current_chord_name:
                status_str += f" | Chord: {synth.current_chord_name}"

            stdscr.addstr(2, 0, status_str, curses.color_pair(2))

            # Draw Waveform
            # We downsample the buffer to fit the screen width
            # Ensure step is at least 1
            step: int = max(1, len(data) // max_x)

            for x in range(0, max_x - 1):
                idx = x * step
                if idx < len(data):
                    # Normalize sample (-1.0 to 1.0) to screen height
                    # Scale by 0.8 to avoid hitting the absolute edges
                    sample: float = float(data[idx][0])

                    # Calculate height offset from center
                    height_offset: int = int(sample * (max_y / 2) * 0.8)

                    # Plot point
                    y_pos: int = center_y - height_offset

                    # Clamp y_pos to stay within screen bounds
                    y_pos = max(0, min(max_y - 1, y_pos))

                    try:
                        stdscr.addch(y_pos, x, "@", curses.color_pair(2))  # ░ ▓ ▒ █
                    except curses.error:
                        # Ignore errors drawing to bottom-right corner
                        pass

            stdscr.refresh()

        except Empty:
            # Queue is empty, just wait for next frame
            pass
        except Exception:
            # Catch other errors to prevent crash, but maybe log them in real app
            pass


def main() -> None:
    synth = Synthesizer()

    # Start Audio Stream
    stream = sd.OutputStream(channels=1, callback=synth.callback, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE)

    try:
        with stream:
            # Start UI
            # Pass synth instance to the visualizer using a lambda or partial
            curses.wrapper(lambda stdscr: draw_visualizer(stdscr, synth))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
