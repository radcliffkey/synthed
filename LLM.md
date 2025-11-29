# LLM.md - Project Context for AI Assistants

## Project Overview

**synthed** is an algorithmic music generator with CLI visualization. It's a polyphonic synthesizer that runs in the terminal, providing real-time audio synthesis and waveform visualization using curses.

## Architecture

### Core Components

1. **Synthesizer Class** (`synthed.py`)
   - Manages multiple voices (polyphonic synthesis)
   - Handles scale switching and chord progression with transitions
   - Provides audio callback for sounddevice
   - Thread-safe queue for visualization data
   - Rhythm pattern management with velocity dynamics
   - Multiple waveform selection

2. **Voice Class** (`synthed.py`)
   - Individual oscillator with phase tracking
   - Smooth amplitude enveloping (separate attack/release)
   - Staggered onset support (`onset_delay_samples`, `pending_target_amp`)

3. **Scale Class** (`synthed.py`)
   - Defines frequency sets (various scales and modes)
   - Contains chord dictionaries for each scale
   - Includes transition rules between chords
   - 6 notes per scale

4. **RhythmPattern / RhythmStep Classes** (`synthed.py`)
   - Define duration and velocity patterns
   - Multiple patterns: Steady, Swing, Syncopated, Contemplative, Chaotic, Driving, Waltz

5. **Visualizer Class** (`synthed.py`)
   - Curses-based terminal UI
   - Real-time waveform display
   - Interactive controls
   - Status display for active notes, chords, rhythm, and tempo

## Audio Pipeline

```
User Input → Synthesizer → Voices (Waveform) → Staggered Onset → Bass Boost
                                                                      ↓
                                                                    LFO
                                                                      ↓
                                                              Soft Clip (tanh)
                                                                      ↓
                                                              Audio Output
                                                                      ↓
                                                           Visualization Queue
```

1. **Sample Rate**: 44100 Hz
2. **Block Size**: 1024 samples
3. **Visualization FPS**: 30
4. **Envelope Smoothing**: Attack: 0.08, Release: 0.04
5. **LFO**: 5 Hz @ 0.15 amplitude
6. **Staggered Onset**: 10-40ms (bass notes first)
7. **Bass Boost**: 1.35x for lower frequencies

## User Controls

- `1-6`: Toggle individual notes (disables random mode)
- `R`: Enable random/generative mode
- `S`: Switch to next scale
- `W`: Switch waveform (Sine, Square, Sawtooth, Triangle, Organ)
- `T`: Switch rhythm pattern
- `↑/↓`: Increase/decrease tempo
- `C`: Clear all notes (disable random mode)
- `Q`: Quit application

## Development Notes

### Dependencies
- `numpy>=2.3.5`: Audio buffer manipulation
- `sounddevice>=0.5.3`: Real-time audio I/O
- `curses`: Built-in, terminal UI (Unix-like systems)

Non-python dependency: `libportaudio2` has to be installed on your system.

### Python Version
- Requires Python 3.12+ (uses modern type hints)

### Package Management
- **IMPORTANT**: This project uses `uv` for dependency management
- Always use `uv run` to execute Python code and tests
- Do NOT use `python` or `python3` directly

### Code Style
- Black formatter (line length: 120)
- Type hints throughout
- Dataclasses for data structures

## Configuration Constants

Key constants at top of `synthed.py`:

### Audio Core
- `SAMPLE_RATE`: 44100 Hz
- `BLOCK_SIZE`: 1024 samples
- `VIS_FPS`: 30 (visualization framerate)

### Synthesis Parameters
- `SMOOTHING_FACTOR_ATTACK`: 0.08 (note on envelope)
- `SMOOTHING_FACTOR_RELEASE`: 0.04 (note off envelope)
- `LFO_FREQ`: 5.0 Hz
- `LFO_AMP`: 0.15
- `OSC_AMP`: 0.2

### Musical Expression
- `STAGGER_MIN_MS`: 10.0 (minimum stagger delay)
- `STAGGER_MAX_MS`: 40.0 (maximum stagger delay)
- `BASS_BOOST`: 1.35 (amplitude multiplier for low frequencies)
- `PRIMARY_TRANSITION_WEIGHT`: 0.3 (probability of preferred chord transition)
- `VOICE_VARIATION`: 0.1 (±10% amplitude variation per voice)
- `STAGGER_FREQ_JITTER`: 50.0 Hz (randomization for stagger ordering)

## Available Scales

12 scales organized into categories:

### Warm & Familiar
- **A Minor Pentatonic**: Classic bluesy, warm
- **D Dorian**: Jazzy, sophisticated minor
- **G Major Pentatonic**: Bright, optimistic, pastoral

### Mystical & Ethereal
- **E Phrygian Dominant**: Spanish/Middle Eastern, dramatic
- **A Hirajoshi**: Japanese, melancholic, mysterious
- **C Whole Tone**: Dreamlike, floating, no gravity

### Contemplative & Ambient
- **F Lydian**: Bright, dreamy, cinematic
- **Eb Major 7**: Warm, lush, neo-soul

### Tension & Release
- **B Minor Natural**: Dark, yearning, cinematic
- **A Blues**: Gritty, soulful

### Higher Register
- **E Minor Aeolian (High)**: Crystalline, bell-like
- **D Mixolydian (Bright)**: Uplifting, celebratory

## Available Waveforms

- **Sine**: Pure, smooth tone
- **Square**: Hollow, rich in odd harmonics
- **Sawtooth**: Bright, buzzy, rich in all harmonics
- **Triangle**: Soft, mellow (between sine and square)
- **Organ**: Sine with added harmonics (drawbar organ style)

## Rhythm Patterns

- **Steady**: Even 4-beat pattern
- **Swing**: Long-short feel
- **Syncopated**: Off-beat accents
- **Contemplative**: Slow, spacious
- **Chaotic**: Irregular, unpredictable
- **Driving**: Fast, energetic
- **Waltz**: 3/4 time feel

## Common Tasks

### Adding a New Scale
1. Add to `SCALES` list in `synthed.py`
2. Define 6 frequencies
3. Define chord dictionary with note indices
4. Define transitions dictionary mapping chord names to lists of valid next chords
5. Scale will automatically appear in rotation (S key)

### Adding a New Waveform
1. Create generator function: `def generate_name(phases: np.ndarray) -> np.ndarray`
2. Add to `WAVEFORM_GENERATORS` dictionary

### Adding a New Rhythm Pattern
1. Add `RhythmPattern` to `RHYTHM_PATTERNS` list
2. Define steps with duration multipliers and velocity values

### Extending Audio DSP
Audio synthesis happens in `Synthesizer.callback()`:
- Voice signals generated in `_generate_voice_signal()`
- LFO added via `_generate_lfo()`
- Soft clipping applied with `np.tanh()`
- Add effects before `outdata[:] = final_signal`

## Important Considerations

### Thread Safety
- The audio callback runs on a separate thread
- Use the Queue (`synth.q`) for passing data to UI thread
- Don't modify voice state from callback (handled via target_amp)

### Performance
- Audio callback must be real-time safe
- Avoid allocations in callback when possible
- Use NumPy for vectorized operations

### Cross-Platform
- Currently Unix/Linux focused (curses)
- Windows would require `windows-curses` package or alternative UI

## Testing

**IMPORTANT**: Always use `uv run` to execute the code.

Run the synthesizer:
```bash
uv run synthed.py
```

Expected behavior:
- Terminal UI appears immediately
- Audio stream starts
- Pressing 1-6 toggles notes
- Pressing R starts generative mode with chord progressions
- Pressing T cycles through rhythm patterns
- Arrow keys adjust tempo
- Waveform displays in real-time


## Project Philosophy

This is a creative audio tool focused on:
- **Simplicity**: Single file, minimal dependencies
- **Interactivity**: Real-time control and visualization
- **Musicality**: Curated scales and chord transitions ensure pleasant harmonies
- **Generative**: Algorithmic composition with rhythm and dynamics

The poetic README ("Forgive me father for I have synthed...") reflects the playful, experimental nature of the project.
