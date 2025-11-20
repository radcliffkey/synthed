# CLAUDE.md - Project Context for AI Assistants

## Project Overview

**synthed** is an algorithmic music generator with CLI visualization. It's a polyphonic synthesizer that runs in the terminal, providing real-time audio synthesis and waveform visualization using curses.

## Architecture

### Core Components

1. **Synthesizer Class** (`main.py`)
   - Manages multiple voices (polyphonic synthesis)
   - Handles scale switching and chord progression
   - Provides audio callback for sounddevice
   - Thread-safe queue for visualization data

2. **Voice Class** (`main.py`)
   - Individual oscillator with phase tracking
   - Smooth amplitude enveloping
   - Target amplitude for smooth transitions

3. **Scale Class** (`main.py`)
   - Defines frequency sets (pentatonic scales)
   - Contains chord dictionaries for each scale
   - 6 notes per scale (5 distinct + octave)

4. **Visualizer** (`draw_visualizer` function)
   - Curses-based terminal UI
   - Real-time waveform display
   - Interactive controls
   - Status display for active notes

## Audio Pipeline

```
User Input → Synthesizer → Voices (Sine Waves) → LFO → Audio Output
                                                     ↓
                                              Visualization Queue
```

1. **Sample Rate**: 44100 Hz
2. **Block Size**: 1024 samples
3. **Smoothing**: Exponential amplitude smoothing (factor: 0.1)
4. **LFO**: 5 Hz global modulation for texture

## User Controls

- `1-6`: Toggle individual notes (manual mode)
- `R`: Enable random/generative mode
- `S`: Switch to next scale
- `C`: Clear all notes (disable random mode)
- `Q`: Quit application

## Development Notes

### Dependencies
- `numpy>=2.3.5`: Audio buffer manipulation
- `sounddevice>=0.5.3`: Real-time audio I/O
- `curses`: Built-in, terminal UI (Unix-like systems)

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

## Common Tasks

### Adding a New Scale
1. Add to `SCALES` list in `main.py`
2. Define 6 frequencies (pentatonic + octave)
3. Define chord dictionary with note indices
4. Scale will automatically appear in rotation (S key)

### Modifying Synthesis Parameters
Key constants at top of `main.py`:
- `SAMPLE_RATE`: Audio sample rate
- `BLOCK_SIZE`: Audio buffer size
- `SMOOTHING_FACTOR`: Envelope smoothing (0-1)
- `LFO_FREQ`: LFO frequency in Hz
- `LFO_AMP`: LFO amplitude
- `OSC_AMP`: Oscillator amplitude

### Extending Audio DSP
Audio synthesis happens in `Synthesizer.callback()`:
- Add effects after line 228 (after LFO)
- Modify before `outdata[:] = final_signal`
- Consider amplitude to prevent clipping

## Important Considerations

### Thread Safety
- The audio callback runs on a separate thread
- Use the Queue (`synth.q`) for passing data to UI thread
- Don't modify voice state from callback

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
uv run main.py
```

Expected behavior:
- Terminal UI appears immediately
- Audio stream starts
- Pressing 1-6 toggles notes
- Pressing R starts generative mode
- Waveform displays in real-time


## Project Philosophy

This is a creative audio tool focused on:
- **Simplicity**: Single file, minimal dependencies
- **Interactivity**: Real-time control and visualization
- **Musicality**: Pentatonic scales ensure pleasant harmonies
- **Generative**: Algorithmic composition capabilities

The poetic README ("Forgive me father for I have synthed...") reflects the playful, experimental nature of the project.

