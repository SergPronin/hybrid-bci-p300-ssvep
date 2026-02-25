## Hybrid BCI SSVEP Stimulus (Stage 1)

This project implements the first stage of a hybrid P300/SSVEP BCI system –
the visual stimulus interface for SSVEP.

### Structure

- `app/main.py` – entry point, wires config and experiment.
- `app/experiment.py` – application layer, manages PsychoPy window and main loop.
- `app/config.py` – experiment configuration (window, grid, colors).
- `stimulus/tile.py` – logical model of a single SSVEP tile.
- `stimulus/grid.py` – logical grid of tiles (rows × cols).
- `stimulus/flicker_controller.py` – controls on/off state based on FPS and frequency.
- `infrastructure/logger.py` – logging abstraction (ready for future LSL/file logging).

### Requirements

Dependencies are listed in `requirements.txt` and include PsychoPy:

```bash
pip install -r requirements.txt
```

### Running

From the repository root:

```bash
python -m app.main
```

This will:

1. Load configuration from `app/config.py`.
2. Open a PsychoPy window (currently windowed mode, 1000×800).
3. Create a 2×2 grid of tiles with default SSVEP frequencies `[8, 10, 12, 15]` Hz.
4. Start the frame-synchronous flicker loop (no `sleep`, only `window.flip()`).
5. Allow exit with the `ESC` key.

### Configuration

Edit `app/config.py` to adjust:

- window size and fullscreen flag,
- target FPS,
- grid size (rows, cols) and tile size,
- tile frequencies (must be exact divisors of FPS),
- background / ON / OFF colors.

The configuration is validated so that:

- frequencies are integer divisors of FPS, and
- their count matches the grid size.

### Extensibility

The architecture separates:

- **Application layer** (`app/`) – lifecycle, PsychoPy window, input handling.
- **Stimulus/domain layer** (`stimulus/`) – pure SSVEP timing and state logic.
- **Infrastructure layer** (`infrastructure/`) – logging (future: LSL, file I/O).

This makes it straightforward to:

- add P300 paradigms alongside SSVEP,
- plug in LSL markers inside the experiment loop,
- change the UI while keeping stimulus timing logic intact.

