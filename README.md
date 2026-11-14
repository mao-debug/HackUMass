# VisionBridge

VisionBridge is a HackUMass smart glasses prototype for blind and visually
impaired users. Its planned capabilities are object detection, short scene
descriptions, OCR, speech output, voice commands, and obstacle awareness.

**Current scope: Phase 1 only.** The application loads validated YAML settings
and runs a bounded simulation using a generated sample image, a local image
file, or an optional webcam. The default run processes one synthetic frame
without opening a camera or display, loading an AI model, producing audio, or
contacting a network service. The sample is a test pattern, not an AI result.

This prototype supplements established mobility aids. It does not provide
precise distance measurements or guaranteed navigation safety. Evaluate future
assistive features in controlled conditions and avoid continuous speech that
could obscure environmental sounds.

## Set up on macOS

Use Python 3.10 or newer. Open this repository folder in VS Code, then run these
commands in its terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

The install command downloads Python packages from PyPI and installs this
repository in editable mode. This project requires user approval before using
third-party services, including package downloads; PyPI downloads were approved
for this initial setup. Ask before introducing another external service.
For runtime dependencies without Pytest, install `requirements.txt` instead.
Runtime dependencies are NumPy, OpenCV, and PyYAML; Pytest is for development.
Raspberry Pi-specific packages are excluded from these requirements.

In VS Code, use **Python: Select Interpreter** and choose
`.venv/bin/python`. Activate the virtual environment in each new terminal.
Editable installation lets source edits take effect without reinstalling.

## Run the simulation

From the repository root, with the virtual environment active:

```sh
# Process one synthetic frame; no camera or desktop display is needed.
python -m visionbridge

# Use the repository's YAML configuration explicitly.
python -m visionbridge --config config/settings.yaml

# Export the final synthetic frame to a local file.
python -m visionbridge --output artifacts/sample.png

# Reuse that file as an image input.
python -m visionbridge --image artifacts/sample.png

# Opt into webcam access and a preview for a bounded run.
python -m visionbridge --source webcam --camera-index 0 --max-frames 60 --preview

# Show all options.
python -m visionbridge --help
```

The installed `visionbridge` command is equivalent to `python -m visionbridge`.
Without `--config`, built-in defaults match `config/settings.yaml` and work
from any directory after installation. Paths supplied on the command line are
relative to the terminal's current directory. Paths inside a YAML file are
relative to that YAML file's directory.

`--source` accepts `sample`, `image`, or `webcam`. Without an explicit source,
`--image PATH` selects image input and `--camera-index N` selects webcam input.
`--max-frames` must be a positive integer. `--preview` enables a desktop window;
`--no-preview` disables it. Press **q**, **Escape**, or **Ctrl+C** to stop a
preview run. `--output PATH` writes the final processed frame and creates its
parent directories; the extension must be supported by OpenCV, such as `.png`.
Use `--log-level DEBUG` for more logging or `--version` for the package version.

The simulation reports frame processing only. It does not identify objects,
read text, describe scenes, estimate obstacle distances, or speak.

## Configuration and credentials

Edit [config/settings.yaml](config/settings.yaml) to set the input source,
image path, webcam index, frame dimensions, target frame rate, maximum frame
count, preview, output path, and log level. Command-line options override the
corresponding YAML values. Unknown settings and invalid values produce a
configuration error instead of silently falling back.

The `models` section reserves backend and model-path settings for later phases.
These fields do not load a model, install a dependency, or enable a feature in
Phase 1. No cloud provider has been selected; any future external service
requires approval before integration.

[.env.example](.env.example) documents the future credentials convention.
Phase 1 needs no environment variables or API keys and does not load `.env`
files. Keep real credentials out of version control. Local virtual
environments, generated artifacts, model files, and `.env` files are ignored.
After dependencies are installed, the simulation runs offline.

## Tests and local prerequisites

```sh
python -m pytest
python scripts/check_hardware.py
python scripts/check_hardware.py --json
```

The tests cover configuration validation, simulated input and capture failure
handling, controller and CLI behavior, and prerequisite reporting without
requiring physical peripherals. Automated tests do not verify real hardware.

The prerequisite report uses only Python's standard library and can run before
installing application dependencies. It discovers optional software and camera
device paths without importing hardware packages, opening devices, recording
audio, or making network requests. Camera capture, microphone input, and audio
output operation remain explicitly **unverified**. Track actual equipment and integration results in
[docs/hardware.md](docs/hardware.md).

## Troubleshooting

- **`No module named visionbridge` or missing dependencies:** activate `.venv`
  and complete the editable installation using the same `python` interpreter.
- **Camera unavailable:** check the webcam index and connection. On macOS,
  enable camera access for the terminal or VS Code in **System Settings >
  Privacy & Security > Camera**. Synthetic and image simulation remain available
  without camera access.
- **No desktop display or preview error:** omit `--preview` or pass
  `--no-preview`. The default run does not create windows.
- **Image cannot be read:** check the path and file format. Use an absolute
  path when moving between working directories or YAML files.
- **No microphone, speaker, or network:** none is needed for Phase 1. Voice
  input, speech output, and cloud services are not active.

## Project layout

The existing repository root is the VisionBridge project; no nested checkout is
needed.

```text
./
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── config/settings.yaml
├── docs/hardware.md
├── src/visionbridge/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── camera/capture.py
│   ├── ai/
│   │   ├── object_detection.py
│   │   ├── scene_description.py
│   │   └── text_recognition.py
│   ├── audio/
│   │   ├── speech_to_text.py
│   │   └── text_to_speech.py
│   ├── assistant/command_handler.py
│   └── core/
│       ├── controller.py
│       └── configuration.py
├── scripts/check_hardware.py
└── tests/
```

The AI, audio, and assistant modules document later-phase boundaries and contain
no active implementation. The controller currently handles simulated frame
capture, optional preview/export, and cleanup only.

## Later phases require approval

1. **Phase 1:** repository, Python packaging, configuration, simulation, setup,
   and tests — current implementation.
2. **Phase 2:** camera integration and configurable object detection.
3. **Phase 3:** OCR and local text-to-speech.
4. **Phase 4:** AI-generated scene descriptions, with local core operation.
5. **Phase 5:** microphone input and voice commands.
6. **Phase 6:** Raspberry Pi deployment, wearable integration, performance
   optimization, and the HackUMass demonstration.

The local simulation is designed to run on both Raspberry Pi 4 and Raspberry
Pi 5 using the same Python interface. CSI camera/Picamera2 support, board
validation, and performance measurements are not implemented or verified in
Phase 1. The Raspberry Pi 4 and Raspberry Pi AI Camera are available; the Pi 5
with 8 GB RAM is requested. Remaining peripheral availability and all hardware
integration results remain to be verified in the hardware inventory. Phase 2
will target the AI Camera, with its model/runtime compatibility verified before
selecting an object detection backend.
