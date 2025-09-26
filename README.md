# BatonVoice

BatonVoice is an advanced Text-to-Speech (TTS) system that provides unified interfaces for multiple TTS modes with emotion control and prosodic feature manipulation. The project integrates state-of-the-art models including CosyVoice2 and Matcha-TTS to deliver high-quality speech synthesis with fine-grained control over speech characteristics.

## Features

### Core Functionality
- **Unified TTS Interface**: Single interface supporting multiple TTS modes
- **Emotion-Controlled Speech**: Generate speech with specific emotional characteristics
- **Prosodic Feature Control**: Fine-tune pitch, energy, and spectral features
- **Audio Feature Extraction**: Extract word-level features from audio files
- **Web-based Interface**: User-friendly Gradio interface for easy interaction

### Four Main Modes

1. **Mode 1: Text + Features to Audio**
   - Input: Text and predefined prosodic features
   - Output: High-quality audio with controlled characteristics
   - Use case: Precise control over speech prosody

2. **Mode 2: Text to Features + Audio**
   - Input: Text only
   - Output: Generated features and corresponding audio
   - Use case: Automatic feature generation with natural speech

3. **Mode 3: Audio to Text Features**
   - Input: Audio file
   - Output: Extracted text and prosodic features
   - Use case: Analysis and feature extraction from existing audio

4. **Mode 4: Text + Instruction to Features**
   - Input: Text and emotional/stylistic instructions
   - Output: AI-generated prosodic features
   - Use case: Emotion-driven feature generation using AI

## Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended)
- Git with submodule support

### Step-by-Step Installation

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive https://github.com/Tencent/digitalhuman.git
   cd digitalhuman/BatonVoice
   ```

2. **Update submodules**:
   ```bash
   git submodule update --init --recursive
   ```

3. **Create and activate Conda environment**:
   ```bash
   conda create -n batonvoice -y python=3.10
   conda activate batonvoice
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the CosyVoice2 model**:
   ```python
   from modelscope import snapshot_download
   snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
   ```

## Quick Start

### Command Line Usage

#### Basic Text-to-Speech
```bash
python unified_tts.py --text "Hello world, how are you today?" --output output.wav
```

#### Text with Custom Features
```bash
python unified_tts.py --text "Hello world" --features '[{"word": "Hello world", "pitch_mean": 280, "pitch_slope": 50, "energy_rms": 0.006, "energy_slope": 15, "spectral_centroid": 2400}]' --output output.wav
```

#### Audio Feature Extraction
```bash
python audio_feature_extractor.py --audio input.wav --output features.json
```

### Web Interface

Launch the Gradio web interface for interactive use:

```bash
python gradio_tts_interface.py
```

Then open the provided URL in your browser to access the web interface.

## Project Structure

```
batonvoice/
├── unified_tts.py              # Main TTS engine with unified interface
├── gradio_tts_interface.py     # Web-based user interface
├── audio_feature_extractor.py  # Audio analysis and feature extraction
├── openrouter_gemini_client.py # AI-powered feature generation
├── requirements.txt            # Python dependencies
├── prompt.wav                  # Default prompt audio file
├── third-party/               # External dependencies
│   ├── CosyVoice/             # CosyVoice2 TTS model
│   └── Matcha-TTS/            # Matcha-TTS model
└── pretrained_models/         # Downloaded model files
    └── CosyVoice2-0.5B/       # CosyVoice2 model directory
```

## API Reference

### UnifiedTTS Class

```python
from unified_tts import UnifiedTTS

# Initialize TTS engine
tts = UnifiedTTS(
    model_path='Yue-Wang/BATONTTS-1.7B',
    cosyvoice_model_dir='./pretrained_models/CosyVoice2-0.5B',
    prompt_audio_path='./prompt.wav'
)

# Mode 1: Text to speech
tts.text_to_speech("Hello world", "output1.wav")

# Mode 2: Text + features to speech
features = '[{"word": "Hello", "pitch_mean": 300, "pitch_slope": 50, "energy_rms": 0.006, "energy_slope": 15, "spectral_centroid": 2400}]'
tts.text_features_to_speech("Hello world", features, "output2.wav")
```

### AudioFeatureExtractor Class

```python
from audio_feature_extractor import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor()

# Extract features from audio
features = extractor.extract_features("input.wav")
print(features)
```

## Configuration

### Model Paths
You can customize model paths by modifying the default parameters:

- **BATON TTS Model**: `Yue-Wang/BATONTTS-1.7B`
- **CosyVoice2 Model**: `./pretrained_models/CosyVoice2-0.5B`
- **Whisper Model**: Default or custom path
- **Wav2Vec2 Model**: Default or custom path


## Vocal Features

The system uses the following vocal features for speech control:

- **pitch_mean**: Average pitch frequency (Hz)
- **pitch_slope**: Pitch contour slope
- **energy_rms**: Root mean square energy level
- **energy_slope**: Energy contour slope
- **spectral_centroid**: Spectral brightness measure



## Examples

### Emotional Speech Generation

```python
# Excited speech
features = '[{"word": "I\'m excited", "pitch_mean": 380, "pitch_slope": 120, "energy_rms": 0.008, "energy_slope": 40, "spectral_centroid": 3000}]'
tts.text_features_to_speech("I'm excited about this project!", features, "excited.wav")

# Calm speech
features = '[{"word": "Stay calm", "pitch_mean": 200, "pitch_slope": -10, "energy_rms": 0.003, "energy_slope": 5, "spectral_centroid": 1800}]'
tts.text_features_to_speech("Please stay calm", features, "calm.wav")
```



## Contributing

We welcome contributions to BatonVoice! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the terms specified in the original repository. Please refer to the LICENSE file for details.

## Acknowledgments

- **CosyVoice2**: Advanced TTS model from FunAudioLLM
- **Matcha-TTS**: High-quality TTS architecture
- **Whisper**: Speech recognition capabilities
- **Wav2Vec2**: Word-level alignment features

## Support

For questions, issues, or contributions, please:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Provide system specifications and error logs
4. Include minimal reproduction examples

---

**Note**: This project is part of the Tencent Digital Human initiative and represents cutting-edge research in controllable speech synthesis.