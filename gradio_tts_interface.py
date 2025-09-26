#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio TTS Interface Script

This script provides a web-based interface for four different TTS and audio processing modes:
1. Mode 1: Text + Features to Audio (unified_tts mode 2) with predefined examples
2. Mode 2: Text to Features + Audio (unified_tts mode 1)
3. Mode 3: Audio to Text Features (audio_feature_extractor)
4. Mode 4: Text + Instruction to Features (openrouter_gemini_client)

Usage:
    python gradio_tts_interface.py
    
Then open the provided URL in your browser to access the interface.
"""

import gradio as gr
import json
import os
import tempfile
import traceback
from typing import Optional, Tuple, List, Dict, Any

# Import the three main modules
try:
    from unified_tts import UnifiedTTS
except ImportError as e:
    print(f"Warning: Could not import unified_tts: {e}")
    UnifiedTTS = None

try:
    from audio_feature_extractor import AudioFeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import audio_feature_extractor: {e}")
    AudioFeatureExtractor = None

try:
    from openrouter_gemini_client import OpenRouterGeminiClient
except ImportError as e:
    print(f"Warning: Could not import openrouter_gemini_client: {e}")
    OpenRouterGeminiClient = None

# Global instances (initialized lazily)
tts_instance = None
extractor_instance = None

# ===== Test Examples for Mode 1 (from unified_tts.py) =====
# These examples are taken from the unified_tts.py test cases and will be used
# as predefined examples in Mode 1 interface
TEST_EXAMPLES = [
    {
        "text": "Kids are talking by the door",
        "features": '[{"word": "Kids are talking","pitch_mean": 315,"pitch_slope": 90,"energy_rms": 0.005,"energy_slope": 25,"spectral_centroid": 2650},{"word": "by the door","pitch_mean": 360,"pitch_slope": -110,"energy_rms": 0.004,"energy_slope": -30,"spectral_centroid": 2900}]',
        "instruction": "Generate natural speech"
    },
    {
        "text": "Hello world, how are you today?",
        "features": '[{"word": "Hello world","pitch_mean": 280,"pitch_slope": 50,"energy_rms": 0.006,"energy_slope": 15,"spectral_centroid": 2400},{"word": "how are you","pitch_mean": 320,"pitch_slope": -20,"energy_rms": 0.007,"energy_slope": 10,"spectral_centroid": 2600},{"word": "today?","pitch_mean": 350,"pitch_slope": -80,"energy_rms": 0.005,"energy_slope": -25,"spectral_centroid": 2800}]',
        "instruction": "Generate cheerful speech"
    },
    {
        "text": "The weather is beautiful outside",
        "features": '[{"word": "The weather is","pitch_mean": 250,"pitch_slope": 30,"energy_rms": 0.004,"energy_slope": 20,"spectral_centroid": 2200},{"word": "beautiful outside","pitch_mean": 300,"pitch_slope": -60,"energy_rms": 0.006,"energy_slope": -15,"spectral_centroid": 2500}]',
        "instruction": "Generate calm and peaceful speech"
    },
    {
        "text": "I'm excited about this project!",
        "features": '[{"word": "I\'m excited","pitch_mean": 380,"pitch_slope": 120,"energy_rms": 0.008,"energy_slope": 40,"spectral_centroid": 3000},{"word": "about this","pitch_mean": 350,"pitch_slope": 80,"energy_rms": 0.007,"energy_slope": 30,"spectral_centroid": 2800},{"word": "project!","pitch_mean": 400,"pitch_slope": -100,"energy_rms": 0.009,"energy_slope": -20,"spectral_centroid": 3200}]',
        "instruction": "Generate excited and energetic speech"
    },
    {
        "text": "Please be quiet in the library",
        "features": '[{"word": "Please be quiet","pitch_mean": 200,"pitch_slope": -10,"energy_rms": 0.003,"energy_slope": 5,"spectral_centroid": 1800},{"word": "in the library","pitch_mean": 180,"pitch_slope": -30,"energy_rms": 0.002,"energy_slope": -10,"spectral_centroid": 1600}]',
        "instruction": "Generate whispered and quiet speech"
    }
]

# ===== Utility Functions =====

def get_tts_instance() -> Optional[UnifiedTTS]:
    """
    Get or create a global TTS instance for reuse across requests.
    
    This function implements lazy loading to avoid initializing heavy models
    until they are actually needed. The instance is cached globally to prevent
    repeated model loading.
    
    Returns:
        UnifiedTTS instance or None if initialization fails
    """
    global tts_instance
    if tts_instance is None and UnifiedTTS is not None:
        try:
            tts_instance = UnifiedTTS()
            print("✅ TTS instance initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize TTS instance: {e}")
            return None
    return tts_instance

def get_extractor_instance() -> Optional[AudioFeatureExtractor]:
    """
    Get or create a global AudioFeatureExtractor instance for reuse.
    
    Similar to get_tts_instance(), this implements lazy loading and caching
    for the audio feature extraction models.
    
    Returns:
        AudioFeatureExtractor instance or None if initialization fails
    """
    global extractor_instance
    if extractor_instance is None and AudioFeatureExtractor is not None:
        try:
            extractor_instance = AudioFeatureExtractor()
            print("✅ Audio extractor instance initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize audio extractor instance: {e}")
            return None
    return extractor_instance

def load_example(example_idx: int) -> Tuple[str, str, str]:
    """
    Load a predefined example for Mode 1.
    
    This function retrieves one of the predefined test examples and returns
    the text, features, and instruction for use in the Gradio interface.
    
    Args:
        example_idx: Index of the example to load (0-4)
        
    Returns:
        Tuple of (text, features_json, instruction)
    """
    if 0 <= example_idx < len(TEST_EXAMPLES):
        example = TEST_EXAMPLES[example_idx]
        return example["text"], example["features"], example["instruction"]
    else:
        return "", "", "Generate natural speech"

# ===== Mode 1: Text + Features to Audio (unified_tts mode 2) =====

def mode1_text_features_to_audio(text: str, features: str, instruction: str, speed: float) -> Tuple[Optional[str], str]:
    """
    Mode 1: Convert text and features to audio using unified_tts mode 2.
    
    This function takes text input along with prosodic features and generates
    speech audio. It uses the UnifiedTTS class in mode 2, which accepts
    pre-defined word-level features to control the prosody of the output.
    
    Args:
        text: Input text to synthesize
        features: JSON string containing word-level prosodic features
        instruction: Instruction for controlling speech characteristics
        speed: Speech speed multiplier (1.0 = normal speed)
        
    Returns:
        Tuple of (audio_file_path, status_message)
        
    Implementation Logic:
        1. Validate inputs and get TTS instance
        2. Create temporary output file
        3. Call unified_tts.text_features_to_speech() method
        4. Return audio file path and status message
    """
    try:
        # Input validation
        if not text.strip():
            return None, "❌ Error: Text input is required"
        if not features.strip():
            return None, "❌ Error: Features input is required"
            
        # Get TTS instance
        tts = get_tts_instance()
        if tts is None:
            return None, "❌ Error: Failed to initialize TTS model"
            
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            
        # Generate audio using mode 2
        success = tts.text_features_to_speech(
            text=text,
            word_features=features,
            output_path=output_path,
            instruct=instruction,
            speed=speed
        )
        
        if success and os.path.exists(output_path):
            return output_path, f"✅ Audio generated successfully! Text: '{text[:50]}...'"
        else:
            return None, "❌ Error: Audio generation failed"
            
    except Exception as e:
        error_msg = f"❌ Error in Mode 1: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return None, error_msg

# ===== Mode 2: Text to Features + Audio (unified_tts mode 1) =====

def mode2_text_to_features_audio(text: str, instruction: str, speed: float) -> Tuple[Optional[str], str, str]:
    """
    Mode 2: Convert text to features and audio using unified_tts mode 1.
    
    This function takes only text input and generates both prosodic features
    and speech audio. It uses the UnifiedTTS class in mode 1, which internally
    generates word-level features and then converts them to speech.
    
    Args:
        text: Input text to synthesize
        instruction: Instruction for controlling speech characteristics
        speed: Speech speed multiplier (1.0 = normal speed)
        
    Returns:
        Tuple of (audio_file_path, generated_features_json, status_message)
        
    Implementation Logic:
        1. Validate inputs and get TTS instance
        2. Create temporary output file
        3. Call unified_tts.text_to_speech() method
        4. Extract generated features from the process
        5. Return audio file, features, and status message
    """
    try:
        # Input validation
        if not text.strip():
            return None, "", "❌ Error: Text input is required"
            
        # Get TTS instance
        tts = get_tts_instance()
        if tts is None:
            return None, "", "❌ Error: Failed to initialize TTS model"
            
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            
        # Generate audio using mode 1
        success = tts.text_to_speech(
            text=text,
            output_path=output_path,
            instruct=instruction,
            speed=speed
        )
        
        if success and os.path.exists(output_path):
            # Note: In the current unified_tts implementation, we don't have direct access
            # to the generated features. This would need to be modified in the original
            # code to return features as well. For now, we return a placeholder.
            features_placeholder = "[Features generated internally - not accessible in current implementation]"
            return output_path, features_placeholder, f"✅ Audio generated successfully! Text: '{text[:50]}...'"
        else:
            return None, "", "❌ Error: Audio generation failed"
            
    except Exception as e:
        error_msg = f"❌ Error in Mode 2: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return None, "", error_msg

# ===== Mode 3: Audio to Text Features (audio_feature_extractor) =====

def mode3_audio_to_features(audio_file) -> Tuple[str, str]:
    """
    Mode 3: Extract text features from audio using audio_feature_extractor.
    
    This function takes an uploaded audio file and extracts both the transcribed
    text and word-level prosodic features. It uses the AudioFeatureExtractor
    class to perform speech recognition and feature extraction.
    
    Args:
        audio_file: Uploaded audio file from Gradio interface
        
    Returns:
        Tuple of (extracted_features_json, status_message)
        
    Implementation Logic:
        1. Validate audio input and get extractor instance
        2. Load audio file using the extractor
        3. Transcribe audio to get text
        4. Extract word-level timestamps and features
        5. Format results as JSON and return with status
    """
    try:
        # Input validation
        if audio_file is None:
            return "", "❌ Error: Audio file is required"
            
        # Get extractor instance
        extractor = get_extractor_instance()
        if extractor is None:
            return "", "❌ Error: Failed to initialize audio feature extractor"
            
        # Load audio file
        audio_path = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
        audio_array, sampling_rate = extractor.load_audio_file(audio_path)
        
        # Transcribe audio
        transcription = extractor.transcribe_audio(audio_array, sampling_rate)
        if not transcription:
            return "", "❌ Error: Failed to transcribe audio"
            
        # Get word-level timestamps
        aligned_segments = extractor.get_word_timestamps(audio_array, transcription)
        
        # Extract features (this would need to be implemented in the original extractor)
        # For now, we return the transcription and basic timing information
        result = {
            "transcription": transcription,
            "segments": []
        }
        
        for segment in aligned_segments:
            segment_data = {
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "words": []
            }
            
            for word in segment.words:
                word_data = {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                    "score": word.score
                }
                segment_data["words"].append(word_data)
                
            result["segments"].append(segment_data)
            
        features_json = json.dumps(result, indent=2, ensure_ascii=False)
        return features_json, f"✅ Features extracted successfully! Transcription: '{transcription[:50]}...'"
        
    except Exception as e:
        error_msg = f"❌ Error in Mode 3: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return "", error_msg

# ===== Mode 4: Text + Instruction to Features (openrouter_gemini_client) =====

def mode4_text_instruction_to_features(api_key: str, text: str, instruction: str, prompt_choice: int) -> Tuple[str, str]:
    """
    Mode 4: Generate features from text and instruction using OpenRouter Gemini.
    
    This function takes text and an emotional/stylistic instruction and generates
    prosodic features using the OpenRouter Gemini API. It supports two different
    prompt templates with different characteristics.
    
    Args:
        api_key: OpenRouter API key for authentication
        text: Input text to generate features for
        instruction: Emotional or stylistic instruction
        prompt_choice: Choice of prompt template (1 or 2)
        
    Returns:
        Tuple of (generated_features_json, status_message)
        
    Implementation Logic:
        1. Validate inputs and API key
        2. Initialize OpenRouter Gemini client
        3. Generate prosodic features using selected prompt template
        4. Extract and validate JSON response
        5. Return features and status message
    """
    try:
        # Input validation
        if not api_key.strip():
            return "", "❌ Error: OpenRouter API key is required"
        if not text.strip():
            return "", "❌ Error: Text input is required"
        if not instruction.strip():
            return "", "❌ Error: Instruction is required"
            
        # Check if OpenRouter client is available
        if OpenRouterGeminiClient is None:
            return "", "❌ Error: OpenRouter Gemini client not available"
            
        # Initialize client
        client = OpenRouterGeminiClient(api_key=api_key)
        
        # Generate features
        result = client.generate_prosodic_features(
            text=text,
            instruction=instruction,
            prompt_choice=prompt_choice
        )
        
        if result['success']:
            features_json = result['prosodic_features']
            # Validate JSON format
            try:
                json.loads(features_json)
                return features_json, f"✅ Features generated successfully! Text: '{text[:50]}...'"
            except json.JSONDecodeError:
                return features_json, "⚠️ Features generated but JSON format may be invalid"
        else:
            error_msg = result.get('error', 'Unknown error')
            return "", f"❌ Error: {error_msg}"
            
    except Exception as e:
        error_msg = f"❌ Error in Mode 4: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return "", error_msg

# ===== Gradio Interface Creation =====

def create_gradio_interface():
    """
    Create and configure the main Gradio interface with four tabs.
    
    This function sets up the complete web interface with four different modes,
    each in its own tab. It configures all the input/output components and
    connects them to the appropriate processing functions.
    
    Returns:
        Configured Gradio interface ready to launch
        
    Interface Structure:
        - Tab 1: Mode 1 (Text + Features → Audio) with examples
        - Tab 2: Mode 2 (Text → Features + Audio)
        - Tab 3: Mode 3 (Audio → Text Features)
        - Tab 4: Mode 4 (Text + Instruction → Features)
    """
    
    with gr.Blocks(title="TTS Multi-Mode Interface", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎙️ TTS Multi-Mode Interface
        
        This interface provides four different modes for text-to-speech and audio processing:
        
        - **Mode 1**: Text + Features → Audio (with predefined examples)
        - **Mode 2**: Text → Features + Audio  
        - **Mode 3**: Audio → Text Features
        - **Mode 4**: Text + Instruction → Features (using OpenRouter Gemini)
        """)
        
        # ===== Tab 1: Mode 1 - Text + Features to Audio =====
        with gr.Tab("Mode 1: Text + Features → Audio"):
            gr.Markdown("""
            ### Mode 1: Text + Features to Audio
            Input text along with prosodic features to generate speech audio.
            Use the example buttons below to load predefined test cases.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    mode1_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=3
                    )
                    mode1_features = gr.Textbox(
                        label="Prosodic Features (JSON)",
                        placeholder="Enter word-level features in JSON format...",
                        lines=8
                    )
                    mode1_instruction = gr.Textbox(
                        label="Instruction",
                        value="Generate natural speech",
                        placeholder="Enter instruction for speech characteristics..."
                    )
                    mode1_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed"
                    )
                    
                with gr.Column(scale=1):
                    mode1_audio_output = gr.Audio(label="Generated Audio")
                    mode1_status = gr.Textbox(label="Status", interactive=False)
                    
            mode1_generate_btn = gr.Button("🎵 Generate Audio", variant="primary")
            
            # Example buttons for Mode 1
            gr.Markdown("### 📋 Predefined Examples")
            with gr.Row():
                example_btns = []
                for i, example in enumerate(TEST_EXAMPLES):
                    btn = gr.Button(f"Example {i+1}: {example['text'][:30]}...", size="sm")
                    example_btns.append(btn)
                    
            # Connect example buttons
            for i, btn in enumerate(example_btns):
                btn.click(
                    fn=lambda idx=i: load_example(idx),
                    outputs=[mode1_text, mode1_features, mode1_instruction]
                )
                
            # Connect generate button
            mode1_generate_btn.click(
                fn=mode1_text_features_to_audio,
                inputs=[mode1_text, mode1_features, mode1_instruction, mode1_speed],
                outputs=[mode1_audio_output, mode1_status]
            )
        
        # ===== Tab 2: Mode 2 - Text to Features + Audio =====
        with gr.Tab("Mode 2: Text → Features + Audio"):
            gr.Markdown("""
            ### Mode 2: Text to Features + Audio
            Input only text to generate both prosodic features and speech audio.
            The model will automatically generate appropriate features internally.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    mode2_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=4
                    )
                    mode2_instruction = gr.Textbox(
                        label="Instruction",
                        value="Generate natural speech",
                        placeholder="Enter instruction for speech characteristics..."
                    )
                    mode2_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speech Speed"
                    )
                    mode2_generate_btn = gr.Button("🎵 Generate Audio & Features", variant="primary")
                    
                with gr.Column(scale=1):
                    mode2_audio_output = gr.Audio(label="Generated Audio")
                    mode2_features_output = gr.Textbox(
                        label="Generated Features",
                        lines=8,
                        interactive=False
                    )
                    mode2_status = gr.Textbox(label="Status", interactive=False)
                    
            # Connect generate button
            mode2_generate_btn.click(
                fn=mode2_text_to_features_audio,
                inputs=[mode2_text, mode2_instruction, mode2_speed],
                outputs=[mode2_audio_output, mode2_features_output, mode2_status]
            )
        
        # ===== Tab 3: Mode 3 - Audio to Text Features =====
        with gr.Tab("Mode 3: Audio → Text Features"):
            gr.Markdown("""
            ### Mode 3: Audio to Text Features
            Upload an audio file to extract transcribed text and word-level features.
            The system will perform speech recognition and feature extraction.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    mode3_audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    mode3_extract_btn = gr.Button("🔍 Extract Features", variant="primary")
                    
                with gr.Column(scale=1):
                    mode3_features_output = gr.Textbox(
                        label="Extracted Features (JSON)",
                        lines=12,
                        interactive=False
                    )
                    mode3_status = gr.Textbox(label="Status", interactive=False)
                    
            # Connect extract button
            mode3_extract_btn.click(
                fn=mode3_audio_to_features,
                inputs=[mode3_audio_input],
                outputs=[mode3_features_output, mode3_status]
            )
        
        # ===== Tab 4: Mode 4 - Text + Instruction to Features =====
        with gr.Tab("Mode 4: Text + Instruction → Features"):
            gr.Markdown("""
            ### Mode 4: Text + Instruction to Features
            Generate prosodic features from text and emotional/stylistic instructions using OpenRouter Gemini API.
            
            **⚠️ Note about Prompt Templates:**
            - **Template 1**: Standard template for reliable feature generation
            - **Template 2**: Experimental template that may be more expressive but could generate additional words not in the original text
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    mode4_api_key = gr.Textbox(
                        label="OpenRouter API Key",
                        type="password",
                        placeholder="Enter your OpenRouter API key..."
                    )
                    mode4_text = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to generate features for...",
                        lines=3
                    )
                    mode4_instruction = gr.Textbox(
                        label="Emotional/Stylistic Instruction",
                        placeholder="e.g., 'happy and excited', 'calm and peaceful', 'sad and melancholic'...",
                        lines=2
                    )
                    mode4_prompt_choice = gr.Radio(
                        choices=[("Template 1 (Standard)", 1), ("Template 2 (Experimental)", 2)],
                        value=1,
                        label="Prompt Template"
                    )
                    mode4_generate_btn = gr.Button("🤖 Generate Features", variant="primary")
                    
                with gr.Column(scale=1):
                    mode4_features_output = gr.Textbox(
                        label="Generated Features (JSON)",
                        lines=12,
                        interactive=False
                    )
                    mode4_status = gr.Textbox(label="Status", interactive=False)
                    
            # Connect generate button
            mode4_generate_btn.click(
                fn=mode4_text_instruction_to_features,
                inputs=[mode4_api_key, mode4_text, mode4_instruction, mode4_prompt_choice],
                outputs=[mode4_features_output, mode4_status]
            )
        
        # ===== Footer Information =====
        gr.Markdown("""
        ---
        ### 📝 Usage Notes:
        - **Mode 1**: Best for precise control over prosodic features
        - **Mode 2**: Best for quick text-to-speech with automatic feature generation
        - **Mode 3**: Best for analyzing existing audio files
        - **Mode 4**: Best for generating features with specific emotional characteristics
        
        ### 🔧 Technical Requirements:
        - CUDA-compatible GPU recommended for optimal performance
        - Sufficient GPU memory for model loading
        - Valid OpenRouter API key for Mode 4
        """)
    
    return interface

# ===== Main Application Entry Point =====

def main():
    """
    Main function to launch the Gradio interface.
    
    This function creates the interface and launches it with appropriate
    configuration for both local development and deployment.
    """
    print("🚀 Initializing TTS Multi-Mode Interface...")
    
    # Create interface
    interface = create_gradio_interface()
    
    # Launch interface
    print("🌐 Launching Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True              # Enable debug mode
    )

if __name__ == "__main__":
    main()