#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified TTS Script
Combines functionality from new_emo_tts_server.py and single_inference_tts.py

Main Features:
1. Mode 1: Input text only, output audio (generates word_features internally)
2. Mode 2: Input text and features, output audio (uses provided features)
3. Uses model paths and test cases from single_inference_tts.py
4. Implements emotion TTS logic from new_emo_tts_server.py

Usage:
    # Mode 1: Text to speech
    tts = UnifiedTTS()
    tts.text_to_speech("Hello world", "output1.wav")
    
    # Mode 2: Text + features to speech
    features = '[{"word": "Hello", "pitch_mean": 300, ...}]'
    tts.text_features_to_speech("Hello world", features, "output2.wav")
"""

import os
import sys
import json
import torch
import torchaudio
import numpy as np
import argparse
import uuid
import logging
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add CosyVoice paths
sys.path.append('third-party/CosyVoice')
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedTTS:
    """
    Unified Text-to-Speech Processor
    
    This class provides a unified interface for two TTS modes:
    1. Mode 1 (text_to_speech): Input text only, generates word_features and speech_token internally
    2. Mode 2 (text_features_to_speech): Input text and word_features, generates speech_token only
    
    Architecture:
    - Uses vLLM for efficient model inference
    - Integrates CosyVoice2 for speech token to audio conversion
    - Supports both emotion-controlled and standard TTS generation
    - Provides easy-to-use API for external script integration
    
    Usage Example:
        tts = UnifiedTTS()
        # Mode 1: Text only
        success = tts.text_to_speech("Hello world", "output1.wav")
        # Mode 2: Text + features
        features = '[{"word": "Hello", "pitch_mean": 300}]'
        success = tts.text_features_to_speech("Hello", features, "output2.wav")
    """
    
    def __init__(self, 
                 model_path: str = 'Yue-Wang/BATONTTS-1.7B',
                 cosyvoice_model_dir: str = './pretrained_models/CosyVoice2-0.5B',
                 prompt_audio_path: str = './prompt.wav',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.7,
                 fp16: bool = False):
        """
        Initialize the Unified TTS Processor
        
        Args:
            model_path: Path to the main TTS model (Qwen3-1.7B-instruct-TTS)
            cosyvoice_model_dir: Directory path to CosyVoice2 model
            prompt_audio_path: Path to prompt audio file for voice cloning
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0)
            fp16: Whether to use half precision for CosyVoice2
        
        Implementation:
            1. Store configuration parameters
            2. Initialize model loading flags
            3. Defer actual model loading until first inference call
        """
        self.model_path = model_path
        self.cosyvoice_model_dir = cosyvoice_model_dir
        self.prompt_audio_path = prompt_audio_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.fp16 = fp16
        
        # Model components (loaded lazily)
        self.llm = None
        self.tokenizer = None
        self.cosyvoice = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = None
        
        # Cached prompt features
        self.prompt_token = None
        self.prompt_feat = None
        self.speaker_embedding = None
        
        # Special tokens and sampling parameters
        self.special_tokens = {}
        self.sampling_params = None
        
        # Model loading flag
        self.models_loaded = False
        
        logger.info(f"UnifiedTTS initialized with model: {model_path}")
    
    def _load_models(self):
        """
        Load all required models and components
        
        Loading Process:
        1. Initialize vLLM model with specified configuration
        2. Load tokenizer for text processing
        3. Initialize CosyVoice2 model for audio generation
        4. Preload prompt audio features for voice cloning
        5. Configure special tokens and sampling parameters
        
        This method is called automatically on first inference request
        to enable lazy loading and reduce initialization time.
        """
        if self.models_loaded:
            return
            
        logger.info("Loading models...")
        
        # Load main TTS model with vLLM
        logger.info(f"Loading main model from {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
        )
        
        # Load tokenizer separately
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load CosyVoice2 model
        logger.info(f"Loading CosyVoice2 model from {self.cosyvoice_model_dir}")
        self.cosyvoice = CosyVoice2(self.cosyvoice_model_dir, fp16=self.fp16)
        self.sample_rate = self.cosyvoice.sample_rate
        
        # Preload prompt audio features
        self._preload_prompt_features()
        
        # Configure special tokens
        self._setup_special_tokens()
        
        # Configure sampling parameters
        self._setup_sampling_params()
        
        self.models_loaded = True
        logger.info("All models loaded successfully!")
    
    def _preload_prompt_features(self):
        """
        Preload prompt audio features for voice cloning
        
        Feature Extraction Process:
        1. Load prompt audio file and resample to 16kHz
        2. Extract speech tokens using CosyVoice2 frontend
        3. Resample audio to model's native sample rate
        4. Extract speech features (mel-spectrogram)
        5. Extract speaker embedding for voice characteristics
        
        These features are cached and reused for all inference calls
        to maintain consistent voice characteristics.
        """
        if os.path.exists(self.prompt_audio_path):
            try:
                # Load and process prompt audio
                prompt_speech = load_wav(self.prompt_audio_path, 16000)
                
                # Extract speech tokens
                self.prompt_token, _ = self.cosyvoice.frontend._extract_speech_token(prompt_speech)
                logger.info(f"Preloaded prompt token, shape: {self.prompt_token.shape}")
                
                # Extract speech features
                prompt_speech_resample = torchaudio.transforms.Resample(
                    orig_freq=16000, new_freq=self.sample_rate
                )(prompt_speech)
                self.prompt_feat, _ = self.cosyvoice.frontend._extract_speech_feat(prompt_speech_resample)
                logger.info(f"Preloaded prompt feat, shape: {self.prompt_feat.shape}")
                
                # Extract speaker embedding
                self.speaker_embedding = self.cosyvoice.frontend._extract_spk_embedding(prompt_speech)
                logger.info(f"Preloaded speaker embedding, shape: {self.speaker_embedding.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to load prompt audio: {e}")
                self._use_default_prompt_features()
        else:
            logger.warning(f"Prompt audio not found: {self.prompt_audio_path}")
            self._use_default_prompt_features()
    
    def _use_default_prompt_features(self):
        """
        Use default (empty) prompt features when prompt audio is unavailable
        
        Default Features:
        - Empty speech token tensor
        - Zero-filled feature tensor with correct dimensions
        - Zero-filled speaker embedding with standard size
        
        These defaults allow the model to generate speech without
        voice cloning, using the model's default voice characteristics.
        """
        self.prompt_token = torch.zeros(1, 0, dtype=torch.int32)
        self.prompt_feat = torch.zeros(1, 0, 80)
        self.speaker_embedding = torch.zeros(1, 192)
        logger.info("Using default prompt features")
    
    def _setup_special_tokens(self):
        """
        Configure special tokens used for input formatting
        
        Special Token Usage:
        - custom_token_0: Marks start of instruction
        - custom_token_1: Separates instruction from text
        - custom_token_2: Separates text from features (mode 2) or marks end (mode 1)
        - custom_token_3: Separates word_features from speech_token in output
        - eos_token_id: Marks end of generation
        
        These tokens structure the input/output format for the model
        to understand different components of the TTS task.
        """
        self.special_tokens = {
            'custom_token_0': self.tokenizer('<custom_token_0>').input_ids[0],
            'custom_token_1': self.tokenizer('<custom_token_1>').input_ids[0],
            'custom_token_2': self.tokenizer('<custom_token_2>').input_ids[0],
            'custom_token_3': self.tokenizer('<custom_token_3>').input_ids[0],
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        logger.info("Special tokens configured")
    
    def _setup_sampling_params(self):
        """
        Configure sampling parameters for text generation
        
        Parameter Settings:
        - temperature: Controls randomness (0.6 for balanced creativity)
        - top_p: Nucleus sampling threshold (1.0 for full vocabulary)
        - max_tokens: Maximum generation length (1500 for speech tokens)
        - stop_token_ids: Stop generation at EOS token
        - repetition_penalty: Reduce repetitive outputs (1.1)
        
        These parameters are optimized for TTS generation quality
        and prevent common issues like repetition or truncation.
        """
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1.0,
            max_tokens=1500,
            stop_token_ids=[self.special_tokens['eos_token_id']],
            repetition_penalty=1.1,
        )
        logger.info("Sampling parameters configured")
    
    def _prepare_mode1_input(self, text: str, instruct: str = "Generate natural speech") -> str:
        """
        Prepare input for Mode 1: Text-only generation
        
        Input Format:
        <custom_token_0> + instruct + <custom_token_1> + text + <custom_token_2>
        
        Expected Output:
        word_features + <custom_token_3> + speech_token + <eos>
        
        Args:
            text: Input text to synthesize
            instruct: Instruction for emotion/style control
            
        Returns:
            Formatted input string for the model
            
        Implementation:
        1. Combine instruction and text with special token separators
        2. Return formatted string ready for tokenization
        """
        input_sequence = (
            "<custom_token_0>" + 
            instruct + 
            "<custom_token_1>" + 
            text + 
            "<custom_token_2>"
        )
        return input_sequence
    
    def _prepare_mode2_input(self, text: str, word_features: str, instruct: str = "Generate natural speech") -> str:
        """
        Prepare input for Mode 2: Text + features generation
        
        Input Format:
        <custom_token_0> + instruct + <custom_token_1> + text + <custom_token_2> + word_features + <custom_token_3>
        
        Expected Output:
        speech_token + <eos>
        
        Args:
            text: Input text to synthesize
            word_features: Pre-generated word-level features
            instruct: Instruction for emotion/style control
            
        Returns:
            Formatted input string for the model
            
        Implementation:
        1. Process word_features (JSON parsing if needed)
        2. Combine all components with special token separators
        3. Return formatted string ready for tokenization
        """
        # Process word_features (try JSON parsing, fallback to string)
        processed_features = self._process_features(word_features)
        
        input_sequence = (
            "<custom_token_0>" + 
            instruct + 
            "<custom_token_1>" + 
            text + 
            "<custom_token_2>" + 
            processed_features + 
            "<custom_token_3>"
        )
        return input_sequence
    
    def _process_features(self, features: str) -> str:
        """
        Process word features input (JSON parsing with fallback)
        
        Processing Logic:
        1. Attempt to parse as JSON for validation
        2. If successful, convert back to string representation
        3. If parsing fails, use original string as-is
        4. Log the processing result for debugging
        
        Args:
            features: Input features string (JSON or plain text)
            
        Returns:
            Processed features string
            
        This flexible approach handles both JSON-formatted features
        and plain text features, ensuring compatibility with various
        input formats while maintaining robustness.
        """
        try:
            # Try JSON parsing for validation
            parsed_features = json.loads(features)
            processed_features = str(parsed_features)
            logger.debug("Successfully parsed features as JSON")
            return processed_features
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Use original string if JSON parsing fails
            logger.debug(f"Using features as plain string: {e}")
            return features
    
    def _generate_response(self, prompt: str) -> Tuple[str, bool]:
        """
        Generate model response using vLLM
        
        Generation Process:
        1. Use vLLM's generate method with configured sampling parameters
        2. Extract generated text from model output
        3. Check if generation was truncated due to length limits
        4. Return both generated text and truncation status
        
        Args:
            prompt: Formatted input prompt for the model
            
        Returns:
            Tuple of (generated_text, is_truncated)
            
        The truncation flag helps identify potential quality issues
        when the generation hits the maximum token limit.
        """
        outputs = self.llm.generate([prompt], self.sampling_params)
        
        # Extract generated text and check truncation
        output = outputs[0]
        generated_text = output.outputs[0].text
        is_truncated = output.outputs[0].finish_reason == 'length'
        
        if is_truncated:
            logger.warning("Generation was truncated, may affect audio quality")
        
        return generated_text, is_truncated
    
    def _extract_mode1_outputs(self, generated_text: str) -> Tuple[str, List[int]]:
        """
        Extract word_features and speech_tokens from Mode 1 output
        
        Extraction Logic:
        1. Find <custom_token_3> as separator between components
        2. Text before separator = word_features
        3. Text after separator = speech_token sequence
        4. Convert speech_token text to integer list
        
        Args:
            generated_text: Complete model output text
            
        Returns:
            Tuple of (word_features_text, speech_token_list)
            
        The separation logic relies on the model's training format
        where custom_token_3 consistently separates these components.
        """
        try:
            custom_token_3_marker = "<custom_token_3>"
            
            if custom_token_3_marker in generated_text:
                # Split at the separator
                parts = generated_text.split(custom_token_3_marker, 1)
                word_features_text = parts[0].strip()
                speech_token_text = parts[1].strip() if len(parts) > 1 else ""
                
                # Convert speech_token text to integers
                speech_tokens = self._text_to_speech_tokens(speech_token_text)
                
                logger.info(f"Extracted word_features length: {len(word_features_text)}")
                logger.info(f"Extracted speech_tokens count: {len(speech_tokens)}")
                
                return word_features_text, speech_tokens
            else:
                logger.warning("custom_token_3 separator not found")
                return generated_text, []
                
        except Exception as e:
            logger.error(f"Error extracting mode1 outputs: {e}")
            return "", []
    
    def _extract_mode2_outputs(self, generated_text: str) -> List[int]:
        """
        Extract speech_tokens from Mode 2 output
        
        Extraction Logic:
        1. Entire generated text represents speech_token sequence
        2. Convert text directly to integer token list
        3. Apply offset correction for proper token values
        
        Args:
            generated_text: Model output containing speech tokens
            
        Returns:
            List of speech token integers
            
        Mode 2 output is simpler than Mode 1 since it only contains
        speech tokens without word_features separation.
        """
        try:
            speech_tokens = self._text_to_speech_tokens(generated_text)
            logger.info(f"Extracted speech_tokens count: {len(speech_tokens)}")
            return speech_tokens
        except Exception as e:
            logger.error(f"Error extracting mode2 outputs: {e}")
            return []
    
    def _text_to_speech_tokens(self, text: str) -> List[int]:
        """
        Convert text representation to speech token integers
        
        Conversion Process:
        1. Tokenize text using the model's tokenizer
        2. Apply offset correction to recover original token values
        3. Filter tokens to ensure they're in valid speech token range
        4. Return list of corrected integer tokens
        
        Args:
            text: Text representation of speech tokens
            
        Returns:
            List of speech token integers
            
        The offset correction (151669 + 100) reverses the encoding
        applied during model training to map speech tokens to the
        model's vocabulary space.
        """
        if not text.strip():
            return []
        
        try:
            # Encode text to token IDs
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Apply offset correction to recover original speech tokens
            speech_tokens = []
            offset = 151669 + 100  # Consistent with original implementation
            
            for token_id in token_ids:
                if token_id >= offset:
                    original_token = token_id - offset
                    speech_tokens.append(original_token)
            
            return speech_tokens
            
        except Exception as e:
            logger.error(f"Text to speech_token conversion failed: {e}")
            return []
    
    def _convert_tokens_to_audio(self, speech_tokens: List[int], speed: float = 1.0) -> Optional[torch.Tensor]:
        """
        Convert speech tokens to audio waveform
        
        Conversion Process:
        1. Validate speech tokens and convert to tensor format
        2. Generate unique inference ID for cache management
        3. Call CosyVoice2's token2wav method with prompt features
        4. Clean up inference cache to prevent memory leaks
        5. Return generated audio tensor
        
        Args:
            speech_tokens: List of speech token integers
            speed: Speech speed multiplier (1.0 = normal speed)
            
        Returns:
            Audio tensor or None if conversion fails
            
        The cache management ensures proper cleanup of temporary
        data structures used during audio generation.
        """
        if len(speech_tokens) == 0:
            logger.warning("Empty speech tokens provided")
            return None
        
        try:
            # Convert to tensor format
            speech_token_tensor = torch.tensor([speech_tokens], dtype=torch.int32)
            logger.debug(f"Converting speech tokens, shape: {speech_token_tensor.shape}")
            
            # Generate unique inference ID
            inference_uuid = str(uuid.uuid1())
            
            # Initialize cache
            with self.cosyvoice.model.lock:
                self.cosyvoice.model.hift_cache_dict[inference_uuid] = None
            
            try:
                # Generate audio using CosyVoice2
                tts_speech = self.cosyvoice.model.token2wav(
                    token=speech_token_tensor,
                    prompt_token=self.prompt_token,
                    prompt_feat=self.prompt_feat,
                    embedding=self.speaker_embedding,
                    token_offset=0,
                    uuid=inference_uuid,
                    finalize=True,
                    speed=speed
                )
                
                return tts_speech.cpu()
                
            finally:
                # Clean up cache
                with self.cosyvoice.model.lock:
                    if inference_uuid in self.cosyvoice.model.hift_cache_dict:
                        self.cosyvoice.model.hift_cache_dict.pop(inference_uuid)
                        
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    def _save_audio(self, audio_tensor: torch.Tensor, output_path: str) -> bool:
        """
        Save audio tensor to file
        
        Saving Process:
        1. Ensure output directory exists
        2. Use torchaudio to save tensor as WAV file
        3. Apply correct sample rate from CosyVoice2 model
        4. Handle any file I/O errors gracefully
        
        Args:
            audio_tensor: Generated audio waveform tensor
            output_path: Target file path for saving
            
        Returns:
            True if saving successful, False otherwise
            
        The method ensures proper directory creation and error handling
        for robust file operations across different environments.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio file
            torchaudio.save(output_path, audio_tensor, self.sample_rate)
            logger.info(f"Audio saved successfully to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def text_to_speech(self, text: str, output_path: str, instruct: str = "Generate natural speech", speed: float = 1.0) -> bool:
        """
        Mode 1: Convert text to speech (generates features internally)
        
        Process Flow:
        1. Load models if not already loaded
        2. Prepare Mode 1 input format with instruction and text
        3. Generate model response to get word_features and speech_tokens
        4. Extract both components from the generated output
        5. Convert speech_tokens to audio waveform
        6. Save audio to specified output path
        
        Args:
            text: Input text to synthesize
            output_path: Path to save the generated audio file
            instruct: Instruction for controlling speech characteristics
            speed: Speech speed multiplier (1.0 = normal)
            
        Returns:
            True if synthesis successful, False otherwise
            
        This mode is ideal when you only have text and want the model
        to automatically generate appropriate prosodic features.
        """
        try:
            # Ensure models are loaded
            self._load_models()
            
            logger.info(f"Mode 1: Text to speech - {text}")
            
            # Validate input
            if not text.strip():
                logger.error("Empty text provided")
                return False
            
            # Prepare input for Mode 1
            prompt = self._prepare_mode1_input(text, instruct)
            
            # Generate response
            generated_text, is_truncated = self._generate_response(prompt)
            
            # Extract word_features and speech_tokens
            word_features, speech_tokens = self._extract_mode1_outputs(generated_text)
            
            if len(speech_tokens) == 0:
                logger.error("Failed to extract speech tokens")
                return False
            
            # Convert to audio
            audio_tensor = self._convert_tokens_to_audio(speech_tokens, speed)
            if audio_tensor is None:
                logger.error("Failed to convert tokens to audio")
                return False
            
            # Save audio file
            success = self._save_audio(audio_tensor, output_path)
            
            if success:
                logger.info(f"Mode 1 synthesis completed successfully!")
                return True
            else:
                logger.error("Failed to save audio file")
                return False
                
        except Exception as e:
            logger.error(f"Mode 1 synthesis failed: {e}")
            return False
    
    def text_features_to_speech(self, text: str, word_features: str, output_path: str, 
                               instruct: str = "Generate natural speech", speed: float = 1.0) -> bool:
        """
        Mode 2: Convert text + features to speech (uses provided features)
        
        Process Flow:
        1. Load models if not already loaded
        2. Prepare Mode 2 input format with instruction, text, and features
        3. Generate model response to get speech_tokens only
        4. Extract speech_tokens from the generated output
        5. Convert speech_tokens to audio waveform
        6. Save audio to specified output path
        
        Args:
            text: Input text to synthesize
            word_features: Pre-generated word-level prosodic features
            output_path: Path to save the generated audio file
            instruct: Instruction for controlling speech characteristics
            speed: Speech speed multiplier (1.0 = normal)
            
        Returns:
            True if synthesis successful, False otherwise
            
        This mode is ideal when you have specific prosodic requirements
        and want precise control over speech characteristics.
        """
        try:
            # Ensure models are loaded
            self._load_models()
            
            logger.info(f"Mode 2: Text + features to speech - {text}")
            logger.info(f"Features: {word_features[:100]}...")  # Show first 100 chars
            
            # Validate inputs
            if not text.strip() or not word_features.strip():
                logger.error("Empty text or features provided")
                return False
            
            # Prepare input for Mode 2
            prompt = self._prepare_mode2_input(text, word_features, instruct)
            
            # Generate response
            generated_text, is_truncated = self._generate_response(prompt)
            
            # Extract speech_tokens
            speech_tokens = self._extract_mode2_outputs(generated_text)
            
            if len(speech_tokens) == 0:
                logger.error("Failed to extract speech tokens")
                return False
            
            # Convert to audio
            audio_tensor = self._convert_tokens_to_audio(speech_tokens, speed)
            if audio_tensor is None:
                logger.error("Failed to convert tokens to audio")
                return False
            
            # Save audio file
            success = self._save_audio(audio_tensor, output_path)
            
            if success:
                logger.info(f"Mode 2 synthesis completed successfully!")
                return True
            else:
                logger.error("Failed to save audio file")
                return False
                
        except Exception as e:
            logger.error(f"Mode 2 synthesis failed: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up model resources and free memory
        
        Cleanup Process:
        1. Delete vLLM model instance
        2. Delete CosyVoice2 model instance
        3. Clear CUDA cache if available
        4. Reset loading flags and cached features
        
        This method should be called when the TTS instance is no longer
        needed to free up GPU memory for other processes.
        """
        if self.llm:
            del self.llm
            self.llm = None
        
        if self.cosyvoice:
            del self.cosyvoice
            self.cosyvoice = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.models_loaded = False
        logger.info("Model resources cleaned up")

def main():
    """
    Command-line interface for testing both TTS modes
    
    CLI Features:
    1. Supports both Mode 1 (text-only) and Mode 2 (text+features)
    2. Uses default configurations from single_inference_tts.py
    3. Provides comprehensive argument parsing for all parameters
    4. Includes example test cases for both modes
    
    Usage Examples:
        # Mode 1: Text only
        python unified_tts.py --mode 1 --text "Hello world" --output "output1.wav"
        
        # Mode 2: Text + features
        python unified_tts.py --mode 2 --text "Hello world" --features '[{...}]' --output "output2.wav"
    
    The CLI provides a convenient way to test the TTS functionality
    and serves as an example for integration into other scripts.
    """
    parser = argparse.ArgumentParser(description='Unified TTS Script - Two modes for text-to-speech synthesis')
    
    # Model configuration arguments
    parser.add_argument('--model_path', type=str, default='Yue-Wang/BATONTTS-1.7B',
                       help='Path to the main TTS model')
    parser.add_argument('--cosyvoice_model_dir', type=str, default='./pretrained_models/CosyVoice2-0.5B',
                       help='Directory path to CosyVoice2 model')
    parser.add_argument('--prompt_audio_path', type=str, default='./prompt.wav',
                       help='Path to prompt audio file for voice cloning')
    
    # Mode selection and input arguments
    parser.add_argument('--mode', type=int, choices=[1, 2], default=1,
                       help='TTS mode: 1=text only, 2=text+features')
    parser.add_argument('--text', type=str, default='Kids are talking by the door',
                       help='Input text to synthesize')
    parser.add_argument('--features', type=str, 
                       default='[{"word": "Kids are talking","pitch_mean": 315,"pitch_slope": 90,"energy_rms": 0.005,"energy_slope": 25,"spectral_centroid": 2650},{"word": "by the door","pitch_mean": 360,"pitch_slope": -110,"energy_rms": 0.004,"energy_slope": -30,"spectral_centroid": 2900}]',
                       help='Word-level features for Mode 2 (JSON format)')
    parser.add_argument('--instruct', type=str, default='Generate natural speech',
                       help='Instruction for controlling speech characteristics')
    
    # Output and performance arguments
    parser.add_argument('--output', type=str, default='unified_output.wav',
                       help='Output audio file path')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speech speed multiplier (1.0 = normal speed)')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7,
                       help='GPU memory utilization ratio (0.0-1.0)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use half precision for CosyVoice2')
    
    args = parser.parse_args()
    
    # Initialize TTS processor
    tts = UnifiedTTS(
        model_path=args.model_path,
        cosyvoice_model_dir=args.cosyvoice_model_dir,
        prompt_audio_path=args.prompt_audio_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        fp16=args.fp16
    )
    
    try:
        # Execute synthesis based on selected mode
        if args.mode == 1:
            print(f"Running Mode 1: Text to Speech")
            print(f"Text: {args.text}")
            print(f"Output: {args.output}")
            
            success = tts.text_to_speech(
                text=args.text,
                output_path=args.output,
                instruct=args.instruct,
                speed=args.speed
            )
            
        elif args.mode == 2:
            print(f"Running Mode 2: Text + Features to Speech")
            print(f"Text: {args.text}")
            print(f"Features: {args.features[:100]}...")  # Show first 100 chars
            print(f"Output: {args.output}")
            
            success = tts.text_features_to_speech(
                text=args.text,
                word_features=args.features,
                output_path=args.output,
                instruct=args.instruct,
                speed=args.speed
            )
        
        # Report results
        if success:
            print(f"\n✅ Synthesis completed successfully!")
            print(f"Audio file saved to: {args.output}")
        else:
            print(f"\n❌ Synthesis failed!")
            print(f"Please check the logs for error details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Synthesis interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        tts.cleanup()
        print("🧹 Resources cleaned up")

if __name__ == "__main__":
    main()