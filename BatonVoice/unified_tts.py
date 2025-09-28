#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified TTS Processor - A Comprehensive Text-to-Speech System

OVERVIEW:
This module provides a unified interface for advanced text-to-speech synthesis using 
the BATONTTS model architecture. It supports two distinct inference modes that offer 
different levels of control over speech generation.

ARCHITECTURE:
- Main Model: BATONTTS-1.7B (based on Qwen3-1.7B-instruct)
- Audio Generator: CosyVoice2-0.5B for speech token to waveform conversion
- Inference Engine: vLLM for efficient large language model inference
- Token Processing: Custom tokenization and offset handling for speech tokens

TWO INFERENCE MODES:

Mode 1 - Text-to-Speech (Automatic Feature Generation):
    Purpose: Generate speech from text input only
    Input: Text string
    Process: Text → Model generates word_features + speech_tokens → Audio
    Use Case: Simple TTS when you don't need fine-grained prosodic control
    Implementation: Matches mode1.py exactly for consistency

Mode 2 - Text+Features-to-Speech (Manual Feature Control):
    Purpose: Generate speech with explicit prosodic control
    Input: Text string + word_features (JSON)
    Process: Text + Features → Model generates speech_tokens → Audio
    Use Case: Advanced TTS with precise control over pitch, duration, etc.
    Implementation: Matches mode2.py exactly for consistency

KEY FEATURES:
- Lazy model loading for memory efficiency
- Voice cloning support via prompt audio
- Configurable speech speed control
- Comprehensive error handling and logging
- GPU memory optimization
- Tensor parallelism support for multi-GPU setups

USAGE PATTERNS:
    # Basic text-to-speech
    tts = UnifiedTTS()
    tts.text_to_speech("Hello world", "output.wav")
    
    # Advanced prosodic control
    features = '[{"word": "Hello", "pitch_mean": 300, "duration": 0.5}]'
    tts.text_features_to_speech("Hello", features, "output.wav")
    
    # Cleanup when done
    tts.cleanup()
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
sys.path.append('third-party/Matcha-TTS')
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
                 model_path: str = 'Yue-Wang/BatonTTS-1.7B',
                 cosyvoice_model_dir: str = './pretrained_models/CosyVoice2-0.5B',
                 prompt_audio_path: str = './prompt.wav',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.7,
                 fp16: bool = False):
        """
        Initialize the Unified TTS Processor
        
        Args:
            model_path: Path to the main TTS model (Yue-Wang/BatonTTS-1.7B)
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
        Load all required models for TTS processing
        
        Model Components:
        1. vLLM: Large language model for text-to-speech token generation
           - Uses BATONTTS-1.7B model
           - Handles both Mode 1 (text→features+speech) and Mode 2 (text+features→speech)
           - Optimized with tensor parallelism for multi-GPU setups
        
        2. Tokenizer: Text tokenization and encoding/decoding
           - Handles special tokens for TTS control (<custom_token_0>, <custom_token_1>, etc.)
           - Converts between text and token IDs for model input/output
           - Supports offset correction for speech token recovery
        
        3. CosyVoice2: Speech token to waveform conversion
           - Converts discrete speech tokens to continuous audio waveforms
           - Supports voice cloning using prompt audio features
           - Configurable with half precision (fp16) for memory efficiency
        
        Performance Optimizations:
        - Tensor parallelism for multi-GPU inference
        - GPU memory utilization control (default 0.7)
        - Half precision support for CosyVoice2
        - Lazy loading pattern to reduce initialization time
        
        Usage:
        This method is called automatically on first inference request.
        All components are initialized once and reused for subsequent calls.
        """
        if self.models_loaded:
            return
            
        logger.info("Loading models...")
        
        # Load main TTS model with vLLM
        # vLLM provides optimized inference for large language models with features like:
        # - Continuous batching for improved throughput
        # - PagedAttention for memory efficiency
        # - Tensor parallelism for multi-GPU scaling
        logger.info(f"Loading main model from {self.model_path}")
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2500,
        )
        
        # Load tokenizer separately
        # The tokenizer handles conversion between text and token IDs
        # It's essential for preparing model inputs and processing outputs
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load CosyVoice2 model
        # CosyVoice2 converts discrete speech tokens to continuous audio waveforms
        # It supports voice cloning using prompt audio features for consistent voice characteristics
        logger.info(f"Loading CosyVoice2 model from {self.cosyvoice_model_dir}")
        self.cosyvoice = CosyVoice2(self.cosyvoice_model_dir, fp16=self.fp16)
        self.sample_rate = self.cosyvoice.sample_rate
        
        # Preload prompt audio features for voice cloning
        # This extracts and caches features from the prompt audio file
        # These features are used to maintain consistent voice characteristics
        self._preload_prompt_features()
        
        # Configure special tokens used for TTS control
        # Special tokens structure the input/output format for different TTS modes
        self._setup_special_tokens()
        
        # Configure sampling parameters for text generation
        # These parameters control the quality and diversity of generated speech tokens
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
        4. Extract speech features
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
        Configure special tokens for TTS control and mode switching
        
        Special Tokens Purpose:
        - custom_token_0: Marks the start of speech token generation phase
          * In Mode 1: Signals transition from text processing to speech token output
          * In Mode 2: Indicates where speech tokens should be inserted
        
        - custom_token_1: Marks the end of speech token generation phase
          * Used as stop token to terminate speech token generation
          * Prevents model from generating beyond intended speech sequence
        
        - custom_token_2: Additional control token for future extensions
          * Reserved for advanced TTS control features
          * Currently used in Mode 2 for feature conditioning
        
        - eos_token_id: Standard end-of-sequence marker
          * Separates different parts of the input sequence
          * Used in prompt construction for both modes
        
        Token Format Compatibility:
        - Stores tokens as lists to match mode1.py format exactly
        - Maintains backward compatibility with dictionary format
        - Ensures consistent token handling across different inference modes
        
        Implementation Details:
        - Tokens are encoded using tokenizer's call method (matching mode1.py)
        - Each token is stored as a list of token IDs (usually single element)
        - Dictionary format provides easy access for legacy code
        """
        # Store as lists (matching mode1.py format)
        # This ensures exact compatibility with the original mode1.py implementation
        self.custom_token_0_ids = self.tokenizer('<custom_token_0>').input_ids
        self.custom_token_1_ids = self.tokenizer('<custom_token_1>').input_ids
        self.custom_token_2_ids = self.tokenizer('<custom_token_2>').input_ids
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Maintain dictionary format for backward compatibility
        # This allows existing code to continue using self.special_tokens['token_name']
        self.special_tokens = {
            'custom_token_0': self.custom_token_0_ids[0],
            'custom_token_1': self.custom_token_1_ids[0], 
            'custom_token_2': self.custom_token_2_ids[0],
            'eos_token_id': self.eos_token_id
        }
        logger.info("Special tokens configured")
    
    def _setup_sampling_params(self):
        """
        Configure sampling parameters for text generation (matches mode1.py exactly)
        
        Sampling Strategy:
        The parameters are carefully tuned for TTS generation to balance quality,
        diversity, and computational efficiency. These settings match mode1.py
        to ensure consistent behavior across different inference implementations.
        
        Parameter Details:
        - temperature (0.8): Controls randomness in token selection
          * Lower values (0.1-0.5): More deterministic, consistent output
          * Higher values (0.8-1.0): More creative, diverse output
          * 0.6 provides optimal balance for natural speech variation
        
        - top_p (1.0): Nucleus sampling threshold for quality control
          * Considers all tokens in the probability distribution
          * Allows full model expressiveness for speech token generation
          * Maintains coherent speech token sequences
        
        - max_tokens (2048): Maximum tokens to generate in one pass
          * Accommodates long speech sequences (up to ~20 seconds of audio)
          * Matches mode1.py configuration for consistency
          * Prevents runaway generation while allowing sufficient length
        
        - stop_token_ids: Tokens that terminate generation
          * Uses custom_token_1 to mark end of speech token sequence
          * Ensures clean separation between speech tokens and other content
          * Prevents model from generating beyond intended speech boundary
        
        - repetition_penalty (1.1): Reduces repetitive output patterns
          * Slightly penalizes recently generated tokens
          * Improves speech naturalness by reducing monotonous patterns
          * Maintains speech quality while encouraging variation
        
        Usage:
        These parameters are used by vLLM during text generation for both
        Mode 1 (text→speech tokens) and Mode 2 (text+features→speech tokens).
        """

        self.sampling_params = SamplingParams(
            temperature=0.6,      # Balanced randomness for natural speech variation
            top_p=1,           # Full nucleus sampling for maximum expressiveness
            max_tokens=2048,     # Maximum sequence length (matches mode1.py)
            stop_token_ids=[self.eos_token_id],  # Stop at end of speech tokens
            repetition_penalty=1.1,  # Slight penalty to reduce repetitive patterns
        )
        logger.info("Sampling parameters configured")
    
    def _prepare_mode1_input(self, text: str) -> str:
        """
        Prepare input prompt for Mode 1 inference (matches mode1.py exactly)
        
        Mode 1 Workflow:
        Mode 1 performs end-to-end text-to-speech conversion in a single pass.
        The model generates both intermediate features and final speech tokens
        from the input text, making it suitable for scenarios where no specific
        voice characteristics are required.
        
        Input Format Structure: <custom_token_0> + text + <custom_token_1>
        
        Component Breakdown:
        1. <custom_token_0>: Speech generation trigger
           - Signals model to begin speech token generation
           - Marks the start of the input sequence
           - Model generates speech tokens after this marker
        
        2. text: The input text to be converted to speech
           - Can be any natural language text
           - Processed by the model's text understanding capabilities
           - No length restrictions beyond model's context window
        
        3. <custom_token_1>: End-of-text marker
           - Signals the end of text input to the model
           - Separates text content from control tokens
           - Essential for proper model interpretation
        
        Processing Steps:
        1. Tokenize input text using the model's tokenizer
        2. Construct sequence: custom_token_0 + text_tokens + custom_token_1
        3. Decode token sequence back to text for vLLM processing
        
        Compatibility:
        This implementation exactly matches mode1.py's prepare_inference_prompt
        method to ensure identical behavior and output quality.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Formatted prompt string for vLLM inference
        """
        # Encode text to tokens (matching mode1.py implementation)
        # Using return_tensors="pt" and converting to list for consistency
        text_tokens = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        
        # Construct input sequence: <custom_token_0> + text + <custom_token_1>
        # This exactly matches mode1.py's prepare_inference_prompt method
        # The sequence structure is critical for proper model interpretation
        input_sequence = (
            self.custom_token_0_ids +        # Speech generation trigger
            text_tokens +                    # Original text content
            self.custom_token_1_ids          # End-of-text marker
        )
        
        # Convert token sequence back to text for vLLM processing
        # vLLM expects text input, so we decode the token sequence
        prompt = self.tokenizer.decode(input_sequence, skip_special_tokens=False)
        return prompt
    def _prepare_mode2_input(self, text: str, word_features: str) -> str:
        """
        Prepare input prompt for Mode 2 inference (matches mode2.py exactly)
        
        Mode 2 Input Format (from mode2.py):
        <custom_token_0> + text + <custom_token_1> + generated_features + <custom_token_2>
        
        Args:
            text: Input text to synthesize
            word_features: JSON string containing word-level prosodic features (called generated_features in mode2.py)
            
        Returns:
            Formatted prompt string for vLLM inference
            
        Implementation Logic:
        1. Tokenize input text and generated_features using the model's tokenizer
        2. Construct sequence: custom_token_0 + text_tokens + custom_token_1 + features_tokens + custom_token_2
        3. Decode back to text format for vLLM processing
        4. Return formatted prompt
        
        The model will generate speech_tokens ending with eos_token_id
        This format matches the exact implementation in mode2.py
        """
        # Encode text and generated_features to tokens (matching mode2.py implementation)
        text_tokens = self.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        features_tokens = self.tokenizer(word_features, return_tensors="pt").input_ids[0].tolist()
        
        # Construct input sequence: custom_token_0 + text + custom_token_1 + generated_features + custom_token_2
        # This exactly matches mode2.py's prepare_mode2_inference_prompt method
        input_sequence = (
            [self.special_tokens['custom_token_0']] +
            text_tokens +
            [self.special_tokens['custom_token_1']] +
            features_tokens +
            [self.special_tokens['custom_token_2']]
        )
        
        # Convert token sequence back to text for vLLM
        prompt = self.tokenizer.decode(input_sequence, skip_special_tokens=False)
        return prompt
    
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
    
    def _extract_mode1_features(self, generated_text: str, original_prompt: str) -> str:
        """
        Extract generated features from Mode 1 output
        
        Mode 1 Output Format:
        According to the sequence format: custom_token_0 + text + custom_token_1 + generated_features + custom_token_2 + features + eos_token_id
        
        This method extracts the generated_features part from the model output.
        The generated_features are located between custom_token_1 and custom_token_2.
        
        Args:
            generated_text: Model generated text
            original_prompt: Original input prompt
            
        Returns:
            JSON string containing the generated features
        """
        try:
            # Find the features section between custom_token_1 and custom_token_2
            custom_token_1_str = self.tokenizer.decode(self.custom_token_1_ids, skip_special_tokens=False)
            custom_token_2_str = self.tokenizer.decode(self.custom_token_2_ids, skip_special_tokens=False)
            
            # Look for features in the generated text
            full_text = original_prompt + generated_text
            
            # Find the start of features (after custom_token_1)
            start_idx = full_text.find(custom_token_1_str)
            if start_idx == -1:
                logger.warning("Could not find custom_token_1 in generated text")
                return ""
            
            start_idx += len(custom_token_1_str)
            
            # Find the end of features (before custom_token_2)
            end_idx = full_text.find(custom_token_2_str, start_idx)
            if end_idx == -1:
                logger.warning("Could not find custom_token_2 in generated text")
                # If no custom_token_2 found, try to find eos_token
                eos_str = self.tokenizer.decode([self.eos_token_id], skip_special_tokens=False)
                end_idx = full_text.find(eos_str, start_idx)
                if end_idx == -1:
                    # Take the rest of the text
                    end_idx = len(full_text)
            
            # Extract features text
            features_text = full_text[start_idx:end_idx].strip()
            
            if features_text:
                logger.info(f"Extracted features: {features_text[:100]}...")
                return features_text
            else:
                logger.warning("No features found in generated text")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting mode1 features: {e}")
            return ""

    def _extract_mode1_outputs(self, generated_text: str, original_prompt: str) -> List[int]:
        """
        Extract speech tokens from Mode 1 generation output (matches mode1.py exactly)
        
        Output Processing Workflow:
        Mode 1 generates a sequence containing both the original prompt and new speech tokens.
        This method extracts only the speech tokens that represent the audio content,
        filtering out text tokens and control tokens to prepare for audio synthesis.
        
        Token Offset Mechanism:
        The model uses a token offset system to distinguish between different token types:
        - Text tokens: IDs in lower range (0 to ~151669)
        - Speech tokens: IDs in higher range (151669+ with offset applied)
        - The offset (151669 + 100) is used to map speech tokens back to their original values
        
        Processing Steps:
        1. Merge prompt and generated text to get complete sequence
           - Prompt contains: text + [EOS] + <custom_token_0>
           - Generated text contains: speech_tokens + <custom_token_1>
           - Full sequence represents the complete model output
        
        2. Encode merged text to token IDs
           - Converts text representation back to numerical token IDs
           - Preserves all tokens including special tokens and speech tokens
           - Uses same tokenizer as model for consistency
        
        3. Apply offset correction to recover speech tokens
           - Speech tokens are stored with offset (151669 + 100) added
           - Subtracting offset recovers original speech token values
           - Only tokens above offset threshold are considered speech tokens
        
        4. Filter and return speech tokens
           - Removes text tokens, control tokens, and invalid tokens
           - Returns clean list of speech token IDs for audio synthesis
        
        Compatibility:
        This implementation exactly matches mode1.py's extract_output_audio_tokens
        method to ensure identical behavior, output format, and audio quality.
        
        Args:
            generated_text: Model generated text
            original_prompt: Original input prompt
            
        Returns:
            List of speech token integers (offset corrected)
        """
        try:
            # Step 1: Merge original prompt and generated text (matching mode1.py exactly)
            # This gives us the complete sequence that the model processed
            full_text = original_prompt + generated_text
            
            if not full_text.strip():
                return []
            
            # Step 2: Encode text to token IDs (matching mode1.py tokenization)
            # Convert the text back to numerical token IDs for processing
            token_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            
            # Step 3: Apply offset correction to recover original speech tokens
            # The model stores speech tokens with an offset of (151669 + 100)
            # We subtract this offset to get the original speech token values
            speech_tokens = []
            offset = 151669 + 100  # Exact same offset as mode1.py
            
            for token_id in token_ids:
                if token_id >= offset:
                    # Recover original speech token by subtracting offset
                    original_token = token_id - offset
                    speech_tokens.append(original_token)
            
            logger.info(f"Extracted speech_tokens count: {len(speech_tokens)}")
            return speech_tokens
            
        except Exception as e:
            logger.error(f"Error extracting mode1 outputs: {e}")
            return []
    
    def _extract_mode2_outputs(self, generated_text: str, original_prompt: str) -> List[int]:
        """
        Extract speech_tokens from Mode 2 output (matches mode2.py exactly)
        
        Extraction Logic (from mode2.py extract_speech_tokens_from_mode2_output):
        1. Encode generated text to token IDs using tokenizer
        2. Apply offset correction to recover original speech token values
        3. Filter tokens to ensure they're in valid speech token range
        
        Args:
            generated_text: Model generated text
            original_prompt: Original input prompt (unused in mode2.py, kept for consistency)
            
        Returns:
            List of speech token integers (offset corrected)
            
        This exactly matches the implementation in mode2.py's extract_speech_tokens_from_mode2_output method
        """
        try:
            if not generated_text.strip():
                return []
            
            # Encode generated text to token IDs (matching mode2.py)
            token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
            
            # Recover original speech_token values (subtract offset)
            speech_tokens = []
            offset = 151669 + 100  # Consistent with mode2.py implementation
            
            for token_id in token_ids:
                if token_id >= offset:
                    original_token = token_id - offset
                    speech_tokens.append(original_token)
            
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
        # Ensure output directory exists
        try:      
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception:
            pass
            
        try:
            # Save audio file
            torchaudio.save(output_path, audio_tensor, self.sample_rate)
            logger.info(f"Audio saved successfully to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def text_to_speech(self, text: str, output_path: str, speed: float = 1.0) -> bool:
        """
        Mode 1: End-to-End Text-to-Speech Conversion (matches mode1.py exactly)
        
        Overview:
        Mode 1 provides a complete text-to-speech pipeline that generates speech directly
        from input text without requiring external features or prompt audio. The model
        internally generates both acoustic features and speech tokens in a single pass,
        making it ideal for general-purpose TTS applications.
        
        Key Advantages:
        - Single-pass inference for optimal performance
        - No dependency on external features or prompt audio
        - Consistent voice characteristics across different texts
        - Simplified API for straightforward TTS tasks
        - Direct compatibility with mode1.py implementation
        
        Technical Workflow:
        1. Model Initialization: Ensures all components are loaded and ready
           - vLLM engine for language model inference
           - Tokenizer for text processing and token conversion
           - CosyVoice2 for speech synthesis from tokens
        
        2. Input Validation and Preparation: Formats text according to Mode 1 protocol
           - Validates input text for proper content
           - Constructs prompt: text + [EOS] + <custom_token_0>
           - This format signals the model to generate speech tokens directly
        
        3. Speech Token Generation: Uses vLLM for efficient model inference
           - Applies optimized sampling parameters for quality control
           - Generates speech token sequence ending with <custom_token_1>
           - Produces discrete tokens representing audio content
        
        4. Token Extraction and Processing: Isolates speech tokens from output
           - Filters out input text and control tokens
           - Applies offset correction (151669 + 100) to recover original values
           - Validates token sequence for successful audio synthesis
        
        5. Audio Synthesis: Converts tokens to waveform using CosyVoice2
           - Transforms discrete speech tokens to continuous audio signal
           - Applies speed adjustment if specified (default 1.0 = normal)
           - Generates high-quality audio with natural voice characteristics
        
        6. File Output: Saves audio to specified path with proper formatting
           - Supports common audio formats (WAV, MP3, etc.)
           - Maintains audio quality and sample rate consistency
           - Provides success/failure feedback for error handling
        
        Error Handling and Validation:
        - Comprehensive input validation for text content
        - Model loading failure detection and recovery
        - Token generation validation and error reporting
        - Audio synthesis error handling with detailed logging
        - File I/O error management with clear feedback
        
        Performance Optimizations:
        - Lazy model loading to reduce initialization overhead
        - Efficient token processing to minimize memory usage
        - Optimized sampling parameters for quality-speed balance
        - Streamlined audio conversion pipeline
        
        Compatibility:
        This implementation exactly matches mode1.py's behavior to ensure:
        - Identical input format and processing logic
        - Same token extraction and offset correction
        - Consistent audio quality and characteristics
        - Compatible output format and file structure
        
        Args:
            text: Input text to convert to speech
                 - Supports natural language text in multiple languages
                 - Handles punctuation, numbers, and special characters
                 - No strict length limitations (within model context window)
                 - Automatically processes formatting and normalization
            
            output_path: File path where generated audio will be saved
                        - Supports various audio formats (WAV, MP3, FLAC, etc.)
                        - Creates parent directories if they don't exist
                        - Overwrites existing files at the specified path
                        - Should include appropriate file extension
            
            speed: Speech speed multiplier for tempo control
                  - 1.0 = normal speaking speed (default)
                  - < 1.0 = slower speech (e.g., 0.8 for 20% slower)
                  - > 1.0 = faster speech (e.g., 1.2 for 20% faster)
                  - Applied during audio synthesis phase
            
        Returns:
            bool: Success status of the TTS operation
                 - True: Audio successfully generated and saved
                 - False: Operation failed (check logs for details)
                 
        Raises:
            Exception: Caught and logged, returns False for graceful handling
            
        Usage Examples:
            # Basic text-to-speech
            success = tts.text_to_speech("Hello world", "output.wav")
            
            # With custom speed
            success = tts.text_to_speech(
                "Welcome to our service", 
                "welcome.wav",
                speed=0.9
            )
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
            prompt = self._prepare_mode1_input(text)
            
            # Generate response
            generated_text, is_truncated = self._generate_response(prompt)
            
            # Extract speech_tokens (Mode 1 now only returns speech_tokens, matching mode1.py)
            speech_tokens = self._extract_mode1_outputs(generated_text, prompt)
            
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

    def text_to_speech_with_features(self, text: str, output_path: str, speed: float = 1.0) -> Tuple[bool, str]:
        """
        Mode 1: End-to-End Text-to-Speech Conversion with Features Extraction
        
        This method extends the standard text_to_speech functionality to also return
        the generated features that the model produces internally. This is useful
        for understanding the prosodic characteristics the model assigns to the text.
        
        Args:
            text: Input text to convert to speech
            output_path: File path where generated audio will be saved
            speed: Speech speed multiplier for tempo control
            
        Returns:
            Tuple[bool, str]: (success_status, generated_features_json)
            - success_status: True if audio generation succeeded, False otherwise
            - generated_features_json: JSON string of generated features, empty if failed
        """
        try:
            # Ensure models are loaded
            self._load_models()
            
            logger.info(f"Mode 1 with features: Text to speech - {text}")
            
            # Validate input
            if not text.strip():
                logger.error("Empty text provided")
                return False, ""
            
            # Prepare input for Mode 1
            prompt = self._prepare_mode1_input(text)
            
            # Generate response
            generated_text, is_truncated = self._generate_response(prompt)
            
            # Extract features from the generated text
            generated_features = self._extract_mode1_features(generated_text, prompt)
            
            # Extract speech_tokens (Mode 1 now only returns speech_tokens, matching mode1.py)
            speech_tokens = self._extract_mode1_outputs(generated_text, prompt)
            
            if len(speech_tokens) == 0:
                logger.error("Failed to extract speech tokens")
                return False, generated_features
            
            # Convert to audio
            audio_tensor = self._convert_tokens_to_audio(speech_tokens, speed)
            if audio_tensor is None:
                logger.error("Failed to convert tokens to audio")
                return False, generated_features
            
            # Save audio file
            success = self._save_audio(audio_tensor, output_path)
            
            if success:
                logger.info(f"Mode 1 synthesis with features completed successfully!")
                return True, generated_features
            else:
                logger.error("Failed to save audio file")
                return False, generated_features
                
        except Exception as e:
            logger.error(f"Mode 1 synthesis with features failed: {e}")
            return False, ""
    
    def text_features_to_speech(self, text: str, word_features: str, output_path: str, 
                               speed: float = 1.0) -> bool:
        """
        Mode 2: Convert text + features to speech (uses provided features)
        
        Process Flow:
        1. Load models if not already loaded
        2. Prepare Mode 2 input format with text and features
        3. Generate model response to get speech_tokens only
        4. Extract speech_tokens from the generated output
        5. Convert speech_tokens to audio waveform
        6. Save audio to specified output path
        
        Args:
            text: Input text to synthesize
            word_features: Pre-generated word-level prosodic features
            output_path: Path to save the generated audio file
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
            prompt = self._prepare_mode2_input(text, word_features)
            
            # Generate response
            generated_text, is_truncated = self._generate_response(prompt)
            
            # Extract speech_tokens (Mode 2 now matches mode2.py implementation)
            speech_tokens = self._extract_mode2_outputs(generated_text, prompt)
            
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
    parser.add_argument('--model_path', type=str, default='Yue-Wang/BatonTTS-1.7B',
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
            
            success, features = tts.text_to_speech_with_features(
                text=args.text,
                output_path=args.output,
                speed=args.speed
            )
            
            if success:
                print(f"Features: {features}")
            
        elif args.mode == 2:
            print(f"Running Mode 2: Text + Features to Speech")
            print(f"Text: {args.text}")
            print(f"Features: {args.features[:100]}...")  # Show first 100 chars
            print(f"Output: {args.output}")
            
            success = tts.text_features_to_speech(
                text=args.text,
                word_features=args.features,
                output_path=args.output,
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
