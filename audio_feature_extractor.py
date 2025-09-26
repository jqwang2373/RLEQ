#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Feature Extractor Script

Main Functions:
1. Extract word-level features from single audio files
2. Use Whisper for speech recognition
3. Use Wav2Vec2 for word-level alignment with Triton optimization
4. Support English audio processing

Usage:
    extractor = AudioFeatureExtractor()
    features = extractor.extract_features("path/to/audio.wav")

Basic Implementation Logic:
1. Load audio file using librosa
2. Use Whisper for speech transcription
3. Use Wav2Vec2 for word-level alignment (Triton optimized)
4. Extract audio features based on word-level timestamps
5. Return feature dictionary with word segments and audio features
"""

import os
import json
import warnings
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import librosa
import soundfile as sf
import parselmouth

# Check Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, will use original implementation")

# HuggingFace libraries
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)

# ===== Configuration Constants =====
# Audio processing parameters
SAMPLE_RATE = 16000  # Standard sample rate for Whisper and Wav2Vec2
MAX_DURATION = 30    # Maximum audio segment duration (seconds)

# Default model paths for English
DEFAULT_WHISPER_MODEL = "/apdcephfs_sh3/share_300730042/hunyuan/wangyue/model/whisper-large-v3"
DEFAULT_ALIGN_MODEL = "/apdcephfs_sh3/share_300730042/hunyuan/wangyue/model/wav2vec2-large-960h-lv60-self"

# ===== Data Structure Definitions =====
@dataclass
class WordSegment:
    """Word-level segment information"""
    word: str                    # Word text
    start: Optional[float]       # Start time (seconds)
    end: Optional[float]         # End time (seconds)
    score: Optional[float]       # Confidence score


@dataclass
class AlignedSegment:
    """Aligned sentence segment"""
    text: str                    # Sentence text
    start: Optional[float]       # Start time (seconds)
    end: Optional[float]         # End time (seconds)
    words: List[WordSegment]     # Word-level information list


class AudioFeatureExtractor:
    """
    Audio Feature Extractor Class
    
    Main Functions:
    1. Load and process single audio files
    2. Extract word-level features using Triton optimization
    3. Support English audio processing
    
    Usage:
        extractor = AudioFeatureExtractor()
        features = extractor.extract_features("audio.wav")
    """
    
    def __init__(self, 
                 whisper_model: str = DEFAULT_WHISPER_MODEL,
                 align_model: str = DEFAULT_ALIGN_MODEL,
                 device: str = "auto",
                 merge_threshold: float = 0.5):
        """
        Initialize Audio Feature Extractor
        
        Args:
            whisper_model: Path to Whisper model for speech recognition
            align_model: Path to Wav2Vec2 model for word alignment
            device: Computing device ("auto", "cpu", "cuda")
            merge_threshold: Word merging threshold (seconds)
        
        Implementation Logic:
        1. Set up device configuration
        2. Load Whisper and Wav2Vec2 models
        3. Initialize vocabulary for alignment
        4. Ensure Triton optimization is available
        """
        self.merge_threshold = merge_threshold
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Initializing Audio Feature Extractor...")
        print(f"   Device: {self.device}")
        print(f"   Whisper Model: {whisper_model}")
        print(f"   Align Model: {align_model}")
        print(f"   Word Merge Threshold: {merge_threshold}s")
        print(f"   Triton Available: {TRITON_AVAILABLE}")
        
        # Ensure Triton is available for optimization
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required but not available. Please install triton.")
        
        if self.device != "cuda":
            raise RuntimeError("Triton optimization requires CUDA device.")
        
        # Load models
        self._load_models(whisper_model, align_model)
        
        print("✅ Audio Feature Extractor initialized successfully")
    
    def _load_models(self, whisper_model: str, align_model: str):
        """
        Load Whisper and Wav2Vec2 models
        
        Args:
            whisper_model: Path to Whisper model
            align_model: Path to Wav2Vec2 alignment model
        
        Implementation Logic:
        1. Load Whisper model for speech recognition
        2. Load Wav2Vec2 model for word-level alignment
        3. Set models to evaluation mode
        4. Build character-level vocabulary dictionary
        """
        try:
            # Load Whisper model
            print("📥 Loading Whisper model...")
            self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
            self.whisper_model.to(self.device)
            self.whisper_model.eval()
            
            # Load Wav2Vec2 alignment model
            print(f"📥 Loading Wav2Vec2 alignment model...")
            self.align_processor = Wav2Vec2Processor.from_pretrained(align_model)
            self.align_model = Wav2Vec2ForCTC.from_pretrained(align_model)
            self.align_model.to(self.device)
            self.align_model.eval()
            
            # Build character-level vocabulary dictionary
            labels = self.align_processor.tokenizer.get_vocab()
            # Create character to ID mapping, convert all characters to lowercase
            self.vocab = {char.lower(): code for char, code in labels.items()}
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            
            print("✅ Models loaded successfully")
            print(f"Vocabulary size: {len(self.vocab)}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def load_audio_file(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        Load audio file from given path
        
        Args:
            audio_path: Path to audio file (absolute or relative)
        
        Returns:
            tuple: (audio_array, sampling_rate)
        
        Implementation Logic:
        1. Check if path exists
        2. Load audio using librosa
        3. Return audio array and sampling rate
        4. Handle errors gracefully
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio file
            audio_array, sampling_rate = librosa.load(audio_path, sr=None)
            
            print(f"📁 Loaded audio: {audio_path}")
            print(f"   Duration: {len(audio_array)/sampling_rate:.2f}s")
            print(f"   Sample Rate: {sampling_rate}Hz")
            
            return audio_array, sampling_rate
            
        except Exception as e:
            print(f"❌ Failed to load audio file: {audio_path}, Error: {e}")
            raise
    
    def transcribe_audio(self, audio: np.ndarray, sampling_rate: int) -> str:
        """
        Transcribe audio using Whisper model
        
        Args:
            audio: Audio array
            sampling_rate: Sampling rate
            
        Returns:
            Transcribed text
        
        Implementation Logic:
        1. Resample audio to 16kHz if needed
        2. Preprocess audio for Whisper
        3. Generate transcription using Whisper model
        4. Return cleaned transcription text
        """
        try:
            # Resample to 16kHz
            if sampling_rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)
            
            # Preprocess audio
            inputs = self.whisper_processor(
                audio, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(inputs["input_features"])
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
            
            print(f"🎯 Transcription: {transcription}")
            return transcription.strip()
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""
    
    def get_word_timestamps(self, audio: np.ndarray, text: str) -> List[AlignedSegment]:
        """
        Get word-level timestamps using Wav2Vec2 forced alignment
        
        Args:
            audio: Audio array
            text: Transcribed text
            
        Returns:
            List of aligned segments with word-level timestamps
        
        Implementation Logic:
        1. Preprocess text for alignment
        2. Use Wav2Vec2 model for CTC alignment
        3. Calculate word-level boundaries using Triton optimization
        4. Return aligned segments with timestamps
        """
        try:
            print("🔄 Starting Wav2Vec2 forced alignment...")
            
            # Preprocess text
            clean_transcript = self._preprocess_text(text)
            if not clean_transcript:
                print("Warning: Text preprocessing resulted in empty string")
                return [AlignedSegment(
                    text=text,
                    start=0.0,
                    end=len(audio) / SAMPLE_RATE,
                    words=[WordSegment(
                        word=text,
                        start=0.0,
                        end=len(audio) / SAMPLE_RATE,
                        score=0.0
                    )]
                )]
            
            # Preprocess audio
            inputs = self.align_processor(
                audio, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # Get model output
            with torch.no_grad():
                logits = self.align_model(inputs.input_values).logits
            
            # Convert to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            emission = log_probs[0]  # Remove batch dimension
            
            # Perform CTC alignment with Triton optimization
            aligned_segments = self._ctc_align_triton(emission, clean_transcript, audio)
            
            print("✅ Forced alignment completed")
            return aligned_segments
            
        except Exception as e:
            print(f"❌ Word-level alignment failed: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing characters not in vocabulary
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        
        Implementation Logic:
        1. Convert to lowercase
        2. Replace spaces with | (Wav2Vec2 convention)
        3. Keep only characters in vocabulary
        4. Replace unknown characters with wildcards
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Replace spaces with | (Wav2Vec2 convention for English)
        text = text.replace(" ", "|")
        
        # Keep only characters in vocabulary
        clean_chars = []
        for char in text:
            if char in self.vocab:
                clean_chars.append(char)
            else:
                # Replace unknown characters with wildcards
                clean_chars.append("*")
        
        return "".join(clean_chars)
    
    def _ctc_align_triton(self, emission: torch.Tensor, transcript: str, audio: np.ndarray) -> List[AlignedSegment]:
        """
        Perform CTC forced alignment using Triton optimization
        
        Args:
            emission: Model output emission probabilities
            transcript: Cleaned transcript text
            audio: Original audio array
            
        Returns:
            List of aligned segments
        
        Implementation Logic:
        1. Convert text to token IDs
        2. Build trellis using Triton-optimized kernels
        3. Backtrack optimal path
        4. Merge repeated characters
        5. Generate word alignments with timestamps
        """
        # Convert text to token IDs
        tokens = [self.vocab.get(char, self.vocab.get("[UNK]", 0)) for char in transcript]
        
        # Get blank token ID
        blank_id = self.vocab.get("[PAD]", 0)
        if "[PAD]" not in self.vocab:
            blank_id = self.vocab.get("<pad>", 0)
        
        # Build trellis using Triton optimization
            trellis = self._get_trellis(emission, tokens, blank_id)
        
        # Backtrack optimal path
        path = self._backtrack(trellis, emission, tokens, blank_id)
        
        if path is None:
            print("Warning: CTC alignment failed, returning original timestamps")
            return [AlignedSegment(
                text=transcript.replace("|", " "),
                start=0.0,
                end=len(audio) / SAMPLE_RATE,
                words=[WordSegment(
                    word=transcript.replace("|", " "),
                    start=0.0,
                    end=len(audio) / SAMPLE_RATE,
                    score=0.0
                )]
            )]
        
        # Merge repeated characters
        char_segments = self._merge_repeats(path, transcript)
        
        # Convert to timestamps
        duration = len(audio) / SAMPLE_RATE
        time_ratio = duration / (emission.size(0) - 1)
        
        # Generate word-level alignments
        words = self._generate_word_alignments(char_segments, transcript, time_ratio)
        
        return [AlignedSegment(
            text=transcript.replace("|", " "),
            start=words[0].start if words else 0.0,
            end=words[-1].end if words else duration,
            words=words
        )]
    
    @staticmethod
    @triton.jit
    def _trellis_row_kernel_optimized(
        # Pointers
        trellis_t_ptr,
        trellis_tm1_ptr,
        emission_t_ptr,
        tokens_ptr,
        # Scalar arguments
        num_tokens,
        blank_emit: tl.float32,
        t,
        # Tensor strides
        trellis_stride_n,
        # Meta-parameters
        BLOCK_SIZE_N: tl.constexpr,
    ):
        """
        Triton-optimized kernel for trellis row computation
        
        This kernel computes one row of the CTC trellis matrix in parallel,
        significantly speeding up the forced alignment process.
        
        Implementation Logic:
        1. Calculate parallel indices for token positions
        2. Load previous trellis values (stay and advance paths)
        3. Load emission probabilities for current tokens
        4. Compute path scores using numerically stable logsumexp
        5. Store results back to trellis
        """
        # Calculate parallel indices starting from j=1
        pid = tl.program_id(axis=0)
        offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) + 1

        # Ensure indices are within bounds
        mask = (offs_n < num_tokens) & (offs_n <= t)

        # Load trellis[t-1, j] (stay path)
        prev_stay_ptr = trellis_tm1_ptr + offs_n * trellis_stride_n
        prev_stay_score = tl.load(prev_stay_ptr, mask=mask, other=float('-inf'))

        # Load trellis[t-1, j-1] (advance path)
        prev_advance_ptr = trellis_tm1_ptr + (offs_n - 1) * trellis_stride_n
        prev_advance_score = tl.load(prev_advance_ptr, mask=mask, other=float('-inf'))

        # Load emission[t, tokens[j]]
        tokens_j = tl.load(tokens_ptr + offs_n, mask=mask, other=0)
        emission_token = tl.load(emission_t_ptr + tokens_j, mask=mask, other=float('-inf'))

        # Calculate path scores
        stay_score = prev_stay_score + blank_emit
        advance_score = prev_advance_score + emission_token

        # Numerically stable logsumexp
        max_val = tl.maximum(stay_score, advance_score)
        min_val = tl.minimum(stay_score, advance_score)
        log_sum = tl.where(
            max_val > float('-inf'),
            max_val + tl.log(1.0 + tl.exp(min_val - max_val)),
            float('-inf')
        )

        # Store results
        trellis_t_ptr_j = trellis_t_ptr + offs_n * trellis_stride_n
        tl.store(trellis_t_ptr_j, log_sum, mask=mask)
    
    def _get_trellis(self, emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        """
        Build CTC alignment trellis using Triton optimization
        
        Args:
            emission: Emission probability matrix [T, V]
            tokens: Token ID list
            blank_id: Blank token ID
            
        Returns:
            Trellis matrix [T, N]
        """
        # Use Triton optimized version if available and on CUDA device
        if TRITON_AVAILABLE and torch.cuda.is_available() and emission.device.type == 'cuda':
            return self._get_trellis_triton(emission, tokens, blank_id)
        else:
            # Fallback to original implementation
            print("Warning: Triton not enabled or CUDA unavailable, falling back to original CTC alignment implementation")
            return self._get_trellis_original(emission, tokens, blank_id)
    
    def _get_trellis_original(self, emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        """
        Original trellis construction implementation (as fallback)
        """
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        
        # Initialize trellis
        trellis = torch.full((num_frame, num_tokens), float('-inf'), device=emission.device)
        trellis[0, 0] = emission[0, blank_id]
        
        # Fill first row
        for t in range(1, num_frame):
            trellis[t, 0] = trellis[t-1, 0] + emission[t, blank_id]
        
        # Fill trellis
        for t in range(1, num_frame):
            for j in range(1, min(num_tokens, t + 1)):
                # Stay at current token (insert blank)
                stay_score = trellis[t-1, j] + emission[t, blank_id]
                
                # Advance to next token
                advance_score = trellis[t-1, j-1] + emission[t, tokens[j]]
                
                trellis[t, j] = torch.logsumexp(torch.stack([stay_score, advance_score]), dim=0)
        
        return trellis
    
    def _get_trellis_triton(self, emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
        """
        Triton-optimized trellis construction - optimized version
        """
        assert emission.is_cuda, "Input tensor must be on CUDA device"
        
        num_frame, vocab_size = emission.size()
        tokens_tensor = torch.tensor(tokens, device=emission.device, dtype=torch.long)
        num_tokens = len(tokens_tensor)

        # --- Optimization 3: Ensure memory contiguity ---
        # Fix: Remove memory_format parameter, use .contiguous() method to ensure memory contiguity
        trellis = torch.full((num_frame, num_tokens), float('-inf'),
                            device=emission.device,
                            dtype=torch.float32).contiguous()

        if num_tokens == 0:
            return trellis

        # --- Optimization 4: Use vectorized cumsum to initialize first column ---
        # Calculate cumulative blank probabilities from t=0
        # Note: For consistency with original logic, use emission directly instead of log_softmax
        trellis[:, 0] = emission[:, blank_id].cumsum(dim=0)

        # --- Optimization 2: Dynamic block size ---
        # Adapt to different token sequence lengths, improve GPU utilization
        BLOCK_SIZE_N = min(1024, triton.next_power_of_2(num_tokens)) if num_tokens > 1 else 1

        # Main loop
        for t in range(1, num_frame):
            # --- Optimization 1: Scalar broadcasting ---
            # Pass blank emission as scalar to avoid redundant loading
            blank_emit = emission[t, blank_id].item()

            # Launch grid, only compute j > 0 part
            if num_tokens > 1:
                grid = lambda meta: (triton.cdiv(num_tokens - 1, meta['BLOCK_SIZE_N']),)
                self._trellis_row_kernel_optimized[grid](
                    trellis_t_ptr=trellis[t],
                    trellis_tm1_ptr=trellis[t-1],
                    emission_t_ptr=emission[t],
                    tokens_ptr=tokens_tensor,
                    num_tokens=num_tokens,
                    blank_emit=blank_emit,
                    t=t,
                    trellis_stride_n=trellis.stride(1),
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                )
                
        return trellis
    
    def _backtrack(self, trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int) -> Optional[List]:
        """
        Backtrack through trellis to find optimal alignment path
        
        Args:
            trellis: Completed trellis matrix
            emission: Emission probabilities
            tokens: Token ID list
            blank_id: Blank token ID
            
        Returns:
            Optimal path through trellis
        
        Implementation Logic:
        1. Start from final position in trellis
        2. Trace back through highest probability path
        3. Record token positions and timestamps
        4. Return path for character merging
        """
        # Implementation details would be similar to the original _backtrack method
        # This is a simplified version for demonstration
        try:
            num_frame, num_tokens = trellis.size()
            
            # Start from the end
            t, j = num_frame - 1, num_tokens - 1
            path = []
            
            while t >= 0 and j >= 0:
                path.append((t, j))
                
                if j == 0:
                    t -= 1
                elif t == 0:
                    j -= 1
                else:
                    # Choose path with higher probability
                    stay_score = trellis[t-1, j] + emission[t, blank_id]
                    advance_score = trellis[t-1, j-1] + emission[t, tokens[j]]
                    
                    if stay_score > advance_score:
                        t -= 1
                    else:
                        t -= 1
                        j -= 1
            
            return list(reversed(path))
            
        except Exception as e:
            print(f"Backtracking failed: {e}")
            return None
    
    def _merge_repeats(self, path: List, transcript: str) -> List:
        """
        Merge repeated characters in alignment path
        
        Args:
            path: Alignment path from backtracking
            transcript: Original transcript
            
        Returns:
            List of character segments with merged repeats
        
        Implementation Logic:
        1. Group consecutive identical characters
        2. Calculate start and end frames for each character
        3. Return character segments for word boundary detection
        """
        if not path:
            return []
        
        char_segments = []
        current_char = None
        start_frame = None
        
        for t, j in path:
            char = transcript[j] if j < len(transcript) else None
            
            if char != current_char:
                if current_char is not None:
                    char_segments.append({
                        'char': current_char,
                        'start': start_frame,
                        'end': t - 1
                    })
                current_char = char
                start_frame = t
        
        # Add final character
        if current_char is not None:
            char_segments.append({
                'char': current_char,
                'start': start_frame,
                'end': path[-1][0]
            })
        
        return char_segments
    
    def _generate_word_alignments(self, char_segments: List, transcript: str, time_ratio: float) -> List[WordSegment]:
        """
        Generate word-level alignments from character segments
        
        Args:
            char_segments: Character-level segments
            transcript: Original transcript
            time_ratio: Frame to time conversion ratio
            
        Returns:
            List of word segments with timestamps
        
        Implementation Logic:
        1. Group characters into words using | delimiter
        2. Calculate word boundaries from character segments
        3. Convert frame indices to time stamps
        4. Return word segments with confidence scores
        """
        words = []
        current_word = ""
        word_start = None
        word_chars = []
        
        for segment in char_segments:
            char = segment['char']
            
            if char == '|':  # Word boundary
                if current_word and word_chars:
                    # Calculate word timing
                    start_time = word_chars[0]['start'] * time_ratio
                    end_time = word_chars[-1]['end'] * time_ratio
                    
                    words.append(WordSegment(
                        word=current_word,
                        start=start_time,
                        end=end_time,
                        score=1.0  # Simplified confidence score
                    ))
                
                current_word = ""
                word_chars = []
            else:
                current_word += char
                word_chars.append(segment)
        
        # Add final word
        if current_word and word_chars:
            start_time = word_chars[0]['start'] * time_ratio
            end_time = word_chars[-1]['end'] * time_ratio
            
            words.append(WordSegment(
                word=current_word,
                start=start_time,
                end=end_time,
                score=1.0
            ))
        
        return words
    
    def merge_short_words(self, word_segments: List[WordSegment]) -> List[WordSegment]:
        """
        Merge short words with neighboring words
        
        Args:
            word_segments: List of word segments
            
        Returns:
            List of merged word segments
        
        Implementation Logic:
        1. Identify words shorter than merge threshold
        2. Find shortest neighboring word for merging
        3. Merge words and update timestamps
        4. Repeat until no more merging is needed
        """
        if not word_segments:
            return []
        
        merged_segments = word_segments.copy()
        
        while True:
            # Find short words
            short_indices = []
            for i, segment in enumerate(merged_segments):
                if segment.start is not None and segment.end is not None:
                    duration = segment.end - segment.start
                    if duration < self.merge_threshold:
                        short_indices.append(i)
            
            if not short_indices:
                break
            
            # Merge shortest word with its shortest neighbor
            shortest_idx = min(short_indices, 
                             key=lambda i: merged_segments[i].end - merged_segments[i].start)
            
            neighbor_idx = self._find_shortest_neighbor(merged_segments, shortest_idx)
            
            if neighbor_idx is not None:
                # Merge segments
                merged_segment = self._merge_two_segments(
                    merged_segments[shortest_idx], 
                    merged_segments[neighbor_idx]
                )
                
                # Remove original segments and insert merged one
                indices_to_remove = sorted([shortest_idx, neighbor_idx], reverse=True)
                for idx in indices_to_remove:
                    merged_segments.pop(idx)
                
                # Insert merged segment at appropriate position
                insert_pos = min(shortest_idx, neighbor_idx)
                merged_segments.insert(insert_pos, merged_segment)
            else:
                break
        
        return merged_segments
    
    def _find_shortest_neighbor(self, segments: List[WordSegment], current_idx: int) -> Optional[int]:
        """
        Find the shortest neighboring word for merging
        
        Args:
            segments: List of word segments
            current_idx: Index of current word
            
        Returns:
            Index of shortest neighbor, or None if no valid neighbor
        """
        neighbors = []
        
        # Check left neighbor
        if current_idx > 0:
            neighbors.append(current_idx - 1)
        
        # Check right neighbor
        if current_idx < len(segments) - 1:
            neighbors.append(current_idx + 1)
        
        if not neighbors:
            return None
        
        # Find shortest neighbor
        shortest_neighbor = min(neighbors, key=lambda i: 
            segments[i].end - segments[i].start if segments[i].start and segments[i].end else float('inf'))
        
        return shortest_neighbor
    
    def _merge_two_segments(self, segment1: WordSegment, segment2: WordSegment) -> WordSegment:
        """
        Merge two word segments into one
        
        Args:
            segment1: First word segment
            segment2: Second word segment
            
        Returns:
            Merged word segment
        
        Implementation Logic:
        1. Combine word texts with space
        2. Use earliest start time
        3. Use latest end time
        4. Average confidence scores
        """
        # Determine order based on start times
        if segment1.start <= segment2.start:
            first, second = segment1, segment2
        else:
            first, second = segment2, segment1
        
        # Merge word texts
        merged_word = f"{first.word} {second.word}"
        
        # Merge timestamps
        merged_start = first.start
        merged_end = second.end
        
        # Average confidence scores
        merged_score = (first.score + second.score) / 2 if first.score and second.score else None
        
        return WordSegment(
            word=merged_word,
            start=merged_start,
            end=merged_end,
            score=merged_score
        )
    
    def extract_audio_features(self, 
                             audio: np.ndarray, 
                             sampling_rate: int,
                             word_segments: List[WordSegment]) -> Dict[str, Any]:
        """
        Extract comprehensive audio features for word segments
        
        Args:
            audio: Audio array
            sampling_rate: Sampling rate
            word_segments: List of word segments with timestamps
            
        Returns:
            Dictionary containing extracted features
        
        Implementation Logic:
        1. Extract features for each word segment
        2. Calculate acoustic features (pitch, energy, spectral)
        3. Compute statistical summaries
        4. Return comprehensive feature dictionary
        """
        features = {
            "word_count": len(word_segments),
            "total_duration": len(audio) / sampling_rate,
            "word_features": []
        }
        
        for word_segment in word_segments:
            if word_segment.start is None or word_segment.end is None:
                continue
            
            # Extract audio segment for this word
            start_sample = int(word_segment.start * sampling_rate)
            end_sample = int(word_segment.end * sampling_rate)
            word_audio = audio[start_sample:end_sample]
            
            if len(word_audio) == 0:
                continue
            
            # Extract acoustic features
            word_features = self._extract_word_features(word_audio, sampling_rate, word_segment)
            features["word_features"].append(word_features)
        
        return features
    
    def _extract_word_features(self, word_audio: np.ndarray, sampling_rate: int, word_segment: WordSegment) -> Dict[str, Any]:
        """
        Extract acoustic features for a single word
        
        Args:
            word_audio: Audio array for the word
            sampling_rate: Sampling rate
            word_segment: Word segment information
            
        Returns:
            Dictionary of acoustic features
        
        Implementation Logic:
        1. Calculate basic timing features
        2. Extract pitch features using Parselmouth
        3. Calculate energy and spectral features
        4. Return comprehensive feature set
        """
        features = {
            "word": word_segment.word,
            "start_time": word_segment.start,
            "end_time": word_segment.end,
            "duration": word_segment.end - word_segment.start,
            "confidence": word_segment.score
        }
        
        try:
            # Basic audio statistics
            features["rms_energy"] = float(np.sqrt(np.mean(word_audio**2)))
            features["max_amplitude"] = float(np.max(np.abs(word_audio)))
            
            # Pitch features using Parselmouth
            sound = parselmouth.Sound(word_audio, sampling_rate)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                features["pitch_mean"] = float(np.mean(pitch_values))
                features["pitch_std"] = float(np.std(pitch_values))
                features["pitch_min"] = float(np.min(pitch_values))
                features["pitch_max"] = float(np.max(pitch_values))
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_min"] = 0.0
                features["pitch_max"] = 0.0
            
            # Spectral features
            stft = librosa.stft(word_audio)
            magnitude = np.abs(stft)
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sampling_rate)[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sampling_rate)[0]
            features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(word_audio)[0]
            features["zero_crossing_rate_mean"] = float(np.mean(zcr))
            
        except Exception as e:
            print(f"Warning: Feature extraction failed for word '{word_segment.word}': {e}")
            # Set default values for failed features
            for key in ["rms_energy", "max_amplitude", "pitch_mean", "pitch_std", 
                       "pitch_min", "pitch_max", "spectral_centroid_mean", 
                       "spectral_rolloff_mean", "zero_crossing_rate_mean"]:
                if key not in features:
                    features[key] = 0.0
        
        return features
    
    def extract_features(self, audio_path: str, text: Optional[str] = None, enable_word_merging: bool = True) -> Dict[str, Any]:
        """
        Main method to extract features from audio file
        
        Args:
            audio_path: Path to audio file
            text: Optional transcription text (if None, will use Whisper)
            enable_word_merging: Whether to merge short words
            
        Returns:
            Dictionary containing all extracted features
        
        Implementation Logic:
        1. Load audio file
        2. Transcribe audio if text not provided
        3. Get word-level timestamps using Triton optimization
        4. Optionally merge short words
        5. Extract comprehensive audio features
        6. Return complete feature dictionary
        """
        try:
            print(f"🎵 Starting feature extraction for: {audio_path}")
            
            # Load audio file
            audio_array, sampling_rate = self.load_audio_file(audio_path)
            
            # Resample to standard rate if needed
            if sampling_rate != SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array, orig_sr=sampling_rate, target_sr=SAMPLE_RATE
                )
                sampling_rate = SAMPLE_RATE
            
            # Transcribe audio if text not provided
            if text is None:
                text = self.transcribe_audio(audio_array, sampling_rate)
            
            if not text.strip():
                return {
                    "error": "Transcription text is empty",
                    "word_features": [],
                    "audio_path": audio_path
                }
            
            # Get word-level timestamps
            aligned_segments = self.get_word_timestamps(audio_array, text)
            
            if not aligned_segments:
                return {
                    "error": "Word-level alignment failed",
                    "word_features": [],
                    "audio_path": audio_path,
                    "transcribed_text": text
                }
            
            # Collect all word segments
            all_word_segments = []
            for segment in aligned_segments:
                all_word_segments.extend(segment.words)
            
            # Record original word count
            original_word_count = len(all_word_segments)
            
            # Optional word merging
            if enable_word_merging:
                all_word_segments = self.merge_short_words(all_word_segments)
                print(f"📊 Word merging: {original_word_count} → {len(all_word_segments)} words")
            
            # Extract audio features
            features = self.extract_audio_features(audio_array, sampling_rate, all_word_segments)
            
            # Add metadata
            features.update({
                "audio_path": audio_path,
                "transcribed_text": text,
                "original_word_count": original_word_count,
                "final_word_count": len(all_word_segments),
                "word_merging_enabled": enable_word_merging,
                "triton_optimization": True
            })
            
            print(f"✅ Feature extraction completed successfully")
            print(f"   Transcribed text: {text}")
            print(f"   Word count: {len(all_word_segments)}")
            print(f"   Total duration: {features['total_duration']:.2f}s")
            
            return features
            
        except Exception as e:
            error_msg = f"Feature extraction failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "word_features": [],
                "audio_path": audio_path
            }


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON data from response text
    
    Args:
        response_text: Text containing JSON data
        
    Returns:
        Extracted JSON dictionary
    
    Implementation Logic:
    1. Try to parse entire text as JSON
    2. If that fails, search for JSON blocks
    3. Return parsed JSON or error information
    """
    try:
        # Try to parse entire text as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Search for JSON blocks in text
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return {"error": "No valid JSON found in response"}


def main():
    """
    Main function demonstrating AudioFeatureExtractor usage
    
    Implementation Logic:
    1. Initialize AudioFeatureExtractor with default settings
    2. Process example audio file (prompt.wav)
    3. Extract features using Triton optimization
    4. Display results and save to JSON file
    5. Demonstrate extract_json_from_response function
    """
    print("🎵 Audio Feature Extractor Demo")
    print("=" * 50)
    
    try:
        # Initialize extractor
        extractor = AudioFeatureExtractor(
            device="cuda",  # Force CUDA for Triton optimization
            merge_threshold=1  # Merge words shorter than 1 seconds
        )
        
        # Example audio file
        audio_file = "prompt.wav"
        
        # Check if example file exists
        if not os.path.exists(audio_file):
            print(f"❌ Example audio file not found: {audio_file}")
            print("Please ensure prompt.wav exists in the current directory")
            return
        
        # Extract features
        print(f"\n🔄 Processing audio file: {audio_file}")
        features = extractor.extract_features(
            audio_path=audio_file,
            text=None,  # Let Whisper transcribe
            enable_word_merging=True
        )
        
        # Display results
        print("\n📊 Extraction Results:")
        print("-" * 30)
        
        if "error" in features:
            print(f"❌ Error: {features['error']}")
        else:
            print(f"✅ Success!")
            print(f"   Audio file: {features.get('audio_path', 'N/A')}")
            print(f"   Transcribed text: {features.get('transcribed_text', 'N/A')}")
            print(f"   Total duration: {features.get('total_duration', 0):.2f}s")
            print(f"   Word count: {features.get('final_word_count', 0)}")
            print(f"   Triton optimization: {features.get('triton_optimization', False)}")
            
            # Display word-level features
            word_features = features.get('word_features', [])
            if word_features:
                print(f"\n📝 Word-level Features (first 3 words):")
                for i, word_feat in enumerate(word_features[:3]):
                    print(f"   Word {i+1}: '{word_feat.get('word', 'N/A')}'")
                    print(f"     Duration: {word_feat.get('duration', 0):.3f}s")
                    print(f"     Pitch mean: {word_feat.get('pitch_mean', 0):.1f}Hz")
                    print(f"     RMS energy: {word_feat.get('rms_energy', 0):.4f}")
        
        # Save results to JSON file
        output_file = "extracted_features.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Demonstrate extract_json_from_response function
        print("\n🔍 Testing extract_json_from_response function:")
        
        # Create a sample response text with JSON
        sample_response = f'''
        Here are the extracted features:
        {json.dumps(features, indent=2)}
        
        The extraction was successful.
        '''
        
        extracted_json = extract_json_from_response(sample_response)
        if "error" not in extracted_json:
            print("✅ JSON extraction successful")
            print(f"   Extracted {len(extracted_json)} top-level keys")
        else:
            print(f"❌ JSON extraction failed: {extracted_json['error']}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()