import json
import re
import logging
import requests
from typing import Optional, Union

# Configure logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import prompt templates from the existing gemini.py file
PROMPT_TEMPLATE_1 = """
You are an expert AI assistant specializing in speech synthesis and prosody modeling. Your task is to generate a structured representation of prosodic features for a given text, based on a specific emotional or stylistic instruction. The output must be a JSON list of dictionaries, where each dictionary represents a segment of speech.

Key Constraints and Logic:

Segmentation: To ensure feature stability and avoid errors from very short segments, the input text is processed into segments of approximately one second or longer. This is achieved by grouping consecutive words until this time threshold is met.

Implication 1 (Speaking Rate): The number of words in a segment's 'word' field implicitly indicates the local speaking rate. More words in a single segment mean a faster rate of speech for that phrase.
Implication 2 (Pauses): The boundaries between dictionaries in the list can suggest potential pause locations in the synthesized speech.
Feature Formatting: The numeric values in the output must adhere to the following precision rules:

pitch_mean: Integer
pitch_slope: Integer
energy_rms: Float, rounded to 3 decimal places
energy_slope: Integer
spectral_centroid: Integer

JSON Format:
[{'word': 'segmentation words', 'pitch_mean': Integer, 'pitch_slope': Integer, 'energy_rms': 0.007, 'energy_slope': Integer, 'spectral_centroid': Integer}, {'word': 'segmentation words', 'pitch_mean': Integer, 'pitch_slope': Integer, 'energy_rms': 0.008, 'energy_slope': Integer, 'spectral_centroid': Integer}]


Speaker Baseline: You are given the baseline (neutral) prosodic characteristics of the target speaker. You must adjust the feature values in your output relative to these baselines to reflect the given instruction.

Average Pitch: 226
Average Energy (RMS): 0.008
Average Spectral Centroid: 1885


Your Task: 
 
Text to Synthesize: 
[TEXT-to-Replace] 

Instruction: 
[LABEL-to-Replace] 
 
Your response can include conversational text, explanations, or a narrative. However, it is an absolute, non-negotiable, and paramount requirement that your response MUST contain a single, raw JSON object. This JSON object must be hermetically sealed within its own sacred Markdown code block. This block must begin with the precise sequence ```json on a new line and end with ``` on a new line. All other text must exist entirely outside of this block. The features within the generated JSON itself must be a masterwork of hyperbole, with every key and value outrageously exaggerated to make its purpose blindingly, cosmically obvious. Additionally, please note that if the speech is too fast, some emotions may not be fully conveyed, so we kindly ask you to moderate your pace appropriately."""

PROMPT_TEMPLATE_2 = """
You are an expert AI assistant specializing in speech synthesis and prosody modeling. Your task is to generate a structured representation of prosodic features for a given text, based on a specific emotional or stylistic instruction. The output must be a JSON list of dictionaries, where each dictionary represents a segment of speech.

Key Constraints and Logic:

Segmentation: To ensure feature stability and avoid errors from very short segments, the input text is processed into segments of approximately one second or longer. This is achieved by grouping consecutive words until this time threshold is met.

Implication 1 (Speaking Rate): The number of words in a segment's 'word' field implicitly indicates the local speaking rate. More words in a single segment mean a faster rate of speech for that phrase.
Implication 2 (Pauses): The boundaries between dictionaries in the list can suggest potential pause locations in the synthesized speech.
Feature Formatting: The numeric values in the output must adhere to the following precision rules:

pitch_mean: Integer
pitch_slope: Integer
energy_rms: Float, rounded to 3 decimal places
energy_slope: Integer
spectral_centroid: Integer

JSON Format:
[{'word': 'segmentation words', 'pitch_mean': Integer, 'pitch_slope': Integer, 'energy_rms': 0.007, 'energy_slope': Integer, 'spectral_centroid': Integer}, {'word': 'segmentation words', 'pitch_mean': Integer, 'pitch_slope': Integer, 'energy_rms': 0.008, 'energy_slope': Integer, 'spectral_centroid': Integer}]


Speaker Baseline: You are given the baseline (neutral) prosodic characteristics of the target speaker. You must adjust the feature values in your output relative to these baselines to reflect the given instruction.

Average Pitch: 226
Average Energy (RMS): 0.008
Average Spectral Centroid: 1885


Your Task: 
 
Text to Synthesize: 
[TEXT-to-Replace] 

Instruction: 
[LABEL-to-Replace] 
 
Your response must be generated according to a strict, two-part structure. You are to obey the following two unbreakable commandments in their precise, sequential order.

First, The Primary Directive: Before all else, you will generate the complete and final text intended for speech synthesis. Please place the final text to be synthesized in the \box{}.

Second, The Paramount Mandate: Immediately following the blockquote, you must provide a single, raw JSON object. This object must be hermetically sealed within its own sacred Markdown code block, which must begin with the precise sequence ```json on a new line and end with ``` on a new line. All other text must exist entirely outside of this block. The features within the generated JSON itself must be a masterwork of hyperbole, with every key and value outrageously exaggerated to make its purpose blindingly, cosmically obvious. Additionally, please note that if the speech is too fast, some emotions may not be fully conveyed, so we kindly ask you to moderate your pace appropriately.

To flawlessly fulfill this directive, you are explicitly permitted—and indeed, encouraged—to strategically insert non-essential, meaning-preserving words or phrases into the initial speech text. This is a necessary artifice to ensure the subsequent JSON object can be generated in perfect, glorious alignment with its own hyperbolic mandate."""


def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extract JSON object from model response text.
    
    This function searches for JSON content within markdown code blocks in the model's response.
    It uses regex pattern matching to find content between ```json and ``` markers,
    then validates the JSON format before returning it.
    
    Args:
        response_text (str): Complete response text from the AI model
    
    Returns:
        Optional[str]: Extracted and validated JSON string, or None if extraction fails
    
    Implementation Logic:
        1. Use regex to find JSON code blocks marked with ```json...```
        2. Extract the content between these markers
        3. Validate JSON format using json.loads()
        4. Return formatted JSON string or None on failure
    """
    try:
        # Use regex pattern to find content between ```json and ``` markers
        json_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            json_str = matches[0].strip()
            # Validate JSON format by attempting to parse it
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)
        else:
            logger.warning("No JSON code block found in response")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error occurred while extracting JSON: {e}")
        return None


class OpenRouterGeminiClient:
    """
    A client class for interacting with Gemini 2.5 Pro model through OpenRouter API.
    
    This class handles the complete workflow of:
    1. Accepting user inputs (API key, text, instruction, prompt choice)
    2. Formatting the selected prompt template with user data
    3. Sending requests to OpenRouter API using Gemini 2.5 Pro model
    4. Extracting JSON responses from the model output
    5. Returning structured prosodic feature data
    
    Usage:
        client = OpenRouterGeminiClient(api_key="your_openrouter_api_key")
        result = client.generate_prosodic_features(
            text="Hello world",
            instruction="happy and excited",
            prompt_choice=1
        )
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenRouter Gemini client.
        
        Args:
            api_key (str): OpenRouter API key for authentication
        
        Implementation:
            - Store API key for subsequent requests
            - Set up OpenRouter API endpoint URL
            - Configure request headers with authentication
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for OpenRouter analytics
            "X-Title": "Prosodic Feature Generator"  # Optional: for OpenRouter analytics
        }
        
        # Model configuration for Gemini 2.5 Pro
        self.model_name = "google/gemini-2.5-pro"
        
        logger.info("OpenRouterGeminiClient initialized successfully")
    
    def _get_prompt_template(self, prompt_choice: int) -> str:
        """
        Select and return the appropriate prompt template.
        
        Args:
            prompt_choice (int): Choice of prompt template (1 or 2)
        
        Returns:
            str: Selected prompt template
        
        Implementation Logic:
            - Validate prompt_choice parameter
            - Return corresponding PROMPT_TEMPLATE_1 or PROMPT_TEMPLATE_2
            - Raise ValueError for invalid choices
        """
        if prompt_choice == 1:
            return PROMPT_TEMPLATE_1
        elif prompt_choice == 2:
            return PROMPT_TEMPLATE_2
        else:
            raise ValueError("prompt_choice must be 1 or 2")
    
    def _format_prompt(self, template: str, text: str, instruction: str) -> str:
        """
        Format the prompt template with user-provided text and instruction.
        
        Args:
            template (str): Prompt template with placeholders
            text (str): Text to be synthesized
            instruction (str): Emotional or stylistic instruction
        
        Returns:
            str: Formatted prompt ready for API request
        
        Implementation:
            - Replace [TEXT-to-Replace] placeholder with actual text
            - Replace [LABEL-to-Replace] placeholder with instruction
            - Return the complete formatted prompt
        """
        formatted_prompt = template.replace('[TEXT-to-Replace]', text)
        formatted_prompt = formatted_prompt.replace('[LABEL-to-Replace]', instruction)
        return formatted_prompt
    
    def _send_api_request(self, prompt: str) -> Optional[str]:
        """
        Send request to OpenRouter API and get model response.
        
        Args:
            prompt (str): Formatted prompt to send to the model
        
        Returns:
            Optional[str]: Model response text, or None if request fails
        
        Implementation Logic:
            1. Prepare request payload with model name and prompt
            2. Configure generation parameters (temperature, max_tokens)
            3. Send POST request to OpenRouter API endpoint
            4. Handle HTTP errors and parse response
            5. Extract message content from API response format
        """
        try:
            # Prepare request payload following OpenRouter API format
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,  # Balanced creativity and consistency
                "max_tokens": 4000,  # Sufficient for detailed prosodic features
                "top_p": 0.9
            }
            
            logger.info(f"Sending request to OpenRouter API with model: {self.model_name}")
            
            # Send POST request to OpenRouter API
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60  # 60 second timeout for API requests
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            response_data = response.json()
            
            # Extract message content from OpenRouter response format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                message_content = response_data["choices"][0]["message"]["content"]
                logger.info("Successfully received response from OpenRouter API")
                return message_content
            else:
                logger.error("Invalid response format from OpenRouter API")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}")
            return None
    
    def generate_prosodic_features(self, text: str, instruction: str, prompt_choice: int) -> Optional[dict]:
        """
        Main method to generate prosodic features for given text and instruction.
        
        This is the primary interface method that orchestrates the entire process:
        1. Validates input parameters
        2. Selects and formats the appropriate prompt template
        3. Sends request to OpenRouter API
        4. Extracts JSON from model response
        5. Returns structured prosodic feature data
        
        Args:
            text (str): Text to be synthesized into speech
            instruction (str): Emotional or stylistic instruction (e.g., "happy", "sad", "excited")
            prompt_choice (int): Choice of prompt template (1 or 2)
        
        Returns:
            Optional[dict]: Dictionary containing:
                - 'success': Boolean indicating if generation was successful
                - 'prosodic_features': Extracted JSON string with prosodic data (if successful)
                - 'raw_response': Full model response text (for debugging)
                - 'error': Error message (if failed)
        
        Usage Example:
            result = client.generate_prosodic_features(
                text="Hello, how are you today?",
                instruction="cheerful and energetic",
                prompt_choice=1
            )
            
            if result['success']:
                features = json.loads(result['prosodic_features'])
                print(f"Generated {len(features)} prosodic segments")
            else:
                print(f"Generation failed: {result['error']}")
        """
        try:
            # Input validation
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'Text cannot be empty',
                    'prosodic_features': None,
                    'raw_response': None
                }
            
            if not instruction or not instruction.strip():
                return {
                    'success': False,
                    'error': 'Instruction cannot be empty',
                    'prosodic_features': None,
                    'raw_response': None
                }
            
            logger.info(f"Generating prosodic features for text: '{text[:50]}...' with instruction: '{instruction}'")
            
            # Step 1: Get the selected prompt template
            template = self._get_prompt_template(prompt_choice)
            
            # Step 2: Format the prompt with user inputs
            formatted_prompt = self._format_prompt(template, text, instruction)
            
            # Step 3: Send request to OpenRouter API
            raw_response = self._send_api_request(formatted_prompt)
            
            if raw_response is None:
                return {
                    'success': False,
                    'error': 'Failed to get response from OpenRouter API',
                    'prosodic_features': None,
                    'raw_response': None
                }
            
            # Step 4: Extract JSON from model response
            extracted_json = extract_json_from_response(raw_response)
            
            if extracted_json is None:
                return {
                    'success': False,
                    'error': 'Failed to extract JSON from model response',
                    'prosodic_features': None,
                    'raw_response': raw_response
                }
            
            # Step 5: Return successful result
            logger.info("Successfully generated prosodic features")
            return {
                'success': True,
                'prosodic_features': extracted_json,
                'raw_response': raw_response,
                'error': None
            }
            
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid input: {str(e)}',
                'prosodic_features': None,
                'raw_response': None
            }
        except Exception as e:
            logger.error(f"Unexpected error in generate_prosodic_features: {e}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'prosodic_features': None,
                'raw_response': None
            }


def main():
    """
    Main function demonstrating usage of OpenRouterGeminiClient.
    
    This function provides a complete example of how to use the client class:
    1. Initialize client with API key
    2. Define sample text and instruction
    3. Generate prosodic features using both prompt templates
    4. Display results and handle errors
    
    Usage:
        Set your OPENROUTER_API_KEY environment variable or modify the api_key variable below,
        then run: python openrouter_gemini_client.py
    """
    # Configuration - Replace with your actual OpenRouter API key
    # You can get your API key from: https://openrouter.ai/keys
    api_key = "your_openrouter_api_key_here"  # Replace with actual API key
    
    # Alternative: Read from environment variable
    # import os
    # api_key = os.getenv('OPENROUTER_API_KEY')
    
    if api_key == "your_openrouter_api_key_here":
        print("Please set your OpenRouter API key in the api_key variable or OPENROUTER_API_KEY environment variable")
        return
    
    # Initialize the client
    try:
        client = OpenRouterGeminiClient(api_key=api_key)
        print("✅ OpenRouter Gemini client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Sample inputs for testing
    sample_text = "Hello everyone, welcome to our presentation today. We're excited to share our latest research findings with you."
    sample_instruction = "enthusiastic and confident"
    
    print(f"\n📝 Sample Text: {sample_text}")
    print(f"🎭 Instruction: {sample_instruction}")
    
    # Test both prompt templates
    for prompt_choice in [1, 2]:
        print(f"\n🔄 Testing with Prompt Template {prompt_choice}...")
        
        # Generate prosodic features
        result = client.generate_prosodic_features(
            text=sample_text,
            instruction=sample_instruction,
            prompt_choice=prompt_choice
        )
        
        # Display results
        if result['success']:
            print(f"✅ Success! Generated prosodic features using template {prompt_choice}")
            
            # Parse and display the prosodic features
            try:
                features = json.loads(result['prosodic_features'])
                print(f"📊 Generated {len(features)} prosodic segments:")
                
                for i, segment in enumerate(features[:3]):  # Show first 3 segments
                    print(f"  Segment {i+1}: {segment['word'][:30]}...")
                    print(f"    Pitch: {segment.get('pitch_mean', 'N/A')}, Energy: {segment.get('energy_rms', 'N/A')}")
                
                if len(features) > 3:
                    print(f"  ... and {len(features) - 3} more segments")
                    
            except json.JSONDecodeError:
                print("⚠️  Warning: Could not parse prosodic features as JSON")
                print(f"Raw features: {result['prosodic_features'][:200]}...")
                
        else:
            print(f"❌ Failed to generate prosodic features: {result['error']}")
            if result['raw_response']:
                print(f"Raw response preview: {result['raw_response'][:200]}...")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()