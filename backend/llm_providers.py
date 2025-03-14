import os
import json
import requests
from llm_provider_base import LLMProvider
import time


# Existing providers (unchanged for brevity)
class OpenAIProvider(LLMProvider):
    """OpenAI API provider for note generation"""
    
    def __init__(self, api_key, model="gpt-4"):
        self.api_key = api_key
        self.model = model
        self.backoff_factor = 2  # Added for _handle_rate_limit

    def _handle_rate_limit(self, retries):
        wait_time = self.backoff_factor ** retries
        time.sleep(wait_time)

    def generate_notes(self, transcript, verbose=False, format="markdown", additional_instructions=""):
        if verbose:
            print(f"Generating notes using OpenAI's {self.model} with enhanced thinking process...")
        
        retries = 0
        while retries < 5:
            try:
                import openai
                openai.api_key = self.api_key
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": f"Summarize the following transcript in a clear, concise, and well-structured format using markdown. Focus on key points, main arguments, and important details while preserving the original meaning. Use bullet points, headings, and subheadings for better organization. Remove filler words and unnecessary details. Include key words from the text. Ensure the summary is easy to read and captures the core essence of the following content: \n\n{transcript}"}],
                    temperature=0.4,
                    max_tokens=3500,
                )
                return response.choices[0].message.content
            
            except openai.error.RateLimitError as e:
                retries += 1
                wait_time = e.retry_after if hasattr(e, 'retry_after') else 5
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                self._handle_rate_limit(retries)
            except ImportError:
                raise ImportError("The 'openai' package is required for OpenAI integration. Install it with 'pip install openai'.")
            except openai.error.OpenAIError as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider for note generation"""
    # (Unchanged, keeping it concise)
    def __init__(self, api_key, model="claude-3-opus-20240229"):
        self.api_key = api_key
        self.model = model
        
    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using Anthropic's {self.model}...")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="You are a helpful assistant that takes transcripts and creates well-structured notes in markdown format. Create headings, bullet points, and emphasize key concepts. Be concise but comprehensive.",
                messages=[{"role": "user", "content": f"Please create organized markdown notes from this transcript:\n\n{transcript}"}],
                temperature=0.3,
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("The 'anthropic' package is required. Install it with 'pip install anthropic'.")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")

class GoogleProvider(LLMProvider):
    """Google Gemini API provider for note generation"""
    # (Unchanged, keeping it concise)
    def __init__(self, api_key, model="gemini-pro"):
        self.api_key = api_key
        self.model = model
        
    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using Google's {self.model}...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                """You are a helpful assistant that takes transcripts and creates well-structured notes in markdown format.
                Create headings, bullet points, and emphasize key concepts. Be concise but comprehensive.
                Please create organized markdown notes from this transcript:\n\n""" + transcript,
                generation_config={"temperature": 0.3}
            )
            return response.text
        except ImportError:
            raise ImportError("The 'google-generativeai' package is required. Install it with 'pip install google-generativeai'.")
        except Exception as e:
            raise RuntimeError(f"Google Gemini API error: {str(e)}")

class CustomEndpointProvider(LLMProvider):
    """Custom API endpoint provider for note generation"""
    # (Unchanged, keeping it concise)
    def __init__(self, endpoint_url, api_key=None, headers=None, payload_template=None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.headers = headers or {}
        if api_key and 'authorization' not in [h.lower() for h in self.headers.keys()]:
            self.headers['Authorization'] = f'Bearer {api_key}'
        self.payload_template = payload_template or '''
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that takes transcripts and creates well-structured notes in markdown format. Create headings, bullet points, and emphasize key concepts. Be concise but comprehensive."},
                {"role": "user", "content": "Please create organized markdown notes from this transcript:\\n\\n{{transcript}}"}
            ],
            "temperature": 0.3
        }
        '''
        
    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using custom endpoint: {self.endpoint_url}...")
        try:
            payload_str = self.payload_template.replace('{{transcript}}', transcript)
            payload = json.loads(payload_str)
            response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return self._extract_content(response.json())
        except Exception as e:
            raise RuntimeError(f"Custom API error: {str(e)}")
    
    def _extract_content(self, response_data):
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
            elif 'text' in choice:
                return choice['text']
        elif 'content' in response_data:
            if isinstance(response_data['content'], list) and len(response_data['content']) > 0:
                return response_data['content'][0].get('text', '')
            return response_data['content']
        elif 'text' in response_data:
            return response_data['text']
        elif 'output' in response_data:
            return response_data['output']
        return json.dumps(response_data)


class XAIProvider(LLMProvider):
    """xAI API provider for note generation (e.g., Grok)"""
    
    def __init__(self, api_key, model="grok", endpoint_url="https://api.xai.com/v1/chat/completions"):
        self.api_key = api_key
        self.model = model
        self.endpoint_url = endpoint_url
        self.backoff_factor = 2

    def _handle_rate_limit(self, retries):
        wait_time = self.backoff_factor ** retries
        time.sleep(wait_time)

    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using xAI's {self.model}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": f"Summarize the following transcript into well-structured markdown notes. Use headings, bullet points, and emphasize key concepts. Be concise, clear, and comprehensive:\n\n{transcript}"}
            ],
            "temperature": 0.4,
            "max_tokens": 3500
        }
        
        retries = 0
        while retries < 5:
            try:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    self._handle_rate_limit(retries)
                else:
                    raise RuntimeError(f"xAI API error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}")

class MistralProvider(LLMProvider):
    """Mistral API provider for note generation"""
    
    def __init__(self, api_key, model="mistral-large", endpoint_url="https://api.mixtral.ai/v1/chat/completions"):
        self.api_key = api_key
        self.model = model
        self.endpoint_url = endpoint_url
        self.backoff_factor = 2

    def _handle_rate_limit(self, retries):
        wait_time = self.backoff_factor ** retries
        time.sleep(wait_time)

    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using Mistral's {self.model}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates well-structured markdown notes from transcripts."},
                {"role": "user", "content": f"Create organized markdown notes from this transcript:\n\n{transcript}"}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        retries = 0
        while retries < 5:
            try:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    self._handle_rate_limit(retries)
                else:
                    raise RuntimeError(f"Mixtral API error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}")

class HuggingFaceProvider(LLMProvider):
    """Hugging Face Inference API provider for note generation"""
    
    def __init__(self, api_key, model="mistralai/Mixtral-8x7B-Instruct-v0.1", endpoint_url="https://api-inference.huggingface.co/models/"):
        self.api_key = api_key
        self.model = model
        self.endpoint_url = endpoint_url + model 
        self.backoff_factor = 2

    def _handle_rate_limit(self, retries):
        wait_time = self.backoff_factor ** retries
        time.sleep(wait_time)

    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using Hugging Face's {self.model}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": f"You are a helpful assistant that creates well-structured markdown notes from transcripts. Create headings, bullet points, and emphasize key concepts. Be concise but comprehensive.\n\nTranscript:\n{transcript}\n\nNotes:",
            "parameters": {
                "temperature": 0.3,
                "max_new_tokens": 2000,
                "return_full_text": False
            }
        }
        
        retries = 0
        while retries < 5:
            try:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                return response_data[0]["generated_text"]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    self._handle_rate_limit(retries)
                else:
                    raise RuntimeError(f"Hugging Face API error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}")
            
class OllamaProvider(LLMProvider):
    """Ollama local API provider for note generation"""
    
    def __init__(self, model="llama3", endpoint_url="http://localhost:11434/api/chat"):
        self.model = model
        self.endpoint_url = endpoint_url
        self.backoff_factor = 2

    def _handle_rate_limit(self, retries):
        wait_time = self.backoff_factor ** retries
        time.sleep(wait_time)

    def generate_notes(self, transcript, verbose=False):
        if verbose:
            print(f"Generating notes using Ollama's {self.model}...")
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates well-structured markdown notes from transcripts. Use headings, bullet points, and emphasize key concepts. Be concise but comprehensive."},
                {"role": "user", "content": f"Create organized markdown notes from this transcript:\n\n{transcript}"}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
        
        retries = 0
        while retries < 5:
            try:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                return response_data["message"]["content"]
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit (unlikely locally, but included for consistency)
                    retries += 1
                    wait_time = 2 ** retries
                    print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                    self._handle_rate_limit(retries)
                else:
                    raise RuntimeError(f"Ollama API error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected error: {str(e)}")

def create_llm_provider(args):
    """Create the appropriate LLM provider based on command line arguments"""
    provider_type = args.llm_provider.lower()
    
    api_key = args.api_key or os.environ.get(f"{provider_type.upper()}_API_KEY")
    if not api_key and provider_type not in ["custom", "ollama"] and not args.config_file:
        raise ValueError(f"API key is required for {provider_type}. Provide it with --api-key or set the {provider_type.upper()}_API_KEY environment variable.")
    
    config = {}
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading config file: {str(e)}")
    
    if provider_type == "openai":
        return OpenAIProvider(api_key, model=args.model_name or "gpt-4")
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key, model=args.model_name or "claude-3-opus-20240229")
    elif provider_type == "google":
        return GoogleProvider(api_key, model=args.model_name or "gemini-pro")
    elif provider_type == "custom":
        endpoint = args.endpoint or config.get("endpoint")
        if not endpoint:
            raise ValueError("Endpoint URL is required for custom provider.")
        headers = config.get("headers", {})
        payload_template = config.get("payload_template")
        return CustomEndpointProvider(endpoint, api_key, headers, payload_template)
    elif provider_type == "xai":
        endpoint = args.endpoint or config.get("endpoint", "https://api.xai.com/v1/chat/completions")
        return XAIProvider(api_key, model=args.model_name or "grok", endpoint_url=endpoint)
    elif provider_type == "mistral":
        endpoint = args.endpoint or config.get("endpoint", "https://api.mixtral.ai/v1/chat/completions")
        return MistralProvider(api_key, model=args.model_name or "mistral-large", endpoint_url=endpoint)
    elif provider_type == "huggingface":
        endpoint = args.endpoint or config.get("endpoint", "https://api-inference.huggingface.co/models/")
        return HuggingFaceProvider(api_key, model=args.model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1", endpoint_url=endpoint)
    elif provider_type == "ollama":
        endpoint = args.endpoint or config.get("endpoint", "http://localhost:11434/api/chat")
        return OllamaProvider(model=args.model_name or "llama3", endpoint_url=endpoint)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_type}")