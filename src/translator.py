from deep_translator import GoogleTranslator
import deepl
from openai import OpenAI
from typing import List, Union, Optional
import base64
import io
import json
from PIL import Image

class TranslatorService:
    def __init__(self, source: str = 'en', target: str = 'de', service_type: str = 'google', api_key: Optional[str] = None):
        """
        Initializes the Translator Service.
        
        Args:
            source: Source language code (default: 'en').
            target: Target language code (default: 'de').
            service_type: 'google', 'deepl', 'openai', or 'xai'.
            api_key: API Key for DeepL, OpenAI or xAI.
        """
        self.service_type = service_type
        self.api_key = api_key
        self.target = target
        self.source = source
        self.usage = {'input_tokens': 0, 'output_tokens': 0}
        
        if self.service_type == 'deepl':
            print("Using DeepL Translator")
            if not self.api_key:
                raise ValueError("DeepL API Key is required for DeepL service.")
            self.translator = deepl.Translator(self.api_key)
            
        elif self.service_type == 'openai':
            print("Using OpenAI (GPT-4o-mini) Translator")
            if not self.api_key:
                raise ValueError("OpenAI API Key is required for OpenAI service.")
            self.client = OpenAI(api_key=self.api_key)

        elif self.service_type == 'xai':
            print("Using xAI Grok Translator")
            if not self.api_key:
                raise ValueError("xAI API Key is required for Grok service.")
            # xAI API is OpenAI-compatible
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")
            
        else:
            print("Using Google Translator (deep-translator)")
            self.translator = GoogleTranslator(source=source, target=target)

    def get_usage_stats(self):
        """Returns accumulated token usage."""
        return self.usage

    def get_cost_estimate(self):
        """
        Returns estimated cost in USD based on GPT-4o-mini pricing.
        Input: $0.15 / 1M tokens
        Output: $0.60 / 1M tokens
        """
        input_cost = (self.usage['input_tokens'] / 1_000_000) * 0.15
        output_cost = (self.usage['output_tokens'] / 1_000_000) * 0.60
        return input_cost + output_cost

    def validate_api_key(self) -> None:
        """Performs a lightweight test call to validate the configured API key.

        Raises:
            Exception: If the key is invalid or the provider returns an auth error.
        """
        # Google (deep-translator) does not use an API key
        if self.service_type not in ['deepl', 'openai', 'xai']:
            return

        if self.service_type == 'deepl':
            # Minimal ping using the official client
            try:
                # This will raise an exception on invalid auth
                _ = self.translator.get_usage()
            except Exception as e:
                raise Exception(f"DeepL API key seems invalid or not authorized: {e}")
            return

        # OpenAI / xAI
        try:
            model = "gpt-4o-mini" if self.service_type == 'openai' else "grok-4-mini"
            # Very small test prompt to minimize cost
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "test"}
                ],
                max_tokens=1,
                temperature=0.0,
            )
            # If we get here without exception, we assume the key works.
            if response.usage:
                self.usage['input_tokens'] += response.usage.prompt_tokens
                self.usage['output_tokens'] += response.usage.completion_tokens
        except Exception as e:
            raise Exception(f"{self.service_type.capitalize()} API key seems invalid or the service is not reachable: {e}")

    def translate_image_with_vision(self, image: Image.Image) -> List[dict]:
        """
        Uses VLM (Vision Language Model) to detect and translate text directly from image.
        Returns list of dicts: {'bbox': [x1, y1, x2, y2], 'original': str, 'translated': str}
        """
        if self.service_type not in ['openai', 'xai']:
             raise ValueError("Vision features only supported for OpenAI and xAI services.")

        # 1. Letterbox the image to be square (helps with coordinate accuracy)
        old_width, old_height = image.size
        new_size = max(old_width, old_height)
        square_img = Image.new("RGB", (new_size, new_size), (255, 255, 255))
        
        # Paste original image centered or top-left? Top-left is easier for coord math.
        square_img.paste(image, (0, 0))
        
        # Convert to base64
        buffered = io.BytesIO()
        square_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{img_str}"
        
        model = "gpt-4o-mini" if self.service_type == 'openai' else "grok-4-latest"
        
        prompt = f"""
        You are a Manga Translator Agent. 
        Look at this manga page. Identify all speech bubbles and text boxes.
        For each text region:
        1. Extract the English text.
        2. Translate it to German.
        3. Estimate the bounding box as [ymin, xmin, ymax, xmax] using a 0-1000 normalized scale based on this square image.
           - (0,0) is top-left corner.
           - (1000,1000) is bottom-right corner.
           - Be extremely precise with the coordinates.
           - The image might have white padding on the right or bottom, ignore that area.
        
        Return ONLY a valid JSON array with this structure:
        [
            {{
                "original": "English text",
                "translated": "German translation",
                "bbox": [ymin, xmin, ymax, xmax]
            }}
        ]
        Do not use markdown code blocks. Return raw JSON only.
        """

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            }
                        ],
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Track usage
            if response.usage:
                self.usage['input_tokens'] += response.usage.prompt_tokens
                self.usage['output_tokens'] += response.usage.completion_tokens
            
            content = response.choices[0].message.content.strip()
            # Cleanup markdown if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content.strip())
            
            results = []
            for item in data:
                ymin, xmin, ymax, xmax = item['bbox']
                
                # Clamp values 0-1000
                ymin = max(0, min(1000, ymin))
                xmin = max(0, min(1000, xmin))
                ymax = max(0, min(1000, ymax))
                xmax = max(0, min(1000, xmax))
                
                # Convert from 0-1000 scale relative to the SQUARE image
                abs_x_min = int((xmin / 1000) * new_size)
                abs_y_min = int((ymin / 1000) * new_size)
                abs_x_max = int((xmax / 1000) * new_size)
                abs_y_max = int((ymax / 1000) * new_size)
                
                # Clip to original image dimensions (remove padding area results)
                abs_x_min = min(abs_x_min, old_width)
                abs_y_min = min(abs_y_min, old_height)
                abs_x_max = min(abs_x_max, old_width)
                abs_y_max = min(abs_y_max, old_height)
                
                # Ensure valid box
                if abs_x_max > abs_x_min and abs_y_max > abs_y_min:
                    bbox_points = [
                        [abs_x_min, abs_y_min], # Top-Left
                        [abs_x_max, abs_y_min], # Top-Right
                        [abs_x_max, abs_y_max], # Bottom-Right
                        [abs_x_min, abs_y_max]  # Bottom-Left
                    ]
                    
                    results.append({
                        'bbox': bbox_points,
                        'original': item.get('original', ''),
                        'translated': item.get('translated', '')
                    })
            
            return results
            
        except Exception as e:
            print(f"Vision translation error: {e}")
            return []

    def translate_text(self, text: str) -> str:
        """
        Translates a single string.
        """
        if not text.strip():
            return ""
            
        try:
            if self.service_type == 'deepl':
                # DeepL uses slightly different language codes (e.g. 'DE' instead of 'de' usually, but 'de' works)
                result = self.translator.translate_text(text, source_lang=None, target_lang=self.target)
                return result.text
                
            elif self.service_type in ['openai', 'xai']:
                # Select model based on service
                model = "gpt-4o-mini" if self.service_type == 'openai' else "grok-4-latest"
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"You are a professional manga translator. Translate the following text from {self.source} to {self.target}. Keep the translation natural and fitting for a comic/manga context. Ensure correct handling of German special characters like ä, ö, ü, ß. Only return the translated text, nothing else."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3
                )
                
                # Track usage
                if response.usage:
                    self.usage['input_tokens'] += response.usage.prompt_tokens
                    self.usage['output_tokens'] += response.usage.completion_tokens

                return response.choices[0].message.content.strip()
                
            else:
                return self.translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translates a list of strings.
        """
        if not texts:
            return []

        try:
            if self.service_type == 'deepl':
                results = self.translator.translate_text(texts, source_lang=None, target_lang=self.target)
                return [r.text for r in results]
                
            elif self.service_type in ['openai', 'xai']:
                # Select model based on service
                model = "gpt-4o-mini" if self.service_type == 'openai' else "grok-4-latest"

                # OpenAI/xAI batch approach
                formatted_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
                prompt = f"Translate the following numbered lines from {self.source} to {self.target}. Return them as a numbered list with the same indices.\n\n{formatted_text}"
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                         {"role": "system", "content": f"You are a professional manga translator. Translate the text from {self.source} to {self.target}. Return ONLY the numbered list of translations."},
                         {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                # Track usage
                if response.usage:
                    self.usage['input_tokens'] += response.usage.prompt_tokens
                    self.usage['output_tokens'] += response.usage.completion_tokens

                content = response.choices[0].message.content.strip()
                
                # Parse results back to list
                translated_lines = []
                # Simple parsing (robustness could be improved)
                for line in content.split('\n'):
                    if '. ' in line:
                        parts = line.split('. ', 1)
                        if len(parts) > 1:
                            translated_lines.append(parts[1])
                        else:
                             translated_lines.append(line)
                    else:
                         translated_lines.append(line)
                         
                # Fallback if counts don't match (rare but possible)
                if len(translated_lines) != len(texts):
                     return [self.translate_text(t) for t in texts]
                     
                return translated_lines
                
            else:
                return self.translator.translate_batch(texts)
        except Exception as e:
            print(f"Batch translation error: {e}")
            return texts
