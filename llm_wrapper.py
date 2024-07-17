import requests
import json
import re
import tiktoken
from typing import Dict, Any, Generator, Union

class OllamaApiWrapper:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", max_retries: int = 3):
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.system_prompt = (
            "You are an AI assistant designed to provide helpful responses. "
            "Your primary function is to assist users while maintaining a helpful and informative output."
        )
        self.json_formatting_prompt = (
            "Format the following input into a valid JSON object. "
            "Return ONLY the JSON object, without any additional text, explanations, or code blocks. "
            "Here's the input to format into a JSON object:\n\n"
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's tokenizer as an approximation

    def generate_response(self, user_prompt: str, stream: bool = False, json_mode: bool = False, temperature: float = 1.0) -> Union[
        Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        sanitized_prompt = self.sanitize_input(user_prompt)

        if json_mode:
            return self._generate_json_response(sanitized_prompt, stream, temperature)
        elif stream:
            return self._stream_response(sanitized_prompt, temperature)
        else:
            return self._generate_full_response(sanitized_prompt, temperature)

    def _generate_json_response(self, prompt: str, stream: bool, temperature: float) -> Union[
        Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        initial_response = self._generate_full_response(prompt, temperature)

        if "error" in initial_response["response"]:
            return initial_response

        json_prompt = self.json_formatting_prompt + initial_response["response"]

        if stream:
            return self._stream_json_formatting(json_prompt, initial_response["usage"], temperature)
        else:
            json_response = self._generate_full_response(json_prompt, temperature)
            return self._process_json_response(json_response, initial_response["usage"])

    def _generate_full_response(self, prompt: str, temperature: float) -> Dict[str, Any]:
        prompt_tokens = self.count_tokens(self.system_prompt + prompt)
        for attempt in range(self.max_retries):
            response = self._make_api_call(prompt, stream=False, temperature=temperature)

            if isinstance(response, dict) and "error" in response:
                return {
                    "response": response["error"],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}
                }

            if "response" in response:
                completion_tokens = self.count_tokens(response["response"])
                total_tokens = prompt_tokens + completion_tokens
                return {
                    "response": response["response"],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": total_tokens}
                }

        return {
            "response": "Failed to generate response after multiple attempts",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens}
        }

    def _stream_response(self, prompt: str, temperature: float) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.count_tokens(self.system_prompt + prompt)
        completion_tokens = 0
        for chunk in self._make_api_call(prompt, stream=True, temperature=temperature):
            if isinstance(chunk, dict) and "error" in chunk:
                yield {
                    "response": chunk["error"],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens + completion_tokens}
                }
                return

            chunk_text = chunk.get("response", "")
            chunk_tokens = self.count_tokens(chunk_text)
            completion_tokens += chunk_tokens
            yield {
                "response": chunk_text,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}
            }

    def _stream_json_formatting(self, json_prompt: str, initial_usage: Dict[str, int], temperature: float) -> Generator[
        Dict[str, Any], None, None]:
        formatting_tokens = 0
        full_json_response = ""
        for chunk in self._make_api_call(json_prompt, stream=True, temperature=temperature):
            if isinstance(chunk, dict) and "error" in chunk:
                yield {
                    "response": self.create_safe_json_response("error", chunk["error"]),
                    "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                                 "completion_tokens": formatting_tokens,
                                                                 "total_tokens": self.count_tokens(
                                                                     json_prompt) + formatting_tokens})
                }
                return

            chunk_text = chunk.get("response", "")
            full_json_response += chunk_text
            formatting_tokens += self.count_tokens(chunk_text)

            try:
                # Attempt to parse the accumulated JSON
                formatted_json = json.loads(full_json_response)
                yield {
                    "response": formatted_json,
                    "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                                 "completion_tokens": formatting_tokens,
                                                                 "total_tokens": self.count_tokens(
                                                                     json_prompt) + formatting_tokens})
                }
                return
            except json.JSONDecodeError:
                # If parsing fails, continue accumulating
                pass

            yield {
                "response": self.create_safe_json_response("streaming", chunk_text),
                "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                             "completion_tokens": formatting_tokens,
                                                             "total_tokens": self.count_tokens(
                                                                 json_prompt) + formatting_tokens})
            }

        # If we've reached this point, we couldn't parse the JSON
        yield {
            "response": self.create_safe_json_response("error", "Failed to create valid JSON from LLM response"),
            "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                         "completion_tokens": formatting_tokens,
                                                         "total_tokens": self.count_tokens(
                                                             json_prompt) + formatting_tokens})
        }

    def _make_api_call(self, prompt: str, stream: bool = False, temperature: float = 1.0) -> Union[
        Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:",
            "stream": stream,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=data, stream=stream)
            response.raise_for_status()

            if stream:
                return self._process_stream(response)
            else:
                return response.json()
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}

    def _process_stream(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    yield {"error": "Failed to parse streaming response"}

    @staticmethod
    def sanitize_input(input_string: str) -> str:
        sanitized = re.sub(r'[^\w\s.,!?-]', '', input_string)
        return sanitized[:1000]  # Limit length to prevent very long inputs

    @staticmethod
    def create_safe_json_response(status: str, message: str, success: bool = None) -> Dict[str, Any]:
        if success is None:
            success = status == "success"
        return {
            "success": success,
            "status": status,
            "response": message
        }

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _combine_usage(usage1: Dict[str, int], usage2: Dict[str, int]) -> Dict[str, int]:
        return {
            "prompt_tokens": usage1["prompt_tokens"] + usage2["prompt_tokens"],
            "completion_tokens": usage1["completion_tokens"] + usage2["completion_tokens"],
            "total_tokens": usage1["total_tokens"] + usage2["total_tokens"]
        }

    def _process_json_response(self, json_response: Dict[str, Any], initial_usage: Dict[str, int]) -> Dict[str, Any]:
        return {
            "response": json_response["response"],
            "usage": self._combine_usage(initial_usage, json_response["usage"])
        }
    @staticmethod
    def _combine_usage(usage1: Dict[str, int], usage2: Dict[str, int]) -> Dict[str, int]:
        return {
            "prompt_tokens": usage1["prompt_tokens"] + usage2["prompt_tokens"],
            "completion_tokens": usage1["completion_tokens"] + usage2["completion_tokens"],
            "total_tokens": usage1["total_tokens"] + usage2["total_tokens"]
        }