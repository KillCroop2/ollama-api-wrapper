import requests
import json
import re
import tiktoken
from typing import Dict, Any, Generator, Union, List

class OllamaApiWrapper:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3", max_retries: int = 3):
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.json_formatting_prompt = (
            "Format the following input into a valid JSON object. "
            "Return ONLY the JSON object, without any additional text, explanations, or code blocks. "
            "Here's the input to format into a JSON object:\n\n"
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_response(self, input_data: Union[str, List[Dict[str, str]]], stream: bool = False,
                          json_mode: bool = False, temperature: float = 1.0, max_tokens: int = 2048,
                          top_p: float = 1.0):
        prompt = self.construct_prompt(input_data)

        if json_mode:
            return self._generate_json_response(prompt, stream, temperature, max_tokens, top_p)
        elif stream:
            return self._stream_response(prompt, temperature, max_tokens, top_p)
        else:
            return self._generate_full_response(prompt, temperature, max_tokens, top_p)

    def construct_prompt(self, input_data: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(input_data, str):
            return self.sanitize_input(input_data)

        prompt_parts = []
        for message in input_data:
            if isinstance(message, dict):
                role = message.get('role', '').capitalize()
                content = self.sanitize_input(message.get('content', ''))
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(message, str):
                prompt_parts.append(self.sanitize_input(message))

        return "\n\n".join(prompt_parts)

    def _make_api_call(self, prompt: str, stream: bool = False, temperature: float = 1.0,
                       max_tokens: int = 2048, top_p: float = 1.0):
        url = f"{self.base_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": f"{prompt}\n\nAssistant:",
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
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

    def _generate_json_response(self, prompt: str, stream: bool, temperature: float,
                                max_tokens: int, top_p: float) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        initial_response = self._generate_full_response(prompt, temperature, max_tokens, top_p)

        if "error" in initial_response["response"]:
            return initial_response

        json_prompt = self.json_formatting_prompt + initial_response["response"]

        if stream:
            return self._stream_json_formatting(json_prompt, initial_response["usage"], temperature, max_tokens, top_p)
        else:
            json_response = self._generate_full_response(json_prompt, temperature, max_tokens, top_p)
            return self._process_json_response(json_response, initial_response["usage"])

    def _generate_full_response(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> Dict[str, Any]:
        prompt_tokens = self.count_tokens(prompt)
        for attempt in range(self.max_retries):
            response = self._make_api_call(prompt, stream=False, temperature=temperature,
                                           max_tokens=max_tokens, top_p=top_p)

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

    def _stream_response(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> Generator[Dict[str, Any], None, None]:
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = 0
        for chunk in self._make_api_call(prompt, stream=True, temperature=temperature,
                                         max_tokens=max_tokens, top_p=top_p):
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

    def _stream_json_formatting(self, json_prompt: str, initial_usage: Dict[str, int], temperature: float,
                                max_tokens: int, top_p: float) -> Generator[Dict[str, Any], None, None]:
        formatting_tokens = 0
        full_json_response = ""
        for chunk in self._make_api_call(json_prompt, stream=True, temperature=temperature,
                                         max_tokens=max_tokens, top_p=top_p):
            if isinstance(chunk, dict) and "error" in chunk:
                yield {
                    "response": self.create_safe_json_response("error", chunk["error"]),
                    "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                                 "completion_tokens": formatting_tokens,
                                                                 "total_tokens": self.count_tokens(json_prompt) + formatting_tokens})
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
                                                                 "total_tokens": self.count_tokens(json_prompt) + formatting_tokens})
                }
                return
            except json.JSONDecodeError:
                # If parsing fails, continue accumulating
                pass

            yield {
                "response": self.create_safe_json_response("streaming", chunk_text),
                "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                             "completion_tokens": formatting_tokens,
                                                             "total_tokens": self.count_tokens(json_prompt) + formatting_tokens})
            }

        # If we've reached this point, we couldn't parse the JSON
        yield {
            "response": self.create_safe_json_response("error", "Failed to create valid JSON from LLM response"),
            "usage": self._combine_usage(initial_usage, {"prompt_tokens": self.count_tokens(json_prompt),
                                                         "completion_tokens": formatting_tokens,
                                                         "total_tokens": self.count_tokens(json_prompt) + formatting_tokens})
        }

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