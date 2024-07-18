from functools import wraps
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import os
import time
from typing import Dict, Any, Generator, List
import logging
import tiktoken

from llm_wrapper import OllamaApiWrapper
from db import verify_api_key, create_api_key, get_allowed_models

app = Flask(__name__)
CORS(app)

ollama_wrapper = OllamaApiWrapper()
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Invalid or missing API key"}), 401

        api_key = auth_header.split(' ')[1]
        if not verify_api_key(api_key):
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)

    return decorated

@app.route('/v1/chat/completions', methods=['POST'])
@require_api_key
def chat_completions():
    data = request.json
    messages = data.get('messages', [])

    if not any(message.get('role') == 'system' for message in messages):
        default_system_message = {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
        messages.insert(0, default_system_message)

    stream = data.get('stream', False)
    model = data.get('model', 'llama3')
    temperature = data.get('temperature', 1.0)
    max_tokens = data.get('max_tokens', 2048)  # New: maximum length
    top_p = data.get('top_p', 1.0)  # New: top_p
    response_format = data.get('response_format', {})
    json_mode = response_format.get('type') == 'json_object' if response_format else False

    api_key = request.headers.get('Authorization').split(' ')[1]
    allowed_models = get_allowed_models(api_key)

    logging.info(f"Requested model: {model}")
    logging.info(f"Allowed models: {[m['id'] for m in allowed_models]}")

    if not any(m['id'] == model for m in allowed_models):
        logging.warning(f"Access denied for model {model} with API key {api_key}")
        return jsonify({"error": f"You do not have access to the model: {model}"}), 403

    prompt = construct_prompt(messages)
    prompt_tokens = count_tokens(prompt)

    if stream:
        return Response(
            stream_with_context(
                generate_streaming_response(messages, model, json_mode, temperature, max_tokens, top_p, prompt_tokens)),
            content_type='text/event-stream')
    else:
        response = generate_complete_response(messages, model, json_mode, temperature, max_tokens, top_p, prompt_tokens)
        return jsonify(response)

@app.route('/v1/models', methods=['GET'])
def get_models():
    try:
        auth_header = request.headers.get('Authorization')
        api_key = auth_header.split(' ')[1] if auth_header and auth_header.startswith('Bearer ') else None

        allowed_models = get_allowed_models(api_key)

        formatted_models = []
        for model in allowed_models:
            formatted_model = {
                "id": model['id'],
                "object": "model",
                "created": model['created'],
                "owned_by": model['owned_by'],
                "permission": model['permission'],
                "root": model['root'],
                "parent": model['parent'],
                "description": model['description'],  # New: description
                "strengths": model['strengths'],  # New: strengths
                "price": {  # New: price
                    "prompt": model['price_prompt'],
                    "completion": model['price_completion']
                }
            }
            formatted_models.append(formatted_model)

        logging.info(f"Returning {len(formatted_models)} models")
        return jsonify({
            "object": "list",
            "data": formatted_models
        })
    except Exception as e:
        logging.error(f"Unexpected error in get_models: {str(e)}")
        return jsonify({"error": {"message": "An unexpected error occurred", "type": "server_error"}}), 500


@app.route('/v1/api_keys', methods=['POST'])
def create_new_api_key():
    new_key = create_api_key()
    if new_key:
        return jsonify({"api_key": new_key})
    else:
        return jsonify({"error": "Failed to create API key"}), 500


# Error handler for 404 Not Found
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": {"message": "Not found", "type": "not_found"}}), 404


# Error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": {"message": "Internal server error", "type": "internal_error"}}), 500


def construct_prompt(messages: list) -> str:
    """Construct a single prompt string from the message list."""
    prompt = ""
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        prompt += f"{role.capitalize()}: {content}\n"
    return prompt.strip()


def generate_streaming_response(messages: List[Dict[str, str]], model: str, json_mode: bool, temperature: float,
                                max_tokens: int, top_p: float, prompt_tokens: int) -> Generator[str, None, None]:
    accumulated_json = ""
    completion_tokens = 0
    for chunk in ollama_wrapper.generate_response(messages, stream=True, json_mode=json_mode, temperature=temperature,
                                                  max_tokens=max_tokens, top_p=top_p):
        chunk_text = chunk.get("response", "")
        completion_tokens += count_tokens(chunk_text)
        total_tokens = prompt_tokens + completion_tokens

        if json_mode:
            accumulated_json += chunk_text
            try:
                parsed_json = json.loads(accumulated_json)
                formatted_chunk = json.dumps(parsed_json, indent=2)
                yield f"data: {json.dumps(format_chunk_as_openai_response({'response': formatted_chunk}, model, prompt_tokens, completion_tokens, total_tokens))}\n\n"
                accumulated_json = ""
            except json.JSONDecodeError:
                pass
        else:
            yield f"data: {json.dumps(format_chunk_as_openai_response(chunk, model, prompt_tokens, completion_tokens, total_tokens))}\n\n"

    if json_mode and accumulated_json:
        try:
            parsed_json = json.loads(accumulated_json)
            formatted_chunk = json.dumps(parsed_json, indent=2)
            yield f"data: {json.dumps(format_chunk_as_openai_response({'response': formatted_chunk}, model, prompt_tokens, completion_tokens, total_tokens))}\n\n"
        except json.JSONDecodeError:
            logging.warning("Failed to parse final JSON chunk. Returning unformatted.")
            yield f"data: {json.dumps(format_chunk_as_openai_response({'response': accumulated_json}, model, prompt_tokens, completion_tokens, total_tokens))}\n\n"

    yield "data: [DONE]\n\n"

def generate_complete_response(messages: List[Dict[str, str]], model: str, json_mode: bool, temperature: float,
                               max_tokens: int, top_p: float, prompt_tokens: int) -> Dict[str, Any]:
    response = ollama_wrapper.generate_response(messages, stream=False, json_mode=json_mode, temperature=temperature,
                                                max_tokens=max_tokens, top_p=top_p)

    if json_mode:
        try:
            parsed_json = json.loads(response["response"])
            formatted_json = json.dumps(parsed_json, indent=2)
            response["response"] = formatted_json
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON response. Returning unformatted response.")

    completion_tokens = count_tokens(response["response"])
    total_tokens = prompt_tokens + completion_tokens

    return format_response_as_openai_response(response, model, prompt_tokens, completion_tokens, total_tokens)

def format_response_as_openai_response(response: Dict[str, Any], model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-" + os.urandom(4).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["response"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }


def format_chunk_as_openai_response(chunk: Dict[str, Any], model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-" + os.urandom(4).hex(),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": chunk["response"]
                },
                "finish_reason": None
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }


if __name__ == '__main__':
    app.run(debug=True, port=5000)
