# Ollama API Wrapper

This project provides a Flask-based API wrapper for Ollama, allowing you to interact with Ollama models using an OpenAI-like API interface. It includes features such as API key authentication, model access control, and JSON response formatting.

## Features

- OpenAI-like API for Ollama models
- API key authentication
- Model access control
- Streaming and non-streaming responses
- JSON mode for structured outputs
- Token counting and usage tracking

## Prerequisites

- Python 3.7+
- MySQL database
- Ollama running locally or on a accessible server

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/KillCroop2/ollama-api-wrapper.git
   cd ollama-api-wrapper
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   - Copy the `.env.template` file to `.env`
   - Fill in the necessary database credentials in the `.env` file

4. Set up the database:
   - Run the SQL commands in `database.sql` to create the necessary tables

## Configuration

Update the `.env` file with your database credentials:

```
DB_HOST=<Your MySQL host>
DB_PORT=<Your MySQL port>
DB_USER=<Your MySQL username>
DB_PASSWORD=<Your MySQL password>
DB_NAME=<Your database name>
```

## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The API will be available at `http://localhost:5000`

3. Use the following endpoints:
   - `/v1/chat/completions`: For chat completions
   - `/v1/models`: To list available models
   - `/v1/api_keys`: To create a new API key

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

Request body:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "llama2",
  "stream": false,
  "temperature": 0.7
}
```

### List Models

```
GET /v1/models
```

### Create API Key

```
POST /v1/api_keys
```

## Development

- `app.py`: Main Flask application
- `db.py`: Database operations
- `llm_wrapper.py`: Ollama API wrapper
- `database.sql`: SQL schema

## License

This project is licensed under the MIT License.