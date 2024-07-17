-- Drop existing tables if they exist
DROP TABLE IF EXISTS api_key_model_access;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS api_keys;

-- Table for API keys (unchanged)
CREATE TABLE IF NOT EXISTS api_keys (
    id INT AUTO_INCREMENT PRIMARY KEY,
    key_value VARCHAR(255) UNIQUE NOT NULL,
    user_id INT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Updated table for models to match OpenAI's structure
CREATE TABLE IF NOT EXISTS models (
    id VARCHAR(255) PRIMARY KEY,
    object VARCHAR(50) DEFAULT 'model',
    created INT UNSIGNED,
    owned_by VARCHAR(255),
    permission JSON,
    root VARCHAR(255),
    parent VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE
);

-- Table for API key to model mappings (unchanged)
CREATE TABLE IF NOT EXISTS api_key_model_access (
    id INT AUTO_INCREMENT PRIMARY KEY,
    api_key_id INT,
    model_id VARCHAR(255),
    FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
    FOREIGN KEY (model_id) REFERENCES models(id),
    UNIQUE (api_key_id, model_id)
);

-- Insert some example models
INSERT INTO models (id, created, owned_by, permission, root, parent, is_public) VALUES
('llama3:latest', UNIX_TIMESTAMP(), 'openai', '[]', 'llama3:latest', NULL, TRUE),
('gemma:7b', UNIX_TIMESTAMP(), 'openai', '[]', 'gemma:7b', NULL, TRUE)