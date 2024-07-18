-- database.sql

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

-- Updated table for models with new fields
CREATE TABLE IF NOT EXISTS models (
    id VARCHAR(255) PRIMARY KEY,
    object VARCHAR(50) DEFAULT 'model',
    created INT UNSIGNED,
    owned_by VARCHAR(255),
    permission JSON,
    root VARCHAR(255),
    parent VARCHAR(255),
    is_public BOOLEAN DEFAULT FALSE,
    description TEXT,
    strengths TEXT,
    price_prompt FLOAT,
    price_completion FLOAT
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

-- Insert example models with new fields
INSERT INTO models (id, created, owned_by, permission, root, parent, is_public, description, strengths, price_prompt, price_completion) VALUES
('llama3:latest', UNIX_TIMESTAMP(), 'meta', '[]', 'llama3:latest', NULL, TRUE, 'Latest version of LLAMA3 model', 'General purpose, strong reasoning capabilities', 0.0003, 0.002),
('gemma:7b', UNIX_TIMESTAMP(), 'google', '[]', 'gemma:7b', NULL, TRUE, 'Gemma 7B model', 'Efficient for various NLP tasks', 0.00028, 0.0018);