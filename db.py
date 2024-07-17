import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import secrets
import logging

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'auth_plugin': 'mysql_native_password'  # Use native password authentication
}


def create_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None


def verify_api_key(api_key):
    connection = create_connection()
    if connection is None:
        logging.error("Failed to create database connection")
        return False

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM api_keys WHERE key_value = %s AND is_active = TRUE"
        cursor.execute(query, (api_key,))
        result = cursor.fetchone()
        is_valid = result is not None
        logging.info(f"API key verification result: {'Valid' if is_valid else 'Invalid'}")
        return is_valid
    except Error as e:
        logging.error(f"Error verifying API key: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def generate_api_key():
    return secrets.token_urlsafe(32)


def create_api_key():
    connection = create_connection()
    if connection is None:
        return None

    try:
        cursor = connection.cursor()
        new_key = generate_api_key()
        query = "INSERT INTO api_keys (key_value) VALUES (%s)"
        cursor.execute(query, (new_key,))
        connection.commit()
        return new_key
    except Error as e:
        print(f"Error creating API key: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_allowed_models(api_key):
    connection = create_connection()
    if connection is None:
        logging.error("Failed to create database connection")
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT DISTINCT m.id, m.object, m.created, m.owned_by, m.permission, m.root, m.parent
        FROM models m
        LEFT JOIN api_key_model_access akma ON m.id = akma.model_id
        LEFT JOIN api_keys ak ON akma.api_key_id = ak.id
        WHERE m.is_public = TRUE OR ak.key_value = %s
        """
        cursor.execute(query, (api_key,))
        results = cursor.fetchall()
        logging.info(f"Retrieved {len(results)} allowed models for API key {api_key}")
        return results
    except Error as e:
        logging.error(f"Error getting allowed models: {e}")
        return []
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
