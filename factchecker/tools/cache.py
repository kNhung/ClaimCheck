import redis
import json
import os
from typing import Optional

# dotenv.load_dotenv() is called in factchecker/__init__.py
HOST = os.getenv('REDIS_HOST', 'localhost')
PORT = int(os.getenv('REDIS_PORT', 6379))
DB = int(os.getenv('REDIS_DB', 0))
PASSWORD = os.getenv('REDIS_PASSWORD', None)


class RedisCache:
    def __init__(self, host=HOST, port=PORT, db=DB, password=PASSWORD):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError:
            print("Warning: Redis connection failed. Caching will be disabled.")
            self.redis_client = None

    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        try:
            value = self.redis_client.get(key)
            if value:
                # Parse JSON if it's stored as JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except redis.RedisError:
            return None

    def set(self, key: str, value: str, expire: int = 3600) -> bool:  # Default 1 hour
        """Set value in cache with expiration"""
        if not self.redis_client:
            return False
        try:
            # Store as JSON string
            json_value = json.dumps(value) if not isinstance(value, str) else value
            return self.redis_client.setex(key, expire, json_value)
        except redis.RedisError:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        try:
            return bool(self.redis_client.delete(key))
        except redis.RedisError:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis_client:
            return False
        try:
            return bool(self.redis_client.exists(key))
        except redis.RedisError:
            return False

# Global cache instance
_cache_instance = None

def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        # Configure from environment variables
        host = os.getenv('REDIS_HOST', 'localhost')
        port = int(os.getenv('REDIS_PORT', 6379))
        db = int(os.getenv('REDIS_DB', 0))
        password = os.getenv('REDIS_PASSWORD', None)
        _cache_instance = RedisCache(host=host, port=port, db=db, password=password)
    return _cache_instance