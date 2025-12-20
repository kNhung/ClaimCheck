from pydantic import BaseModel
from typing import List, Optional 

class Settings(BaseModel):
    # Fast API settings
    APP_NAME: str = "ClaimCheck"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOW_ORIGINS: List[str] = ["*"]
    
    # Fact Check settings
    FACTCHECKER_MODEL_NAME: str = "qwen2.5:0.5b"
    FACTCHECKER_MAX_ACTIONS: int = 2
    FACTCHECKER_EMBED_DEVICE: Optional[str] = None
    FACTCHECKER_BI_ENCODER: str = "paraphrase-multilingual-MiniLM-L12-v2"
    FACTCHECKER_CROSS_ENCODER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FACTCHECKER_BERT_MODEL: str = "distilbert-base-multilingual-cased"
    
    # Fact Check Tools 
    FACTCHECKER_SCRAPE_TIMEOUT: float = 4 
    
    # API Keys
    SERPER_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_CX: Optional[str] = None
    
    # Paths
    REPORTS_DIR: str = "reports"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()