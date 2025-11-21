"""
Module để track token usage và tính toán chi phí cho Groq API
"""
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Groq pricing per million tokens (USD)
# Pricing có thể thay đổi, cần cập nhật từ https://console.groq.com/docs/pricing
GROQ_PRICING = {
    # Llama models
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},  # $0.05/$0.08 per 1M tokens
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-405b-reasoning": {"input": 2.49, "output": 10.99},
    "llama-3-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3-70b-8192": {"input": 0.59, "output": 0.79},
    
    # Mixtral models
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "mixtral-8x22b-instruct": {"input": 0.55, "output": 0.55},
    
    # Gemma models
    "gemma-7b-it": {"input": 0.07, "output": 0.07},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
    
    # OpenAI OSS models
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},  # $0.10/$0.50 per 1M tokens
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.6},  # $0.15/$0.75 per 1M tokens
    
    # Llama Guard models
    "meta-llama/llama-guard-4-12b": {"input": 0.10, "output": 0.10},  # Estimated, cần kiểm tra pricing chính xác
    
    # Default pricing (nếu model không có trong list)
    "default": {"input": 0.10, "output": 0.10},
}


@dataclass
class TokenUsage:
    """Lưu trữ thông tin token usage cho một request"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    
    def calculate_cost(self, pricing: Optional[Dict[str, float]] = None) -> float:
        """Tính toán chi phí dựa trên pricing"""
        if pricing is None:
            pricing = GROQ_PRICING.get(self.model, GROQ_PRICING["default"])
        
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        self.cost_usd = input_cost + output_cost
        return self.cost_usd
    
    def to_dict(self) -> Dict:
        """Chuyển đổi sang dictionary"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_usd": self.cost_usd
        }


class TokenTracker:
    """Class để track tổng token usage và cost"""
    
    def __init__(self):
        self.usage_by_model: Dict[str, TokenUsage] = defaultdict(lambda: TokenUsage())
        self.total_cost: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.request_count: int = 0
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Thêm token usage vào tracker"""
        total = prompt_tokens + completion_tokens
        
        # Update per-model usage
        usage = self.usage_by_model[model]
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.total_tokens += total
        usage.model = model
        
        # Calculate cost for this model
        pricing = GROQ_PRICING.get(model, GROQ_PRICING["default"])
        usage.calculate_cost(pricing)
        
        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.request_count += 1
        
        # Recalculate total cost
        self.total_cost = sum(u.cost_usd for u in self.usage_by_model.values())
    
    def get_summary(self) -> Dict:
        """Lấy tổng kết token usage và cost"""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "request_count": self.request_count,
            "usage_by_model": {
                model: usage.to_dict() 
                for model, usage in self.usage_by_model.items()
            }
        }
    
    def reset(self):
        """Reset tất cả counters"""
        self.usage_by_model.clear()
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0


# Global tracker instance
_global_tracker = TokenTracker()


def get_global_tracker() -> TokenTracker:
    """Lấy global tracker instance"""
    return _global_tracker


def reset_global_tracker():
    """Reset global tracker"""
    _global_tracker.reset()

