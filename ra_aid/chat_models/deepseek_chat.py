from langchain_openai import ChatOpenAI
from typing import Any, Optional, Dict
from openai import OpenAI

class ChatDeepseekReasoner(ChatOpenAI):
    """ChatDeepseekReasoner with custom params handling for R1/reasoner models."""
    
    def __init__(self, *args, **kwargs):
        # Explicitly create DeepSeek client
        kwargs['client'] = OpenAI(
            base_url=kwargs.pop('base_url', 'https://api.deepseek.com'),
            api_key=kwargs.pop('api_key', ''),
        )
        super().__init__(*args, **kwargs)
    
    def invocation_params(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        params = super().invocation_params(options, **kwargs)
        
        # Remove unsupported params for R1 models
        params.pop("temperature", None)
        params.pop("top_p", None)
        params.pop("presence_penalty", None)
        params.pop("frequency_penalty", None)
        
        return params

    async def _acompletion_with_retry(self, *args, **kwargs) -> Any:
        # Force usage of our custom client
        kwargs['client'] = self.client
        return await super()._acompletion_with_retry(*args, **kwargs)
