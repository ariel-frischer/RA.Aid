from langchain_openai import ChatOpenAI
from typing import Any, Optional, Dict
import json

class ChatDeepseekReasoner(ChatOpenAI):
    """ChatDeepseekReasoner with custom params handling for R1/reasoner models."""
    
    def invocationParams(self, options: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        params = super().invocation_params(options, **kwargs)
        
        # Remove unsupported params for R1 models
        params.pop("temperature", None)
        params.pop("top_p", None) 
        params.pop("presence_penalty", None)
        params.pop("frequency_penalty", None)
        
        return params

    async def acompletion_with_retry(self, *args, **kwargs) -> Any:
        response = await super()._acompletion_with_retry(*args, **kwargs)
        if response.choices:
            msg = response.choices[0].message
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get("reasoning"):
                print(f"\n[Reasoning Chain]\n{msg.additional_kwargs['reasoning']}\n")
        return response
