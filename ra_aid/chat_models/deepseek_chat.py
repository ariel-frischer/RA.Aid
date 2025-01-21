from langchain_openai import ChatOpenAI
from typing import Any, Optional, Dict


class ChatDeepseekReasoner(ChatOpenAI):
    """ChatDeepseekReasoner with custom params handling for R1/reasoner models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def invocation_params(
        self, options: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        params = super().invocation_params(options, **kwargs)

        # Remove unsupported params for R1 models
        params.pop("temperature", None)
        params.pop("top_p", None)
        params.pop("presence_penalty", None)
        params.pop("frequency_penalty", None)

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _generate to ensure message alternation."""
        # Print original messages for debugging
        print("\n[Message Debug] Original messages:")
        for i, msg in enumerate(messages):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            print(f"Message {i}: {role} - {msg.content[:100]}...")

        # Process messages to ensure alternation
        processed = []
        prev_role = None
        
        for msg in messages:
            current_role = "user" if isinstance(msg, HumanMessage) else "assistant"
            
            if prev_role == current_role:
                # Merge consecutive same-role messages
                if processed:
                    processed[-1].content += f"\n\n{msg.content}"
            else:
                processed.append(msg)
                prev_role = current_role

        # Print processed messages for debugging
        print("\n[Message Debug] Processed messages:")
        for i, msg in enumerate(processed):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            print(f"Message {i}: {role} - {msg.content[:100]}...")

        # Call parent implementation with processed messages
        return super()._generate(processed, stop=stop, run_manager=run_manager, **kwargs)

    async def acompletion_with_retry(self, *args, **kwargs) -> Any:
        response = await super().acompletion_with_retry(*args, **kwargs)
        if response.choices:
            msg = response.choices[0].message
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs.get(
                "reasoning"
            ):
                print(f"\n[Reasoning Chain]\n{msg.additional_kwargs['reasoning']}\n")
        return response

    # async def _acompletion_with_retry(self, *args, **kwargs) -> Any:
    #     # Force usage of our custom client
    #     kwargs['client'] = self.client
    #     return await super()._acompletion_with_retry(*args, **kwargs)
