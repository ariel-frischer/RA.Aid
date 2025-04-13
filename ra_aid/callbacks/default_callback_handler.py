import threading
import time
from langchain.chat_models.base import BaseChatModel
import litellm
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Optional, Union, Any, List
from decimal import Decimal, getcontext # <<< Ensure Decimal is imported

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ra_aid.model_detection import (
    get_model_name_from_chat_model,
    get_provider_from_chat_model,
)
from ra_aid.utils.singleton import Singleton
from ra_aid.database.repositories.trajectory_repository import get_trajectory_repository
from ra_aid.database.repositories.session_repository import get_session_repository
from ra_aid.logging_config import get_logger

# Added imports
from ra_aid.config import DEFAULT_SHOW_COST
from ra_aid.database.repositories.config_repository import get_config_repository
from ra_aid.console.formatting import cpm

logger = get_logger(__name__)

getcontext().prec = 16 # Set Decimal precision

# --- START: Original MODEL_COSTS ---
MODEL_COSTS = {
    "claude-3-7-sonnet-20250219": {
        "input": Decimal("0.000003"),
        "output": Decimal("0.000015"),
    },
    "claude-3-opus-20240229": {
        "input": Decimal("0.000015"),
        "output": Decimal("0.000075"),
    },
    "claude-3-sonnet-20240229": {
        "input": Decimal("0.000003"),
        "output": Decimal("0.000015"),
    },
    "claude-3-haiku-20240307": {
        "input": Decimal("0.00000025"),
        "output": Decimal("0.00000125"),
    },
    "claude-2": {
        "input": Decimal("0.00001102"),
        "output": Decimal("0.00003268"),
    },
    "claude-instant-1": {
        "input": Decimal("0.00000163"),
        "output": Decimal("0.00000551"),
    },
    "google/gemini-2.5-pro-exp-03-25:free": {
        "input": Decimal("0"),
        "output": Decimal("0"),
    },
    # Newly added models
    "weaver-ai": {
        "input": Decimal("0.001875"),
        "output": Decimal("0.00225"),
    },
    "airoboros-v1": {
        "input": Decimal("0.0005"),
        "output": Decimal("0.0005"),
    },
    "mistral-nemo": {
        "input": Decimal("0.00015"),
        "output": Decimal("0.00015"),
    },
    "pixtral-12b": {
        "input": Decimal("0.00015"),
        "output": Decimal("0.00015"),
    },
    "mistral-large-24b11": {
        "input": Decimal("0.002"),
        "output": Decimal("0.006"),
    },
}
# --- END: Original MODEL_COSTS ---


class DefaultCallbackHandler(BaseCallbackHandler, metaclass=Singleton):
    # --- START: Add TIERED_MODEL_COSTS ---
    # Structure: { model_name: { threshold_start: {'input': rate, 'output': rate}, ... } }
    # The rate applies to tokens *from* the threshold_start *up to* the next threshold.
    # The threshold 0 typically holds the base rate for the first block of tokens.
    TIERED_MODEL_COSTS: Dict[str, Dict[int, Dict[str, Decimal]]] = {
        "gpt-4-turbo-preview": {
            # Base rate for 0 to 128k tokens (exclusive end)
            0: {"input": Decimal("0.00001"), "output": Decimal("0.00003")},
            # Rate for tokens *from* 128k onwards
            128000: {"input": Decimal("0.00002"), "output": Decimal("0.00006")},
        },
        "claude-3-opus-20240229": {
             # Base rate for 0 to 200k tokens (exclusive end)
             0: {"input": Decimal("0.000015"), "output": Decimal("0.000075")},
             # Rate for tokens *from* 200k onwards
             200000: {"input": Decimal("0.000030"), "output": Decimal("0.000150")},
        }
        # Add other models with tiered pricing here
    }
    # --- END: Add TIERED_MODEL_COSTS ---

    def __init__(self, model_name: str, provider: Optional[str] = None):
        super().__init__()
        self._lock = threading.Lock()
        self._initialize(model_name, provider)

    def _initialize(self, model_name: str, provider: Optional[str] = None):
        with self._lock:
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.successful_requests = 0
            self.total_cost = Decimal("0.0")
            self.model_name = model_name
            self.provider = provider
            self._last_request_time = None
            self.__post_init__()

    cumulative_total_tokens: int = 0
    cumulative_prompt_tokens: int = 0
    cumulative_completion_tokens: int = 0

    trajectory_repo = None
    session_repo = None

    # These will store the base rates (typically the rate for the first tier/lowest usage)
    input_cost_per_token: Decimal = Decimal("0.0")
    output_cost_per_token: Decimal = Decimal("0.0")

    session_totals = {
        "cost": Decimal("0.0"),
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": None,
        "duration": 0.0,
    }

    def __post_init__(self):
        try:
            if not hasattr(self, "trajectory_repo") or self.trajectory_repo is None:
                self.trajectory_repo = get_trajectory_repository()

            if not hasattr(self, "session_repo") or self.session_repo is None:
                self.session_repo = get_session_repository()

            if self.session_repo:
                current_session = self.session_repo.get_current_session_record()
                if current_session:
                    self.session_totals["session_id"] = current_session.get_id()

            self._initialize_model_costs() # Initialize base costs
        except Exception as e:
            logger.error(f"Failed to initialize callback handler: {e}", exc_info=True)

    def _initialize_model_costs(self) -> None:
        """Initializes the base input/output costs per token."""
        # Attempt to get from litellm first (might provide base rates)
        try:
            model_info = litellm.get_model_info(
                model=self.model_name, custom_llm_provider=self.provider
            )
            if model_info:
                input_cost = model_info.get("input_cost_per_token", 0.0)
                output_cost = model_info.get("output_cost_per_token", 0.0)
                self.input_cost_per_token = Decimal(str(input_cost))
                self.output_cost_per_token = Decimal(str(output_cost))
                # Don't return early if we have tiered costs defined,
                # as litellm might only give the base rate.
                # We'll rely on the tiered logic later if tiers exist for this model.
        except Exception as e:
            logger.debug(f"Could not get model info from litellm: {e}")

        # Fallback/Override: Check our hardcoded costs (base and tiered)
        # Use base rate from TIERED_MODEL_COSTS if available at threshold 0
        model_tiers = self.TIERED_MODEL_COSTS.get(self.model_name)
        if model_tiers and 0 in model_tiers:
             base_tier_costs = model_tiers[0]
             self.input_cost_per_token = base_tier_costs.get("input", self.input_cost_per_token)
             self.output_cost_per_token = base_tier_costs.get("output", self.output_cost_per_token)
        elif self.model_name in MODEL_COSTS: # Fallback to non-tiered MODEL_COSTS if no tier 0
            model_cost = MODEL_COSTS[self.model_name]
            self.input_cost_per_token = model_cost["input"]
            self.output_cost_per_token = model_cost["output"]
        else: # If completely unknown, default to 0 (or keep litellm's value if any)
             # Ensure they are Decimal if not set previously
             if not isinstance(self.input_cost_per_token, Decimal):
                 self.input_cost_per_token = Decimal("0.0")
             if not isinstance(self.output_cost_per_token, Decimal):
                 self.output_cost_per_token = Decimal("0.0")

             # Optionally warn if costs are effectively zero
             config_repo = get_config_repository()
             show_cost = config_repo.get("show_cost", DEFAULT_SHOW_COST)
             if show_cost and self.input_cost_per_token == 0 and self.output_cost_per_token == 0:
                 cpm(
                     f"Could not determine base costs for model '{self.model_name}'. Costs will be calculated as 0.",
                     border_style="yellow",
                 )


    def __repr__(self) -> str:
        return (
            f"Tokens Used (Last Call): {self.prompt_tokens + self.completion_tokens}\n" # Clarified this is last call
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests (Instance): {self.successful_requests}\n" # Clarified scope
            f"Total Cost (Instance USD): ${self.total_cost:.6f}" # Clarified scope
        )

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        try:
            self._last_request_time = time.time()
            # Update model name if provided in serialized data (might change mid-session?)
            # TODO: Consider implications if model name changes - should costs reset/re-initialize?
            # For now, assume model name set at init is the primary one.
            # if "name" in serialized:
            #     self.model_name = serialized.get("name", self.model_name)
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}", exc_info=True)

    # --- START: Add _calculate_tiered_cost method ---
    def _calculate_tiered_cost(
        self,
        current_tokens: int,
        cumulative_tokens_before: int,
        tiers: Dict[int, Dict[str, Decimal]],
        base_rate: Decimal,
        cost_type: str,
    ) -> Decimal:
        """
        Calculates the cost for a batch of tokens based on tiered pricing.

        Args:
            current_tokens: The number of tokens in the current request.
            cumulative_tokens_before: Total session tokens of this type *before* this request.
            tiers: Pricing tiers for the specific model (threshold -> {input/output rates}).
            base_rate: The base cost per token (used if no tier 0 or as fallback).
            cost_type: 'input' or 'output'.

        Returns:
            The calculated cost for the current_tokens as a Decimal.
        """
        if not tiers or current_tokens <= 0:
            # Use base_rate if no tiers defined or no tokens
            return Decimal(current_tokens) * base_rate

        total_cost = Decimal("0.0")
        tokens_processed = 0
        cumulative_after_request = cumulative_tokens_before + current_tokens

        # Sort thresholds to process tiers in ascending order of token count
        sorted_thresholds = sorted(tiers.keys())

        # Start with the rate defined at threshold 0, or the base_rate as fallback
        current_rate = tiers.get(0, {}).get(cost_type, base_rate)

        for i, threshold in enumerate(sorted_thresholds):
            # Rate for the tier *starting* at this threshold
            tier_rate = tiers[threshold].get(cost_type)
            if tier_rate is None:
                 logger.warning(f"Missing '{cost_type}' rate for threshold {threshold} in model {self.model_name}. Using previous rate: {current_rate}")
                 tier_rate = current_rate # Use rate from previous tier if missing

            # Determine the token range for the *previous* rate segment
            # This segment ends just before the current threshold starts.
            segment_start = sorted_thresholds[i - 1] if i > 0 else 0
            segment_end = threshold # The current threshold marks the end of the previous segment

            # Calculate how many tokens from the current request fall into this segment
            # based on the cumulative count *before* this request.
            # Start counting from where the previous cumulative total left off.
            effective_start = max(segment_start, cumulative_tokens_before)
            # End counting at the end of this segment or when all current tokens are accounted for.
            effective_end = min(segment_end, cumulative_after_request)

            tokens_in_segment = max(0, effective_end - effective_start)
            tokens_to_cost = min(current_tokens - tokens_processed, tokens_in_segment)

            if tokens_to_cost > 0:
                 total_cost += Decimal(tokens_to_cost) * current_rate
                 tokens_processed += tokens_to_cost

            # Update the rate for the next segment (the one starting at 'threshold')
            current_rate = tier_rate

            # Stop if all tokens for the current request have been costed
            if tokens_processed >= current_tokens:
                break

        # Cost any remaining tokens (those falling into the highest tier)
        remaining_tokens = current_tokens - tokens_processed
        if remaining_tokens > 0:
            # These tokens are charged at the rate of the last processed tier
            total_cost += Decimal(remaining_tokens) * current_rate

        return total_cost
    # --- END: Add _calculate_tiered_cost method ---


    def _extract_token_usage(self, response: LLMResult) -> dict:
        """Extract token usage information from various response formats."""
        token_usage = {}

        # Check in llm_output (common case)
        if hasattr(response, "llm_output") and response.llm_output:
            llm_output = response.llm_output
            if isinstance(llm_output.get("token_usage"), dict):
                token_usage = llm_output["token_usage"]
            elif isinstance(llm_output.get("usage"), dict):
                usage = llm_output["usage"]
                token_usage["prompt_tokens"] = usage.get("input_tokens", 0)
                token_usage["completion_tokens"] = usage.get("output_tokens", 0)
                token_usage["total_tokens"] = usage.get("total_tokens", 0)
            # Update model name if available in output (e.g., from LiteLLM routing)
            if "model_name" in llm_output:
                 # Check if model actually changed and re-initialize costs if needed?
                 # current_model = self.model_name
                 # new_model = llm_output["model_name"]
                 # if new_model != current_model:
                 #     logger.info(f"Model changed during session from {current_model} to {new_model}. Re-initializing costs.")
                 #     self._initialize_model_costs() # This might reset base rates based on the new model
                 self.model_name = llm_output["model_name"]


        # Check in response.usage (alternative structure)
        elif hasattr(response, "usage") and response.usage:
            usage = response.usage
            token_usage["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
            token_usage["completion_tokens"] = getattr(usage, "completion_tokens", 0)
            token_usage["total_tokens"] = getattr(usage, "total_tokens", 0)

        # Check in generations (less common, sometimes in streaming or specific providers)
        if not token_usage.get("prompt_tokens") and not token_usage.get("completion_tokens"):
             if hasattr(response, "generations") and response.generations:
                 for gen_list in response.generations:
                     if gen_list:
                         gen = gen_list[0] # Typically use the first generation
                         # Check generation_info (e.g., Anthropic)
                         if hasattr(gen, "generation_info") and isinstance(gen.generation_info, dict):
                             gen_info = gen.generation_info
                             if isinstance(gen_info.get("usage"), dict):
                                 usage = gen_info["usage"]
                                 # Anthropic uses 'input_tokens', 'output_tokens'
                                 token_usage["prompt_tokens"] = usage.get("input_tokens", 0)
                                 token_usage["completion_tokens"] = usage.get("output_tokens", 0)
                                 # Calculate total if not present
                                 if not token_usage.get("total_tokens"):
                                     token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                                 break # Found usage, exit loop

                         # Check message.usage_metadata (e.g., Gemini)
                         if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata") and gen.message.usage_metadata:
                              usage_metadata = gen.message.usage_metadata
                              # Gemini often provides candidate_token_count, prompt_token_count
                              token_usage["prompt_tokens"] = usage_metadata.get("prompt_token_count", 0)
                              token_usage["completion_tokens"] = usage_metadata.get("candidates_token_count", 0) # Note the 's'
                              token_usage["total_tokens"] = usage_metadata.get("total_token_count", 0)
                              # Verify consistency
                              if token_usage["total_tokens"] == 0 and (token_usage["prompt_tokens"] > 0 or token_usage["completion_tokens"] > 0):
                                   token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                              break # Found usage, exit loop

        # Ensure all keys exist, default to 0
        token_usage.setdefault("prompt_tokens", 0)
        token_usage.setdefault("completion_tokens", 0)
        token_usage.setdefault("total_tokens", 0)

        # Final consistency check: if total is zero but others aren't, sum them.
        if token_usage["total_tokens"] == 0 and (token_usage["prompt_tokens"] > 0 or token_usage["completion_tokens"] > 0):
             token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]

        return token_usage


    def _update_token_counts(self, token_usage: dict, duration: float) -> None:
        """Update token counts and costs, considering tiered pricing."""
        with self._lock:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

            # If total is provided but breakdown isn't, approximate
            if total_tokens > 0 and prompt_tokens == 0 and completion_tokens == 0:
                # Simple approximation, might need refinement based on typical usage patterns
                prompt_tokens = int(total_tokens * 0.8) # Assume prompt is larger
                completion_tokens = total_tokens - prompt_tokens

            # Get cumulative session counts *before* this request
            cumulative_prompt_before = self.session_totals["input_tokens"]
            cumulative_completion_before = self.session_totals["output_tokens"]

            # Check if the current model has defined tiered pricing
            model_tiers = self.TIERED_MODEL_COSTS.get(self.model_name)

            if model_tiers:
                # Calculate cost using the tiered logic helper function
                input_cost = self._calculate_tiered_cost(
                    current_tokens=prompt_tokens,
                    cumulative_tokens_before=cumulative_prompt_before,
                    tiers=model_tiers,
                    base_rate=self.input_cost_per_token, # Pass the initialized base rate
                    cost_type="input",
                )
                output_cost = self._calculate_tiered_cost(
                    current_tokens=completion_tokens,
                    cumulative_tokens_before=cumulative_completion_before,
                    tiers=model_tiers,
                    base_rate=self.output_cost_per_token, # Pass the initialized base rate
                    cost_type="output",
                )
                logger.debug(f"Tiered cost calculation for {self.model_name}: Input={input_cost}, Output={output_cost}")
            else:
                # Calculate costs using flat base rates (original logic)
                input_cost = Decimal(prompt_tokens) * self.input_cost_per_token
                output_cost = Decimal(completion_tokens) * self.output_cost_per_token
                logger.debug(f"Flat cost calculation for {self.model_name}: Input={input_cost}, Output={output_cost}")


            cost = input_cost + output_cost

            # Update instance-level cumulative totals (for the lifetime of the handler instance)
            self.cumulative_prompt_tokens += prompt_tokens
            self.cumulative_completion_tokens += completion_tokens
            self.cumulative_total_tokens += total_tokens

            # Store token counts for the *current* request (for __repr__, get_stats)
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens # Sum of prompt and completion for this specific call

            # Update overall instance cost and request count
            self.total_cost += cost
            self.successful_requests += 1

            # Update session totals (crucial for tracking across multiple calls in a session)
            self.session_totals["cost"] += cost
            self.session_totals["tokens"] += total_tokens # Add this call's total tokens
            self.session_totals["input_tokens"] += prompt_tokens # Add this call's input tokens
            self.session_totals["output_tokens"] += completion_tokens # Add this call's output tokens
            self.session_totals["duration"] += duration

            # Persist trajectory data for this specific LLM call, including its calculated cost
            self._handle_callback_update(
                 total_tokens=total_tokens,
                 prompt_tokens=prompt_tokens,
                 completion_tokens=completion_tokens,
                 cost=cost, # Pass the calculated cost (tiered or flat)
                 duration=duration,
            )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            duration = 0.0
            if self._last_request_time is not None:
                duration = time.time() - self._last_request_time
                self._last_request_time = None # Reset for the next call
            else:
                logger.debug("No request start time found for on_llm_end, duration=0")

            token_usage = self._extract_token_usage(response)
            logger.debug(f"Extracted token usage: {token_usage}")

            self._update_token_counts(token_usage, duration)

        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}", exc_info=True)


    def _handle_callback_update(
        self,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Decimal, # Receive the already calculated cost
        duration: float,
    ) -> None:
        """Stores token usage and cost information for the current trajectory step."""
        try:
            if not self.trajectory_repo:
                logger.warning("Trajectory repository not available, cannot store usage data.")
                return

            if not self.session_totals["session_id"]:
                logger.warning("session_id not initialized, cannot store usage data.")
                return

            # Convert Decimal cost to float for JSON serialization
            cost_float = float(cost)

            trajectory_record = self.trajectory_repo.create(
                record_type="model_usage",
                current_cost=cost_float, # Cost specifically for this LLM call
                input_tokens=prompt_tokens, # Input tokens for this call
                output_tokens=completion_tokens, # Output tokens for this call
                session_id=self.session_totals["session_id"],
                step_data={
                    "duration": duration,
                    "model": self.model_name,
                    # Log cumulative session stats *after* this update for context
                    "cumulative_session_cost": float(self.session_totals['cost']),
                    "cumulative_session_tokens": self.session_totals['tokens'],
                    "cumulative_session_input_tokens": self.session_totals['input_tokens'],
                    "cumulative_session_output_tokens": self.session_totals['output_tokens'],
                },
            )
            logger.debug(f"Stored trajectory record for model usage: Cost={cost_float}, Tokens={total_tokens}")
        except Exception as e:
            logger.error(f"Failed to store token usage data: {e}", exc_info=True)

    def reset_session_totals(self) -> None:
        """Resets only the session-specific accumulators."""
        try:
            current_session_id = self.session_totals.get("session_id")
            self.session_totals = {
                "cost": Decimal("0.0"),
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "session_id": current_session_id, # Keep the session ID
                "duration": 0.0,
            }
            logger.info(f"Session totals reset for session_id: {current_session_id}")
        except Exception as e:
            logger.error(f"Error resetting session totals: {e}", exc_info=True)

    def reset_all_totals(self) -> None:
        """Resets all instance and session accumulators."""
        with self._lock:
            # Reset instance-level stats
            self.total_tokens = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.successful_requests = 0
            self.total_cost = Decimal("0.0")
            self._last_request_time = None

            # Reset session totals
            self.reset_session_totals() # Resets session dict, keeps session_id if possible

            # Reset cumulative instance totals
            self.cumulative_total_tokens = 0
            self.cumulative_prompt_tokens = 0
            self.cumulative_completion_tokens = 0

            # Re-initialize base model costs (model name might be reset or needed)
            # Note: This assumes self.model_name is still relevant or will be updated.
            self._initialize_model_costs()

            # Try to re-fetch current session ID if session_repo is available
            if self.session_repo and self.session_totals.get("session_id") is None:
                current_session = self.session_repo.get_current_session_record()
                if current_session:
                    self.session_totals["session_id"] = current_session.get_id()

            logger.info("All callback handler totals reset.")


    def get_stats(self) -> Dict[str, Union[int, float, Decimal, str]]:
        """Returns a snapshot of the current statistics."""
        try:
            # Create a copy of session_totals to avoid modification issues
            session_totals_copy = dict(self.session_totals)
            # Ensure cost is included, even if Decimal
            session_totals_copy['cost'] = self.session_totals.get('cost', Decimal('0.0'))

            return {
                # Stats for the last processed request
                "last_prompt_tokens": self.prompt_tokens,
                "last_completion_tokens": self.completion_tokens,
                "last_total_tokens": self.total_tokens,
                # Instance lifetime stats
                "instance_total_cost": self.total_cost,
                "instance_successful_requests": self.successful_requests,
                "instance_cumulative_prompt_tokens": self.cumulative_prompt_tokens,
                "instance_cumulative_completion_tokens": self.cumulative_completion_tokens,
                "instance_cumulative_total_tokens": self.cumulative_total_tokens,
                # Current model info
                "model_name": self.model_name,
                "provider": self.provider,
                "base_input_cost_per_token": self.input_cost_per_token,
                "base_output_cost_per_token": self.output_cost_per_token,
                # Current session stats
                "session_totals": session_totals_copy,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {}

# --- ContextVar and Context Manager (Unchanged) ---
default_callback_var: ContextVar[Optional[DefaultCallbackHandler]] = ContextVar(
    "default_callback", default=None
)


@contextmanager
def get_default_callback(
    model_name: str,
    provider: Optional[str] = None,
) -> DefaultCallbackHandler:
    cb = DefaultCallbackHandler(model_name=model_name, provider=provider)
    token = default_callback_var.set(cb)
    try:
        yield cb
    finally:
        default_callback_var.reset(token)


# --- Initialization Functions (Unchanged, rely on DefaultCallbackHandler instantiation) ---
def _initialize_callback_handler_internal(
    model_name: str, provider: Optional[str] = None, track_cost: bool = True
) -> tuple[Optional[DefaultCallbackHandler], dict]:
    cb = None
    stream_config = {"callbacks": []}

    if not track_cost:
        logger.debug("Cost tracking is disabled, skipping callback handler")
        return cb, stream_config

    try:
        logger.debug(f"Initializing DefaultCallbackHandler for model {model_name}")
        cb = DefaultCallbackHandler(model_name, provider)
        stream_config["callbacks"].append(cb)
        logger.debug(f"Callback handler initialized with base costs: Input={cb.input_cost_per_token}, Output={cb.output_cost_per_token}")
    except Exception as e:
         logger.error(f"Failed to initialize DefaultCallbackHandler: {e}", exc_info=True)
         return None, {"callbacks": []} # Return None if init fails

    return cb, stream_config


def initialize_callback_handler(
    model: BaseChatModel, track_cost: bool = True
) -> tuple[Optional[DefaultCallbackHandler], dict]:
    model_name = get_model_name_from_chat_model(model)
    provider = get_provider_from_chat_model(model)
    if not model_name:
        logger.error("Could not determine model name from model object. Cannot initialize callback handler.")
        return None, {"callbacks": []}
    return _initialize_callback_handler_internal(model_name, provider, track_cost)

