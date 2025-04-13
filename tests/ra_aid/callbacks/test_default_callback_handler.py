"""Unit tests for default_callback_handler.py."""

import time
from unittest.mock import patch, MagicMock
import pytest
from decimal import Decimal, getcontext
from typing import Optional # <<< Added import

from langchain_core.outputs import LLMResult
from ra_aid.callbacks.default_callback_handler import (
    DefaultCallbackHandler,
    MODEL_COSTS, # Import for checking non-tiered models
)
from ra_aid.config import DEFAULT_MODEL

# Set precision for Decimal calculations in tests
getcontext().prec = 16

@pytest.fixture(autouse=True)
def mock_repositories_and_config():
    """Mock repository getters and config repo to prevent side effects."""
    with patch(
        "ra_aid.callbacks.default_callback_handler.get_trajectory_repository"
    ) as mock_get_traj_repo, patch( # Renamed mock for clarity
        "ra_aid.callbacks.default_callback_handler.get_session_repository"
    ) as mock_get_session_repo, patch( # Renamed mock for clarity
        "ra_aid.callbacks.default_callback_handler.get_config_repository" # Mock config repo used in _initialize_model_costs
    ) as mock_get_config_repo: # Renamed mock for clarity
        # Configure the mocks for the *instances* returned by the getters
        mock_traj_repo_instance = MagicMock()
        mock_get_traj_repo.return_value = mock_traj_repo_instance

        mock_session_instance = MagicMock()
        mock_session_record_instance = MagicMock()
        mock_session_record_instance.get_id.return_value = 123  # Example session ID
        mock_session_instance.get_current_session_record.return_value = (
            mock_session_record_instance
        )
        mock_get_session_repo.return_value = mock_session_instance

        # Mock config repo to return default show_cost value
        mock_config_repo_instance = MagicMock()
        mock_config_repo_instance.get.return_value = True # Assume show_cost is true
        mock_get_config_repo.return_value = mock_config_repo_instance

        # Yield the mocked *instances* for potential use in tests if needed,
        # although tests should primarily interact via the handler instance.
        yield mock_traj_repo_instance, mock_session_instance, mock_config_repo_instance


@pytest.fixture
def default_callback_handler(mock_repositories_and_config): # Renamed fixture for clarity
    """Fixture providing a fresh DefaultCallbackHandler instance FOR THE DEFAULT MODEL."""
    # Clear any existing singleton instance to ensure isolation
    if DefaultCallbackHandler in DefaultCallbackHandler._instances:
        del DefaultCallbackHandler._instances[DefaultCallbackHandler]

    # Initialize with a default model (e.g., one with tiers for coverage)
    # __post_init__ will call the mocked getters and assign the mocked instances
    handler = DefaultCallbackHandler(model_name="gpt-4-turbo-preview")
    # Reset all state (including costs) before the test runs
    # Note: reset_all_totals also re-initializes costs for the initial model_name
    # and re-fetches session_id
    handler.reset_all_totals()
    # Session ID should be set correctly by reset_all_totals via the mock
    assert handler.session_totals["session_id"] == 123
    return handler


# --- Tests for _calculate_tiered_cost helper method ---

# Sample tiers for direct testing of the helper method
sample_tiers = {
    0: {"input": Decimal("0.1"), "output": Decimal("0.2")},       # Rate for 0 <= tokens < 100
    100: {"input": Decimal("0.3"), "output": Decimal("0.4")},     # Rate for 100 <= tokens < 200
    200: {"input": Decimal("0.5"), "output": Decimal("0.6")}      # Rate for 200 <= tokens
}
base_input_rate = sample_tiers[0]["input"]
base_output_rate = sample_tiers[0]["output"]

@pytest.mark.parametrize(
    "current_tokens, cumulative_before, cost_type, expected_cost_str",
    [
        # --- Input Cost Tests ---
        # Entirely within base tier (0-99)
        (50, 0, "input", "5.0"),   # 50 * 0.1
        (50, 20, "input", "5.0"),  # 50 * 0.1 (cumulative doesn't change tier)
        (99, 0, "input", "9.9"),   # 99 * 0.1
        # Crossing into second tier (100-199)
        (150, 0, "input", "25.0"),  # (100 * 0.1) + (50 * 0.3) = 10.0 + 15.0 = 25.0
        (50, 80, "input", "11.0"),  # (20 * 0.1) + (30 * 0.3) = 2.0 + 9.0 = 11.0
        (100, 50, "input", "20.0"), # (50 * 0.1) + (50 * 0.3) = 5.0 + 15.0 = 20.0
        # Crossing into third tier (200+)
        (250, 0, "input", "65.0"),  # (100 * 0.1) + (100 * 0.3) + (50 * 0.5) = 10 + 30 + 25 = 65.0
        (150, 100, "input", "55.0"),# (0 * 0.1) + (100 * 0.3) + (50 * 0.5) = 0 + 30 + 25 = 55.0
        (50, 180, "input", "21.0"), # (0 * 0.1) + (20 * 0.3) + (30 * 0.5) = 0 + 6 + 15 = 21.0
        # Starting exactly at a threshold
        (50, 100, "input", "15.0"), # 50 * 0.3
        (50, 200, "input", "25.0"), # 50 * 0.5
        # Zero tokens
        (0, 100, "input", "0.0"),
        # --- Output Cost Tests ---
        (50, 0, "output", "10.0"),  # 50 * 0.2
        (150, 0, "output", "40.0"), # (100 * 0.2) + (50 * 0.4) = 20.0 + 20.0 = 40.0
        (250, 0, "output", "90.0"), # (100 * 0.2) + (100 * 0.4) + (50 * 0.6) = 20 + 40 + 30 = 90.0
        (50, 180, "output", "26.0"), # (20 * 0.4) + (30 * 0.6) = 8.0 + 18.0 = 26.0
    ]
)
def test_calculate_tiered_cost_logic(default_callback_handler, current_tokens, cumulative_before, cost_type, expected_cost_str):
    """Test the _calculate_tiered_cost helper method directly."""
    # Determine base rate based on cost_type for this specific test setup
    base_rate = base_input_rate if cost_type == "input" else base_output_rate

    # Use the handler instance provided by the fixture
    calculated_cost = default_callback_handler._calculate_tiered_cost(
        current_tokens=current_tokens,
        cumulative_tokens_before=cumulative_before,
        tiers=sample_tiers,
        base_rate=base_rate, # Pass the relevant base rate
        cost_type=cost_type
    )
    expected_cost = Decimal(expected_cost_str)
    # Use direct Decimal comparison for exactness in helper tests
    assert calculated_cost == expected_cost


# --- Tests for Integration with on_llm_end ---

def create_mock_response(prompt_tokens: int, completion_tokens: int, model_name: Optional[str] = None) -> MagicMock:
    """Helper to create a mock LLMResult."""
    mock_response = MagicMock(spec=LLMResult)
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    mock_response.llm_output = {"token_usage": token_usage}
    if model_name:
         mock_response.llm_output["model_name"] = model_name
    return mock_response

@pytest.mark.parametrize(
    "model_name, token_batches, expected_final_cost_str",
    [
        # --- Tiered Model: gpt-4-turbo-preview ---
        # Tiers: 0: {in: 1e-5, out: 3e-5} / 128k: {in: 2e-5, out: 6e-5}
        (
            "gpt-4-turbo-preview",
            [(100000, 20000)], # Batch 1: 120k total, all within base tier
            # Cost1 = (100k * 1e-5) + (20k * 3e-5) = 1.0 + 0.6 = 1.6
            "1.6"
        ),
        (
            "gpt-4-turbo-preview",
            [
                (100000, 20000), # Batch 1: Cum: In=100k, Out=20k. Cost = 1.6
                (50000, 10000),  # Batch 2: In: 100k->150k, Out: 20k->30k.
                                 # In Cost: (28k @ base) + (22k @ tier1) = (28k*1e-5)+(22k*2e-5) = 0.28+0.44 = 0.72
                                 # Out Cost: (10k @ base) = 10k*3e-5 = 0.30
                                 # Batch 2 Cost = 0.72 + 0.30 = 1.02
            ],
            # Total Cost = 1.6 + 1.02 = 2.62
            "2.62"
        ),
        # --- Tiered Model: gpt-4-turbo-preview (Edge Case: Hit Threshold Exactly) ---
        (
            "gpt-4-turbo-preview",
            [
                (128000, 10000), # Batch 1: Input hits threshold exactly. Cum: In=128k, Out=10k.
                                 # Cost1 = (128k * 1e-5) + (10k * 3e-5) = 1.28 + 0.30 = 1.58
                (10000, 10000),  # Batch 2: Next input starts exactly at tier 1. Cum: In=138k, Out=20k.
                                 # In Cost: (10k @ tier1) = 10k * 2e-5 = 0.20
                                 # Out Cost: (10k @ base) = 10k * 3e-5 = 0.30
                                 # Batch 2 Cost = 0.20 + 0.30 = 0.50
            ],
            # Total Cost = 1.58 + 0.50 = 2.08
            "2.08"
        ),
        # --- Tiered Model: claude-3-opus-20240229 ---
        # Tiers: 0: {in: 15e-6, out: 75e-6} / 200k: {in: 30e-6, out: 150e-6}
        (
            "claude-3-opus-20240229",
            [(150000, 40000)], # Batch 1: 190k total, all within base tier
            # Cost1 = (150k * 15e-6) + (40k * 75e-6) = 2.25 + 3.0 = 5.25
            "5.25"
        ),
        (
             "claude-3-opus-20240229",
            [
                (150000, 40000), # Batch 1: Cum: In=150k, Out=40k. Cost = 5.25
                (100000, 20000), # Batch 2: In: 150k->250k, Out: 40k->60k
                                 # In Cost: (50k @ base) + (50k @ tier1) = (50k*15e-6)+(50k*30e-6) = 0.75 + 1.5 = 2.25
                                 # Out Cost: (20k @ base) = 20k * 75e-6 = 1.5
                                 # Batch 2 Cost = 2.25 + 1.5 = 3.75
            ],
            # Total Cost = 5.25 + 3.75 = 9.00
            "9.0"
        ),
         # --- Non-Tiered Model (Should use MODEL_COSTS fallback) ---
        (
            "claude-3-haiku-20240307", # Not in TIERED_MODEL_COSTS, uses MODEL_COSTS
             # Rates: in: 0.25e-6, out: 1.25e-6
            [(100000, 50000)],
            # Cost = (100k * 0.25e-6) + (50k * 1.25e-6) = 0.025 + 0.0625 = 0.0875
            "0.0875"
        ),
         (
             "claude-3-haiku-20240307",
             [
                 (100000, 50000), # Batch 1 cost: 0.0875
                 (200000, 100000) # Batch 2 cost: (200k * 0.25e-6) + (100k * 1.25e-6) = 0.05 + 0.125 = 0.175
             ],
             # Total Cost = 0.0875 + 0.175 = 0.2625
             "0.2625"
         ),
          # --- Unknown Model (Should have zero cost) ---
         (
             "completely-unknown-model-xyz",
             [(100000, 50000)],
             "0.0"
         ),
    ]
)
def test_tiered_cost_calculation_integration(model_name, token_batches, expected_final_cost_str):
    """Test on_llm_end calculates cost correctly, including tiers, with isolation."""
    # Create a FRESH handler instance for THIS test parameter run to ensure isolation
    # Clear singleton explicitly before creating a new one for this specific model
    if DefaultCallbackHandler in DefaultCallbackHandler._instances:
        del DefaultCallbackHandler._instances[DefaultCallbackHandler]
    handler = DefaultCallbackHandler(model_name=model_name)
    handler.reset_all_totals() # Ensure clean state upon creation for this model
    handler.session_totals["session_id"] = 123 # Set mock session ID

    expected_final_cost = Decimal(expected_final_cost_str)
    cumulative_prompt = 0
    cumulative_completion = 0
    cumulative_tokens = 0

    # Simulate multiple calls
    call_count = 0
    total_duration = 0.0
    # Use patch on the handler instance created specifically for this test run
    # Need to patch time.time for duration calculation
    with patch.object(handler, '_last_request_time', None), patch("time.time", side_effect = lambda: 100.0 + call_count * 0.1 + 0.1) as mock_time:
        for prompt_tokens, completion_tokens in token_batches:
            # Manually set _last_request_time before the call, simulating on_llm_start
            setattr(handler, '_last_request_time', 100.0 + call_count * 0.1)
            mock_response = create_mock_response(prompt_tokens, completion_tokens)
            handler.on_llm_end(mock_response)
            cumulative_prompt += prompt_tokens
            cumulative_completion += completion_tokens
            cumulative_tokens += prompt_tokens + completion_tokens
            call_count += 1
            total_duration += 0.1 # Each call takes 0.1s based on side_effect

    # Verify final costs (both instance total and session total should match for this test)
    # Use direct Decimal comparison for integration tests as well, for better precision checking
    # assert handler.total_cost == pytest.approx(expected_final_cost)
    # assert handler.session_totals["cost"] == pytest.approx(expected_final_cost)
    assert handler.total_cost == expected_final_cost
    assert handler.session_totals["cost"] == expected_final_cost

    # Verify cumulative totals match the sum of batches
    assert handler.session_totals["input_tokens"] == cumulative_prompt
    assert handler.session_totals["output_tokens"] == cumulative_completion
    assert handler.session_totals["tokens"] == cumulative_tokens
    assert handler.session_totals["duration"] == pytest.approx(total_duration)


# --- Existing tests using the default fixture ---

def test_singleton_pattern(default_callback_handler): # Use renamed fixture
    """Test that DefaultCallbackHandler follows singleton pattern."""
    # Pass model_name when calling again to allow re-initialization if needed
    handler2 = DefaultCallbackHandler(model_name=default_callback_handler.model_name)
    assert handler2 is default_callback_handler


def test_initial_state(default_callback_handler): # Use renamed fixture
    """Test initial state after fixture setup and reset."""
    assert default_callback_handler.prompt_tokens == 0 # Last call tokens
    assert default_callback_handler.completion_tokens == 0
    assert default_callback_handler.total_tokens == 0 # Last call total
    assert default_callback_handler.successful_requests == 0 # Instance lifetime requests
    assert default_callback_handler.total_cost == Decimal("0.0") # Instance lifetime cost

    # Cumulative instance tokens (also reset)
    assert default_callback_handler.cumulative_prompt_tokens == 0
    assert default_callback_handler.cumulative_completion_tokens == 0
    assert default_callback_handler.cumulative_total_tokens == 0

    # Session totals (reset by fixture's reset_all_totals)
    assert default_callback_handler.session_totals == {
        "cost": Decimal("0.0"),
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": 123,  # Restored by fixture
        "duration": 0.0,
    }
    # Check model name set by fixture
    assert default_callback_handler.model_name == "gpt-4-turbo-preview"


def test_on_llm_end_no_token_usage(default_callback_handler): # Use renamed fixture
    """Test on_llm_end with no token usage data doesn't break."""
    mock_response = create_mock_response(0, 0) # Explicitly zero tokens

    with patch("time.time", return_value=100.1):
        default_callback_handler._last_request_time = 100.0
        default_callback_handler.on_llm_end(mock_response)

    assert default_callback_handler.prompt_tokens == 0
    assert default_callback_handler.completion_tokens == 0
    assert default_callback_handler.total_tokens == 0
    assert default_callback_handler.session_totals["tokens"] == 0
    assert default_callback_handler.session_totals["cost"] == Decimal("0.0")


def test_on_llm_end_basic_token_update(default_callback_handler): # Use renamed fixture
    """Test on_llm_end updates basic token counts correctly (using default fixture model)."""
    mock_response = create_mock_response(100, 50) # Using gpt-4-turbo-preview rates

    with patch("time.time", return_value=100.1):
        default_callback_handler._last_request_time = 100.0
        default_callback_handler.on_llm_end(mock_response)

    assert default_callback_handler.prompt_tokens == 100
    assert default_callback_handler.completion_tokens == 50
    assert default_callback_handler.total_tokens == 150 # Last call tokens
    assert default_callback_handler.session_totals["tokens"] == 150 # Cumulative session tokens
    assert default_callback_handler.session_totals["input_tokens"] == 100
    assert default_callback_handler.session_totals["output_tokens"] == 50
    assert default_callback_handler.session_totals["duration"] == pytest.approx(0.1)

    # Verify cost for the default model (gpt-4-turbo-preview, base tier)
    expected_cost = (Decimal(100) * Decimal("1e-5")) + (Decimal(50) * Decimal("3e-5"))
    assert default_callback_handler.total_cost == expected_cost
    assert default_callback_handler.session_totals["cost"] == expected_cost


# Parametrize for non-tiered models using MODEL_COSTS
@pytest.mark.parametrize(
    "model_name, expected_base_in_cost, expected_base_out_cost",
    [
        ("claude-3-sonnet-20240229", Decimal("0.000003"), Decimal("0.000015")),
        ("claude-instant-1", Decimal("0.00000163"), Decimal("0.00000551")),
        # Add more non-tiered models from MODEL_COSTS if needed
    ]
)
def test_cost_calculation_non_tiered(model_name, expected_base_in_cost, expected_base_out_cost):
    """Test cost calculation for models defined in MODEL_COSTS but not TIERED_MODEL_COSTS."""
    # Create a FRESH handler instance for THIS test parameter run
    if DefaultCallbackHandler in DefaultCallbackHandler._instances:
        del DefaultCallbackHandler._instances[DefaultCallbackHandler]
    handler = DefaultCallbackHandler(model_name=model_name)
    handler.reset_all_totals() # Ensure clean state
    handler.session_totals["session_id"] = 123 # Set mock session ID

    # Verify base costs were loaded correctly
    assert handler.input_cost_per_token == expected_base_in_cost
    assert handler.output_cost_per_token == expected_base_out_cost

    mock_response = create_mock_response(100, 50)

    with patch.object(handler, '_last_request_time', None), patch("time.time", return_value=100.1):
        # Set _last_request_time directly before the call
        setattr(handler, '_last_request_time', 100.0)
        handler.on_llm_end(mock_response)

    expected_cost_decimal = (Decimal(100) * expected_base_in_cost) + (Decimal(50) * expected_base_out_cost)
    assert handler.total_cost == expected_cost_decimal
    assert handler.session_totals["cost"] == expected_cost_decimal


def test_unknown_model_zero_cost():
    """Test that a completely unknown model results in zero cost."""
    unknown_model = "completely-unknown-model-xyz"
    # Create a FRESH handler instance for THIS test
    if DefaultCallbackHandler in DefaultCallbackHandler._instances:
        del DefaultCallbackHandler._instances[DefaultCallbackHandler]
    handler = DefaultCallbackHandler(model_name=unknown_model)
    handler.reset_all_totals() # Ensure clean state
    handler.session_totals["session_id"] = 123 # Set mock session ID

    # Verify costs initialized to zero
    assert handler.input_cost_per_token == Decimal("0.0")
    assert handler.output_cost_per_token == Decimal("0.0")

    mock_response = create_mock_response(100, 50)

    with patch.object(handler, '_last_request_time', None), patch("time.time", return_value=100.1):
        # Set _last_request_time directly before the call
        setattr(handler, '_last_request_time', 100.0)
        handler.on_llm_end(mock_response)

    assert handler.total_cost == Decimal("0.0")
    assert handler.session_totals["cost"] == Decimal("0.0")


def test_reset_session_totals(default_callback_handler): # Use renamed fixture
    """Test reset_session_totals clears only session data, preserves instance data."""
    mock_response1 = create_mock_response(100, 50)
    mock_response2 = create_mock_response(200, 100) # Use gpt-4-turbo-preview rates (default handler model)

    # Call 1 (base tier)
    with patch("time.time", return_value=100.1):
        default_callback_handler._last_request_time = 100.0
        default_callback_handler.on_llm_end(mock_response1)
    cost1 = default_callback_handler.session_totals["cost"]
    tokens1 = default_callback_handler.session_totals["tokens"]
    assert tokens1 == 150

    # Call 2 (base tier again as session totals aren't high enough yet)
    with patch("time.time", return_value=100.3):
        default_callback_handler._last_request_time = 100.2
        default_callback_handler.on_llm_end(mock_response2)
    cost2_batch = (Decimal(200) * Decimal("1e-5")) + (Decimal(100) * Decimal("3e-5")) # Cost of *this* batch
    total_cost_before_reset = default_callback_handler.total_cost # Instance total
    session_cost_before_reset = default_callback_handler.session_totals["cost"] # Session total
    session_tokens_before_reset = default_callback_handler.session_totals["tokens"]
    last_call_tokens = default_callback_handler.total_tokens # Tokens from the last call

    assert session_tokens_before_reset == 150 + 300 # 450
    assert session_cost_before_reset == cost1 + cost2_batch
    assert total_cost_before_reset == session_cost_before_reset # Instance and session cost match here
    assert last_call_tokens == 300 # From mock_response2

    # --- Reset Session Totals ---
    default_callback_handler.reset_session_totals()

    # Check session totals are reset
    assert default_callback_handler.session_totals == {
        "cost": Decimal("0.0"),
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "session_id": 123,  # session_id is preserved
        "duration": 0.0,
    }

    # Check instance totals and last call tokens remain unchanged
    assert default_callback_handler.total_cost == total_cost_before_reset
    assert default_callback_handler.successful_requests == 2 # Instance request count
    assert default_callback_handler.prompt_tokens == 200 # Last call prompt tokens
    assert default_callback_handler.completion_tokens == 100 # Last call completion tokens
    assert default_callback_handler.total_tokens == last_call_tokens # Last call total tokens


def test_reset_all_totals(default_callback_handler): # Use renamed fixture
    """Test reset_all_totals clears instance and session data."""
    mock_response = create_mock_response(100, 50)

    with patch("time.time", return_value=100.1):
        default_callback_handler._last_request_time = 100.0
        default_callback_handler.on_llm_end(mock_response)

    # --- Reset All Totals ---
    default_callback_handler.reset_all_totals()

    # Check instance totals
    assert default_callback_handler.total_cost == Decimal("0.0")
    assert default_callback_handler.successful_requests == 0
    assert default_callback_handler.prompt_tokens == 0
    assert default_callback_handler.completion_tokens == 0
    assert default_callback_handler.total_tokens == 0

    # Check cumulative instance totals
    assert default_callback_handler.cumulative_prompt_tokens == 0
    assert default_callback_handler.cumulative_completion_tokens == 0
    assert default_callback_handler.cumulative_total_tokens == 0

    # Check session totals are reset
    assert default_callback_handler.session_totals["cost"] == Decimal("0.0")
    assert default_callback_handler.session_totals["tokens"] == 0
    assert default_callback_handler.session_totals["input_tokens"] == 0
    assert default_callback_handler.session_totals["output_tokens"] == 0
    # session_id should be re-fetched by reset_all_totals via the mock
    assert default_callback_handler.session_totals["session_id"] == 123 # Check if it's restored


def test_get_stats(default_callback_handler): # Use renamed fixture
    """Test get_stats returns correct data structure after a call."""
    mock_response = create_mock_response(100, 50) # Using gpt-4-turbo-preview rates

    with patch("time.time", return_value=100.1):
        default_callback_handler._last_request_time = 100.0
        default_callback_handler.on_llm_end(mock_response)

    stats = default_callback_handler.get_stats()

    # Calculate expected cost using Decimal (base tier for this call)
    expected_cost = (Decimal(100) * Decimal("1e-5")) + (Decimal(50) * Decimal("3e-5"))

    # Check stats related to the last call
    assert stats["last_prompt_tokens"] == 100
    assert stats["last_completion_tokens"] == 50
    assert stats["last_total_tokens"] == 150

    # Check instance lifetime stats
    assert stats["instance_total_cost"] == expected_cost
    assert stats["instance_successful_requests"] == 1
    assert stats["instance_cumulative_prompt_tokens"] == 100
    assert stats["instance_cumulative_completion_tokens"] == 50
    assert stats["instance_cumulative_total_tokens"] == 150

    # Check model info
    assert stats["model_name"] == "gpt-4-turbo-preview"
    assert stats["base_input_cost_per_token"] == Decimal("1e-5")
    assert stats["base_output_cost_per_token"] == Decimal("3e-5")

    # Check session totals within stats
    assert isinstance(stats["session_totals"], dict)
    assert stats["session_totals"]["tokens"] == 150
    assert stats["session_totals"]["input_tokens"] == 100
    assert stats["session_totals"]["output_tokens"] == 50
    assert stats["session_totals"]["cost"] == expected_cost
    assert stats["session_totals"]["duration"] == pytest.approx(0.1)
    assert stats["session_totals"]["session_id"] == 123


@patch('ra_aid.callbacks.default_callback_handler.logger')
def test_handle_callback_update_stores_data(mock_logger, default_callback_handler, mock_repositories_and_config): # Use renamed fixture
    """Verify that _handle_callback_update calls trajectory_repo.create."""
    # Get the actual mock instance held by the handler
    mock_traj_repo_instance = default_callback_handler.trajectory_repo
    assert isinstance(mock_traj_repo_instance, MagicMock) # Verify it's the mock

    test_cost = Decimal("1.23")
    test_duration = 0.5

    # Manually call the method (it's normally called by on_llm_end)
    # Need to update session totals *before* calling _handle_callback_update
    # to simulate the real flow in _update_token_counts
    default_callback_handler.session_totals['cost'] = test_cost
    default_callback_handler.session_totals['tokens'] = 150
    default_callback_handler.session_totals['input_tokens'] = 100
    default_callback_handler.session_totals['output_tokens'] = 50

    default_callback_handler._handle_callback_update(
        total_tokens=150,
        prompt_tokens=100,
        completion_tokens=50,
        cost=test_cost, # Pass the calculated cost (tiered or flat)
        duration=test_duration,
    )

    # Assert create was called once on the instance held by the handler
    mock_traj_repo_instance.create.assert_called_once() # <<< Assert on the instance mock
    call_args, call_kwargs = mock_traj_repo_instance.create.call_args # <<< Get args from instance mock

    # Check specific arguments passed to create
    assert call_kwargs.get("record_type") == "model_usage"
    # Cost is stored as float
    assert call_kwargs.get("current_cost") == pytest.approx(float(test_cost))
    assert call_kwargs.get("input_tokens") == 100
    assert call_kwargs.get("output_tokens") == 50
    assert call_kwargs.get("session_id") == 123 # From mock

    step_data = call_kwargs.get("step_data", {})
    assert step_data.get("duration") == pytest.approx(test_duration)
    assert step_data.get("model") == default_callback_handler.model_name

    # Verify cumulative stats are logged in step_data (reflecting the state *after* update)
    assert step_data.get("cumulative_session_cost") == pytest.approx(float(test_cost))
    assert step_data.get("cumulative_session_tokens") == 150
    assert step_data.get("cumulative_session_input_tokens") == 100
    assert step_data.get("cumulative_session_output_tokens") == 50

