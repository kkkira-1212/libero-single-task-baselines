"""Explicit causal-v1 index contract; versioned, opt-in; old training defaults are preserved.

LIBERO exported obs[t] is AFTER action[t]. Therefore the next command is
action[t+1]. Episode-local boundary repetition is accompanied by validity masks.
"""
import numpy as np

CONTRACT_VERSION = "libero-poststep-next-command-v1"


def causal_indices(length, observed_index, history=8, future=8):
    if length < 2 or not 0 <= observed_index < length - 1:
        raise ValueError("Need a post-step observation with a real next command")
    raw_history = np.arange(observed_index - history + 1, observed_index + 1)
    raw_future = np.arange(observed_index + 1, observed_index + 1 + future)
    return {
        "history_indices": np.clip(raw_history, 0, length - 1),
        "future_indices": np.clip(raw_future, 0, length - 1),
        "history_valid": raw_history >= 0,
        "future_valid": raw_future < length,
    }

LEGACY_CONTRACT = "libero-same-index-v0"


def sample_options(config):
    contract = config.get("sample_contract", LEGACY_CONTRACT)
    anchor = config.get("sample_anchor_policy", "all")
    if contract not in (LEGACY_CONTRACT, CONTRACT_VERSION):
        raise ValueError("Unknown sample contract: " + str(contract))
    if anchor not in ("all", "has_next_command"):
        raise ValueError("Unknown anchor policy: " + str(anchor))
    if contract == CONTRACT_VERSION and anchor != "has_next_command":
        raise ValueError("Causal contract must exclude anchors without a next command")
    return {"action_offset": int(contract == CONTRACT_VERSION),
            "require_next_action": anchor == "has_next_command"}


def sequence_indices(length, timestep, horizon, n_obs_steps,
                     action_offset=0, require_next_action=False):
    """Pack for unchanged author slice [n_obs_steps-1:n_obs_steps-1+8].

    With offset1, action slot7 contains command[t+1], while obs slot7
    still contains post-step obs[t]. Action slots before7 are unused by A2A.
    """
    if action_offset not in (0, 1):
        raise ValueError("Only explicit legacy0/next-command1 offsets supported")
    if action_offset and not require_next_action:
        raise ValueError("Offset1 requires a real next-command anchor")
    if not 1 <= n_obs_steps <= horizon:
        raise ValueError("Invalid horizon/history")
    if not 0 <= timestep < length - int(require_next_action):
        raise IndexError(timestep)
    raw_obs = np.arange(timestep - n_obs_steps + 1, timestep + 1)
    raw_action = np.arange(timestep - n_obs_steps + 1,
                           timestep - n_obs_steps + 1 + horizon) + action_offset
    return {"obs_indices": np.clip(raw_obs, 0, length - 1),
            "action_indices": np.clip(raw_action, 0, length - 1),
            "obs_padding": (raw_obs < 0) | (raw_obs >= length),
            "action_padding": (raw_action < 0) | (raw_action >= length)}
