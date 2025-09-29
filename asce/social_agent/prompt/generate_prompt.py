"""
Centralized prompt generation module.

This module provides a unified interface to generate prompts for different models and thinking modes.
It imports and calls the appropriate prompt generator based on the specified parameters.
"""

from typing import Dict, List, Any, Optional, Union

# Import all prompt generators
from asce.social_agent.prompt.generate_aim_prompt import generate_aim_prompt
from asce.social_agent.prompt.generate_ecb_prompt import generate_ecb_prompt
from asce.social_agent.prompt.generate_tpb_prompt import generate_tpb_prompt
from asce.social_agent.prompt.generate_crc_prompt import generate_crc_prompt
from asce.social_agent.prompt.generate_crc_dbn_prompt import generate_crc_dbn_prompt
from asce.social_agent.prompt.generate_oasis_prompt import generate_oasis_prompt
from asce.social_agent.prompt.generate_label_prompt import generate_label_prompt
from asce.social_agent.prompt.modular_prompt import generate_modular_prompt_label
from asce.social_agent.prompt.generate_normal_prompt import generate_normal_prompt

# Define a mapping of think_mode to prompt generator functions for ASCE
ASCE_PROMPT_GENERATORS = {
    'AIM': generate_aim_prompt,
    'ECB': generate_ecb_prompt,
    'TPB': generate_tpb_prompt,
    'CRC': generate_crc_prompt,
    'CRC-DBN': generate_crc_dbn_prompt
}

# Define a mapping for OASIS
OASIS_PROMPT_GENERATORS = {
    'OASIS': generate_oasis_prompt
}

# Define a mapping for label generation
LABEL_PROMPT_GENERATORS = {
    'LABEL': generate_modular_prompt_label
}

def generate_prompt(
    user: Dict[str, Any],
    env_prompt: str,
    cognitive_profile: Dict[str, Any],
    prompt_mode: str = 'asce',
    think_mode: str = 'CRC',
    action_space_prompt: Optional[str] = None,
    cognition_space_dict: Optional[Dict[str, Any]] = None,
    causal_prompt: Optional[str] = None,
    memory_content: Optional[str] = None,
    time_step: Optional[int] = None,
    target_context: Optional[str] = None
) -> str:
    """
    Generate a prompt based on the specified parameters.

    Args:
        user: User information dictionary
        env_prompt: Environment prompt content
        cognitive_profile: Cognitive profile dictionary
        prompt_mode: Mode for prompt generation ('asce', 'oasis', or 'label')
        think_mode: Thinking mode ('AIM', 'ECB', 'TPB', 'CRC', 'CRC-DBN')
        action_space_prompt: Action space prompt content
        cognition_space_dict: Cognition space dictionary
        causal_prompt: Causal prompt content
        memory_content: Memory content
        time_step: Current time step
        target_context: Target context for label generation

    Returns:
        str: Generated prompt
    """
    # Convert prompt_mode to lowercase for case-insensitive comparison
    prompt_mode = prompt_mode.lower()

    # For label generation
    if prompt_mode == 'label':
        return generate_modular_prompt_label(
            user=user,
            env_prompt=env_prompt,
            cognitive_profile=cognitive_profile,
            action_space_prompt=action_space_prompt,
            cognition_space_dict=cognition_space_dict,
            target_context=target_context,
            time_step=time_step,
            think_mode=think_mode
        )

        # return generate_label_prompt(
        #     user=user,
        #     env_prompt=env_prompt,
        #     cognitive_profile=cognitive_profile,
        #     action_space_prompt=action_space_prompt,
        #     cognition_space_dict=cognition_space_dict,
        #     target_context=target_context,
        #     time_step=time_step,
        #     think_mode=think_mode
        # )

    # For OASIS mode
    elif prompt_mode == 'oasis':
        return generate_oasis_prompt(
            user=user,
            env_prompt=env_prompt,
            cognitive_profile=cognitive_profile,
            action_space_prompt=action_space_prompt,
            cognition_space_dict=cognition_space_dict,
            memory_content=memory_content,
            time_step=time_step,
            think_mode=think_mode
        )
    elif prompt_mode == 'normal':
        return generate_normal_prompt(
            user=user,
            env_prompt=env_prompt,
            cognitive_profile=cognitive_profile,
            action_space_prompt=action_space_prompt,
            cognition_space_dict=cognition_space_dict,
            causal_prompt=causal_prompt,
            memory_content=memory_content,
            time_step=time_step,
            think_mode=think_mode
        )
    

    # For ASCE mode (default)
    else:  # prompt_mode == 'asce' or any other value
        # Get the appropriate generator based on think_mode
        generator = ASCE_PROMPT_GENERATORS.get(think_mode)

        # If the think_mode is not recognized, default to CRC
        if generator is None:
            print(f"Warning: Unrecognized think_mode '{think_mode}'. Defaulting to 'CRC'.")
            generator = generate_crc_prompt

        # Generate the prompt
        return generator(
            user=user,
            env_prompt=env_prompt,
            cognitive_profile=cognitive_profile,
            action_space_prompt=action_space_prompt,
            cognition_space_dict=cognition_space_dict,
            causal_prompt=causal_prompt,
            memory_content=memory_content,
            time_step=time_step,
            think_mode=think_mode
        )