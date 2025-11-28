"""
LLM Guidance for RL Agent - Uncertainty-Based

Asks LLM for navigation hints when agent is uncertain about what to do
Uses action entropy to measure uncertainty
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from stable_baselines3 import PPO


class LLMGuidedAgent:
    """
    Wraps a trained PPO agent with LLM guidance
    
    Key idea: When agent is uncertain (high entropy), ask LLM for hint
    """
    
    def __init__(self, model, uncertainty_threshold=1.0, llm_query_fn=None):
        """
        Args:
            model: Trained PPO model
            uncertainty_threshold: Entropy threshold to trigger LLM query
            llm_query_fn: Function that queries LLM (we'll implement this)
        """
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.llm_query_fn = llm_query_fn
        
        # Statistics
        self.llm_queries = 0
        self.total_steps = 0
        self.last_llm_hint = None
        
    def predict(self, obs, env_state=None):
        """
        Predict action, but ask LLM if uncertain
        
        Args:
            obs: Observation from environment
            env_state: Dict with agent_pos, target_pos, walls for LLM context
            
        Returns:
            action: Selected action
            was_uncertain: Whether LLM was queried
        """
        self.total_steps += 1
        
        # Get action probabilities from PPO
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
            
            # Get policy distribution
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs[0]
            
            # Calculate entropy (uncertainty measure)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
            entropy = entropy.item()
        
        # Check if uncertain
        if entropy > self.uncertainty_threshold:
            # Agent is uncertain! Ask LLM for guidance
            if self.llm_query_fn and env_state:
                llm_hint = self.llm_query_fn(env_state)
                self.last_llm_hint = llm_hint
                self.llm_queries += 1
                
                # Modify action based on LLM hint
                action = self._apply_llm_hint(action_probs, llm_hint)
                return action, True
        
        # Normal prediction (agent is confident)
        action = torch.argmax(action_probs).item()
        return action, False
    
    def _apply_llm_hint(self, action_probs, llm_hint):
        """
        Apply LLM hint to action selection
        
        LLM hint is a direction: 'forward', 'backward', 'left', 'right'
        We boost the probability of that action
        """
        action_map = {
            'forward': 0,
            'backward': 1,
            'left': 2,
            'right': 3
        }
        
        suggested_action = action_map.get(llm_hint, None)
        
        if suggested_action is not None:
            # Boost LLM suggested action
            modified_probs = action_probs.clone()
            modified_probs[suggested_action] *= 2.0  # Double the probability
            modified_probs = modified_probs / modified_probs.sum()  # Normalize
            
            # Sample from modified distribution
            action = torch.multinomial(modified_probs, 1).item()
            return action
        else:
            # LLM didn't give valid hint, use original
            return torch.argmax(action_probs).item()
    
    def get_statistics(self):
        """Return statistics about LLM usage"""
        query_rate = self.llm_queries / self.total_steps if self.total_steps > 0 else 0
        return {
            'total_steps': self.total_steps,
            'llm_queries': self.llm_queries,
            'query_rate': query_rate,
            'last_hint': self.last_llm_hint
        }


def simple_llm_query(env_state):
    """
    Simple rule-based LLM simulation for testing
    
    In real version, this would call Claude API
    For now, use heuristics to simulate LLM reasoning
    
    Args:
        env_state: Dict with 'agent_pos', 'target_pos', 'walls'
    
    Returns:
        hint: 'forward', 'backward', 'left', 'right'
    """
    agent_pos = env_state['agent_pos']
    target_pos = env_state['target_pos']
    walls = env_state.get('walls', {})
    
    # Calculate direction to goal
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    
    # Check what walls are around
    wall_forward = walls.get('forward', False)
    wall_left = walls.get('left', False)
    wall_right = walls.get('right', False)
    wall_backward = walls.get('backward', False)
    
    # Simple heuristic: suggest direction toward goal that's not blocked
    if abs(dx) > abs(dy):  # x-distance is larger
        if dx > 0 and not wall_forward:
            return 'forward'
        elif dx < 0 and not wall_backward:
            return 'backward'
    else:  # y-distance is larger
        if dy > 0 and not wall_right:
            return 'right'
        elif dy < 0 and not wall_left:
            return 'left'
    
    # If preferred direction blocked, try any open direction
    if not wall_forward:
        return 'forward'
    elif not wall_right:
        return 'right'
    elif not wall_left:
        return 'left'
    else:
        return 'backward'


def real_llm_query(env_state):
    """
    Real LLM query using Claude API
    
    This will call Claude to get strategic navigation advice
    """
    # TODO: Implement actual Claude API call
    # For now, use simple heuristic
    return simple_llm_query(env_state)


if __name__ == "__main__":
    print("LLM Guidance System")
    print("="*60)
    print("\nHow it works:")
    print("1. Agent acts normally when confident (low entropy)")
    print("2. When uncertain (high entropy), asks LLM for hint")
    print("3. LLM provides direction based on maze state")
    print("4. Agent's action distribution is adjusted toward hint")
    print("\nNext: Integrate with training loop to test effectiveness")
