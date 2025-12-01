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
    
    def __init__(self, model, uncertainty_threshold=1.0, llm_query_fn=None, 
                 llm_boost=2.0, boost_type='multiplicative'):
        """
        Args:
            model: Trained PPO model
            uncertainty_threshold: Entropy threshold to trigger LLM query
            llm_query_fn: Function that queries LLM (we'll implement this)
            llm_boost: How much to boost LLM-suggested action (default: 2.0x for multiplicative, 0.2 for additive)
            boost_type: 'multiplicative' (default) or 'additive'
        """
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.llm_query_fn = llm_query_fn
        self.llm_boost = llm_boost
        self.boost_type = boost_type
        
        # Statistics
        self.llm_queries = 0
        self.llm_followed = 0  # Times agent followed LLM advice
        self.llm_ignored = 0   # Times agent ignored LLM advice
        self.total_steps = 0
        self.uncertain_steps = 0  # Steps where entropy > threshold
        self.last_llm_hint = None
        self.hint_distribution = {'forward': 0, 'backward': 0, 'left': 0, 'right': 0}
        
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
            self.uncertain_steps += 1
            
            # Agent is uncertain! Ask LLM for guidance
            if self.llm_query_fn and env_state:
                llm_hint = self.llm_query_fn(env_state)
                self.last_llm_hint = llm_hint
                self.llm_queries += 1
                
                # Track hint distribution
                if llm_hint in self.hint_distribution:
                    self.hint_distribution[llm_hint] += 1
                
                # Get original best action
                original_action = torch.argmax(action_probs).item()
                
                # Modify action based on LLM hint
                action = self._apply_llm_hint(action_probs, llm_hint)
                
                # Track if we followed LLM
                if action == self._hint_to_action(llm_hint):
                    self.llm_followed += 1
                else:
                    self.llm_ignored += 1
                
                return action, True
        
        # Normal prediction (agent is confident)
        action = torch.argmax(action_probs).item()
        return action, False
    
    def _hint_to_action(self, hint):
        """Convert hint string to action index"""
        action_map = {
            'forward': 0,
            'backward': 1,
            'left': 2,
            'right': 3
        }
        return action_map.get(hint, None)
    
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
            
            if self.boost_type == 'additive':
                # Additive boost: add fixed amount
                modified_probs[suggested_action] += self.llm_boost
            else:  # multiplicative (default)
                # Multiplicative boost: multiply by factor
                modified_probs[suggested_action] *= self.llm_boost
            
            # Normalize to ensure probabilities sum to 1
            modified_probs = modified_probs / modified_probs.sum()
            
            # Sample from modified distribution
            action = torch.multinomial(modified_probs, 1).item()
            return action
        else:
            # LLM didn't give valid hint, use original
            return torch.argmax(action_probs).item()
    
    def get_statistics(self):
        """Return detailed statistics about LLM usage"""
        query_rate = self.llm_queries / self.total_steps if self.total_steps > 0 else 0
        uncertain_rate = self.uncertain_steps / self.total_steps if self.total_steps > 0 else 0
        follow_rate = self.llm_followed / self.llm_queries if self.llm_queries > 0 else 0
        
        return {
            'total_steps': self.total_steps,
            'uncertain_steps': self.uncertain_steps,
            'uncertain_rate': uncertain_rate,
            'llm_queries': self.llm_queries,
            'query_rate': query_rate,
            'llm_followed': self.llm_followed,
            'llm_ignored': self.llm_ignored,
            'follow_rate': follow_rate,
            'hint_distribution': self.hint_distribution,
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
    Real LLM query using OpenAI API (GPT-4o-mini)
    
    Args:
        env_state: Dict with 'agent_pos', 'target_pos', 'walls'
    
    Returns:
        hint: 'forward', 'backward', 'left', 'right'
    """
    import os
    
    # Try to load from .env file if exists
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load from .env in root directory
    except ImportError:
        pass  # dotenv not installed, will use environment variable
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found, using simple heuristic")
        return simple_llm_query(env_state)
    
    try:
        from openai import OpenAI
        
        # Set up OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Extract state
        agent_pos = env_state['agent_pos']
        target_pos = env_state['target_pos']
        walls = env_state['walls']
        
        # Calculate relative direction in natural terms
        dx = target_pos[0] - agent_pos[0]  # positive = need to go right
        dy = target_pos[1] - agent_pos[1]  # positive = need to go up
        
        # Convert walls to natural directions for LLM
        # Internal mapping: forward=x+1(right), backward=x-1(left), right=y+1(up), left=y-1(down)
        natural_walls = {
            'up': walls['right'],      # y+1
            'down': walls['left'],      # y-1
            'right': walls['forward'],  # x+1
            'left': walls['backward']   # x-1
        }
        
        # Create simple, natural prompt
        prompt = f"""You are helping a robot navigate a maze on a grid.

CURRENT SITUATION:
- Robot is at position: row {agent_pos[0]}, column {agent_pos[1]}
- Goal is at position: row {target_pos[0]}, column {target_pos[1]}

To reach the goal, the robot needs to:
- Move {'RIGHT' if dx > 0 else 'LEFT' if dx < 0 else 'neither horizontally'}: {abs(dx)} steps
- Move {'UP' if dy > 0 else 'DOWN' if dy < 0 else 'neither vertically'}: {abs(dy)} steps

WALLS (blocked directions):
  * UP: {'❌ BLOCKED' if natural_walls['up'] else '✓ OPEN'}
  * DOWN: {'❌ BLOCKED' if natural_walls['down'] else '✓ OPEN'}
  * LEFT: {'❌ BLOCKED' if natural_walls['left'] else '✓ OPEN'}
  * RIGHT: {'❌ BLOCKED' if natural_walls['right'] else '✓ OPEN'}

The robot is uncertain which direction to move. Which direction should it go?

Respond with EXACTLY ONE WORD: up, down, left, or right
"""
        
        # Call GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a maze navigation expert. You must respond with exactly one word: up, down, left, or right. Choose the direction that avoids walls and moves toward the goal."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.1
        )
        
        # Extract hint (LLM gives natural directions: up/down/left/right)
        hint_natural = response.choices[0].message.content.strip().lower()
        
        # Map natural directions to internal actions
        # Natural: up/down/left/right (what LLM says)
        # Internal: forward/backward/left/right (what agent uses)
        # Mapping: up→right(y+1), down→left(y-1), right→forward(x+1), left→backward(x-1)
        natural_to_internal = {
            'up': 'right',       # y+1
            'down': 'left',      # y-1  
            'right': 'forward',  # x+1
            'left': 'backward'   # x-1
        }
        
        hint_internal = natural_to_internal.get(hint_natural, None)
        
        # Validate hint
        if hint_internal:
            return hint_internal
        else:
            # LLM gave invalid response, fallback to heuristic
            print(f"Warning: LLM gave invalid hint '{hint_natural}', using heuristic")
            return simple_llm_query(env_state)
    
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return simple_llm_query(env_state)


def cached_llm_query(env_state, cache={}):
    """
    Cached version to avoid repeated API calls for same situation
    
    This saves money by caching LLM responses
    """
    # Create cache key from state
    key = (
        env_state['agent_pos'],
        env_state['target_pos'],
        tuple(sorted(env_state['walls'].items()))
    )
    
    if key not in cache:
        cache[key] = real_llm_query(env_state)
    
    return cache[key]


if __name__ == "__main__":
    print("LLM Guidance System")
    print("="*60)
    print("\nHow it works:")
    print("1. Agent acts normally when confident (low entropy)")
    print("2. When uncertain (high entropy), asks LLM for hint")
    print("3. LLM provides direction based on maze state")
    print("4. Agent's action distribution is adjusted toward hint")
    print("\nNext: Integrate with training loop to test effectiveness")