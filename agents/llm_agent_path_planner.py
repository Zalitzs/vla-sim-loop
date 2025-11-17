# FILE: agents/llm_agent_path_planner.py
"""
LLM Agent with Full Path Planning

Key changes from original:
1. Plans ENTIRE path at once (not step-by-step)
2. Uses gpt-4o-mini (cheapest model)
3. Limits replanning to 3 attempts max
4. Uncapped tokens (2000 max)
5. Returns JSON with path as list
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import re

load_dotenv()


def grid_to_text(grid):
    """Convert numpy grid to visual text representation"""
    symbols = {
        0: '.',  # empty
        1: '#',  # wall
        2: 'C',  # cube (you)
        3: 'T'   # target
    }
    
    # Flip so high X is at top (visual intuition)
    grid_flipped = np.flipud(grid)
    
    rows = []
    for row in grid_flipped:
        row_symbols = [symbols[cell] for cell in row]
        row_text = ' '.join(row_symbols)
        rows.append(row_text)
    
    return '\n'.join(rows)


class LLMPathPlannerAgent:
    """LLM Agent that plans full path at once"""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0, max_replans=3):
        """Initialize the path planning agent
        
        Args:
            model: OpenAI model to use
            temperature: Sampling temperature (0.0 = deterministic)
            max_replans: Maximum number of times to replan if path fails
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_replans = max_replans
        
        # Statistics
        self.num_calls = 0
        self.total_tokens = 0
        
        # Path planning state
        self.planned_path = []       # List of visual directions
        self.path_index = 0          # Current position in path
        self.replan_count = 0        # How many times we've replanned
        self.last_pos = None         # Track if we're stuck
        
        print(f"✓ LLM Path Planner initialized")
        print(f"  Model: {model}")
        print(f"  Max replans: {max_replans}")
        
    def get_action(self, env, verbose=False):
        """Get next action from planned path, or replan if needed"""
        
        cube_pos = env.get_cube_grid_pos()
        target_pos = env.get_target_grid_pos()
        
        # Check if already at goal
        if cube_pos == target_pos:
            return 'forward'
        
        # Check if last action failed (we're stuck)
        action_failed = False
        if self.last_pos is not None and cube_pos == self.last_pos:
            action_failed = True
            print(f"  ⚠️  Last action FAILED - we're stuck at {cube_pos}")
        
        # Decide if we need to (re)plan
        need_to_plan = False
        
        if len(self.planned_path) == 0:
            # No path yet - need initial plan
            need_to_plan = True
            print(f"  📋 No path yet - planning initial path...")
            
        elif self.path_index >= len(self.planned_path):
            # Exhausted current path - need to replan
            need_to_plan = True
            print(f"  📋 Path exhausted - replanning...")
            
        elif action_failed:
            # Hit a wall - need to replan
            need_to_plan = True
            print(f"  📋 Hit obstacle - replanning...")
        
        # Plan new path if needed
        if need_to_plan:
            # Check replan limit
            if self.replan_count >= self.max_replans:
                print(f"  ❌ Replan limit reached ({self.max_replans})! Giving up.")
                # Return random action as fallback
                import random
                return random.choice(['forward', 'backward', 'left', 'right'])
            
            # Plan new path
            self.replan_count += 1
            print(f"  🔄 Replan #{self.replan_count}/{self.max_replans}")
            
            grid = env.get_grid_state()
            grid_text = grid_to_text(grid)
            
            # DEBUG: Always show what LLM sees
            print(f"\n=== VISUAL GRID LLM SEES ===")
            print(grid_text)
            print(f"Cube at: {cube_pos}")
            print(f"Target at: {target_pos}")
            print(f"===========================\n")
            
            if verbose:
                print(f"\n=== Visual Grid ===")
                print(grid_text)
                print(f"===================\n")
            
            prompt = self._create_prompt(grid_text, cube_pos, target_pos)
            
            if verbose:
                print(f"=== Full Prompt ===")
                print(prompt)
                print(f"===================\n")
            
            response = self._call_llm(prompt)
            
            print(f"\n--- LLM Response ---")
            print(response)
            print(f"--- End Response ---\n")
            
            # Parse path from JSON response
            self.planned_path = self._parse_path(response)
            self.path_index = 0
            
            if not self.planned_path:
                print(f"  ❌ Failed to parse path! Using fallback.")
                return 'forward'
            
            print(f"  ✓ New path planned: {self.planned_path} ({len(self.planned_path)} steps)")
        
        # Get next action from path
        if self.path_index < len(self.planned_path):
            visual_action = self.planned_path[self.path_index]
            self.path_index += 1
            
            print(f"  Step {self.path_index}/{len(self.planned_path)}: Using '{visual_action}' from plan")
            
            # Map to environment action
            actual_action = self._map_visual_to_env(visual_action)
            
            # Remember position for next time
            self.last_pos = cube_pos
            
            return actual_action
        else:
            # Shouldn't reach here, but fallback
            print(f"  ⚠️  Path index out of bounds!")
            return 'forward'
    
    def _create_prompt(self, grid_text, cube_pos, target_pos):
        """Create prompt asking for COMPLETE path as JSON"""
        
        # Calculate relative direction for hint  
        dx = target_pos[0] - cube_pos[0]  # Positive = target has higher X
        dy = target_pos[1] - cube_pos[1]  # Positive = target has higher Y
        
        # After flipud, higher X appears at BOTTOM visually
        x_direction = "down" if dx > 0 else ("up" if dx < 0 else "same row")
        # Higher Y appears to the RIGHT visually
        y_direction = "RIGHT" if dy > 0 else ("LEFT" if dy < 0 else "same column")
        
        # Make column positions explicit
        cube_col = cube_pos[1]
        target_col = target_pos[1]
        
        prompt = f"""Navigate maze from C to T. Plan COMPLETE path.

GRID:
{grid_text}

ANALYZE THE GRID:
- Find C in the grid above (you are here)
- Find T in the grid above (your goal)
- C is in column {cube_col} (counting from left, starting at 0)
- T is in column {target_col} (counting from left, starting at 0)
- Visually: T is {x_direction} and {y_direction} from C

DIRECTIONS:
- "up" = move toward TOP of display
- "down" = move toward BOTTOM of display
- "left" = move toward LEFT edge (lower column numbers)
- "right" = move toward RIGHT edge (higher column numbers)

Since T is in column {target_col} and C is in column {cube_col}:
- If {target_col} > {cube_col}, you need to go RIGHT
- If {target_col} < {cube_col}, you need to go LEFT

LEGEND:
C = you, T = goal, # = wall (avoid!), . = empty

TASK: 
1. First, explain your reasoning (which way to go and why)
2. Then output the JSON path

FORMAT YOUR RESPONSE LIKE THIS:
REASONING: [explain your analysis here - where is C, where is T, what obstacles are there, which direction to go]

JSON: {{"path": ["right", "right", "down"]}}

Your response:"""
        
        return prompt
    
    def _call_llm(self, prompt):
        """Call OpenAI API with prompt"""
        try:
            self.num_calls += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert maze navigator. You analyze grids visually and plan complete paths. You always respond with valid JSON containing a path array using only: up, down, left, right."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000  # Uncapped for full path planning
            )
            
            response_text = response.choices[0].message.content.strip()
            self.total_tokens += response.usage.total_tokens
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            return '{"path": []}'  # Return empty path on error
    
    def _parse_path(self, response):
        """Extract path array from JSON response
        
        The response may contain reasoning text before the JSON.
        We look for the JSON object anywhere in the response.
        
        Returns:
            list: List of direction strings (up/down/left/right)
        """
        # Remove markdown code blocks if present
        response_clean = re.sub(r'```json\s*', '', response)
        response_clean = re.sub(r'```\s*', '', response_clean)
        response_clean = response_clean.strip()
        
        # Try to find JSON object in the response
        # Look for pattern {"path": [...]}
        json_match = re.search(r'\{[^{}]*"path"\s*:\s*\[[^\]]*\][^{}]*\}', response_clean)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                path = data.get('path', [])
                
                # Validate that path only contains valid directions
                valid_directions = {'up', 'down', 'left', 'right'}
                path = [d.lower() for d in path if d.lower() in valid_directions]
                
                return path
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parse error: {e}")
                print(f"  JSON string: {json_str}")
                return []
        else:
            # Fallback: try to parse entire response as JSON
            try:
                data = json.loads(response_clean)
                path = data.get('path', [])
                valid_directions = {'up', 'down', 'left', 'right'}
                path = [d.lower() for d in path if d.lower() in valid_directions]
                return path
            except:
                print(f"  ❌ Could not find JSON in response")
                return []
    
    def _map_visual_to_env(self, visual_action):
        """Map visual direction to environment action
        
        Args:
            visual_action: up/down/left/right (what LLM sees)
        
        Returns:
            str: forward/backward/left/right (environment action)
        """
        mapping = {
            'up': 'forward',      # Visual up = move forward (increase X)
            'down': 'backward',   # Visual down = move backward (decrease X)
            'left': 'right',      # Visual left = env right (FLIPPED)
            'right': 'left'       # Visual right = env left (FLIPPED)
        }
        
        return mapping.get(visual_action, 'forward')


# Test code
if __name__ == "__main__":
    print("="*70)
    print("Testing LLM Path Planner Agent")
    print("="*70)
    
    # Test 1: Grid converter
    print("\n[Test 1: Grid to Text]")
    test_grid = np.array([
        [0, 0, 0, 3, 0],  # Target at top
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [2, 0, 0, 0, 0],  # Cube at bottom
    ])
    print(grid_to_text(test_grid))
    
    # Test 2: Create agent
    print("\n[Test 2: Initialize Agent]")
    agent = LLMPathPlannerAgent(max_replans=3)
    
    # Test 3: Parse path from JSON
    print("\n[Test 3: Parse Path from JSON]")
    test_responses = [
        '{"path": ["up", "right", "down"]}',
        '```json\n{"path": ["left", "left", "up"]}\n```',
        '{"path": ["UP", "RIGHT", "forward"]}',  # Mixed case + invalid
    ]
    for resp in test_responses:
        path = agent._parse_path(resp)
        print(f"  Response: {resp[:40]}...")
        print(f"  Parsed path: {path}")
    
    # Test 4: Map directions
    print("\n[Test 4: Direction Mapping]")
    for direction in ['up', 'down', 'left', 'right']:
        env_action = agent._map_visual_to_env(direction)
        print(f"  Visual '{direction}' → Env '{env_action}'")
    
    print("\n" + "="*70)
    print("✓ All tests complete!")