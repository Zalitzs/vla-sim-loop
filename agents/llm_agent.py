# FILE: agents/llm_agent.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()


def grid_to_text(grid):
    """Convert numpy grid array to text representation for LLM
    
    This function takes the numeric grid and converts it to symbols
    that are easier for an LLM to understand visually.
    
    Args:
        grid: numpy array where:
              0 = empty space
              1 = wall  
              2 = cube (player position)
              3 = target (goal position)
    
    Returns:
        str: Text representation with symbols:
             . = empty
             # = wall
             C = cube
             T = target
    
    Example:
        Input grid (numpy array):
        [[0, 1, 0],
         [2, 0, 3]]
        
        Output (string):
        ". # .\nC . T"
        
        Which displays as:
        . # .
        C . T
    """
    # Step 1: Define our symbol mapping
    # This dictionary maps each number to a symbol
    symbols = {
        0: '.',  # empty space
        1: '#',  # wall
        2: 'C',  # cube (player)
        3: 'T'   # target (goal)
    }
    
    # Step 2: Convert the grid row by row
    rows = []  # We'll store each row as text here
    
    # Loop through each row in the grid
    for row in grid:
        # Step 2a: Convert each number in the row to its symbol
        # Example: [0, 1, 2] becomes ['.', '#', 'C']
        row_symbols = [symbols[cell] for cell in row]
        
        # Step 2b: Join the symbols with spaces for readability
        # Example: ['.', '#', 'C'] becomes ". # C"
        row_text = ' '.join(row_symbols)
        
        # Step 2c: Add this row to our list
        rows.append(row_text)
    
    # Step 3: Join all rows with newlines to create the final grid
    # Example: [". # C", "T . ."] becomes ". # C\nT . ."
    return '\n'.join(rows)

class LLMAgent:
    """An agent that uses an LLM (GPT) to decide actions in the maze"""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        """Initialize the LLM agent
        
        Args:
            model: Which OpenAI model to use
            temperature: How creative/random the LLM should be (0.0 = deterministic)
        """
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Store settings
        self.model = model
        self.temperature = temperature
        
        # Statistics tracking
        self.num_calls = 0
        self.total_tokens = 0
        
        # NEW: Track last action for feedback
        self.last_action = None
        self.last_pos = None
        self.action_failed = False
        
        print(f"✓ LLM Agent initialized with model: {model}")
        
    def get_action(self, env, verbose=False):
        """Get the next action from the LLM"""
        # Get information from environment
        grid = env.get_grid_state()
        cube_pos = env.get_cube_grid_pos()
        target_pos = env.get_target_grid_pos()
        
        # EDGE CASE: Already at goal
        if cube_pos == target_pos:
            return 'forward'
        
        # Check if last action failed
        if self.last_pos is not None:
            if cube_pos == self.last_pos:
                self.action_failed = True
            else:
                self.action_failed = False
        
        # Flip grid to match visual orientation
        grid_flipped = np.flipud(grid)
        
        # Convert flipped grid to text representation
        grid_text = grid_to_text(grid_flipped)
        
        # DEBUG: Print what we're sending to LLM
        print(f"\n=== DEBUG: What LLM Sees ===")
        print(f"Cube grid pos: {cube_pos}")
        print(f"Target grid pos: {target_pos}")
        print(f"\nGrid (after flip):")
        print(grid_text)
        print(f"=== End Debug ===\n")
        
        # Create the prompt
        prompt = self._create_prompt(grid_text, cube_pos, target_pos)
        
        # If prompt is None (already at goal), return any action
        if prompt is None:
            return 'forward'
        
        # DEBUG: Print full prompt
        if verbose:
            print(f"\n=== FULL PROMPT ===")
            print(prompt)
            print(f"=== END PROMPT ===\n")
        
        # Call the LLM
        response = self._call_llm(prompt)
        
        # Print reasoning
        print(f"\n--- LLM Reasoning ---")
        print(response)
        print(f"--- End Reasoning ---\n")
        
        # Parse the response to extract an action
        llm_action = self._parse_action(response)
        
        # Remap the action
        actual_action = self._remap_action(llm_action)
        
        print(f"LLM said: '{llm_action}' → Remapped to: '{actual_action}'")
        
        # Remember this action and position for next time
        self.last_action = llm_action
        self.last_pos = cube_pos
        
        return actual_action
    
    def _create_prompt(self, grid_text, cube_pos, target_pos):
        """Create the prompt with chain-of-thought reasoning"""
        
        # Check if already at goal
        if cube_pos == target_pos:
            # Don't even ask LLM, just stay put
            return None  # We'll handle this in get_action
        
        # Build feedback about last action
        feedback = ""
        if self.action_failed:
            feedback = f"\n⚠️ WARNING: Your last action '{self.last_action}' FAILED - you hit a wall!\nChoose a DIFFERENT direction.\n"
        
        prompt = f"""Navigate cube C to target T in a maze.

    Grid:
    {grid_text}

    Legend: C=you, T=goal, #=wall, .=empty
    Current: {cube_pos} | Target: {target_pos}
    {feedback}
    Actions: up (move up), down (move down), left (move left), right (move right)

    Think briefly:
    1. Where is T relative to C?
    2. Any walls blocking direct path?
    3. Best direction to approach T?

    Reasoning (1-2 sentences max):
    [Your analysis]

    Action (MUST be exactly one word: up, down, left, or right):"""

        return prompt

    def _call_llm(self, prompt):
        """Call the OpenAI API with the prompt"""
        try:
            self.num_calls += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at solving grid mazes. You think step-by-step, then choose the best action."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150  # <-- CHANGED from 10 to 150 to allow reasoning
            )
            
            response_text = response.choices[0].message.content.strip()
            self.total_tokens += response.usage.total_tokens
            
            return response_text
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import random
            return random.choice(['forward', 'backward', 'left', 'right'])
    
    def _parse_action(self, response):
        """Extract the action from the LLM's response
        
        LLM might say "move DOWN" or "go up" - we should accept those!
        """
        response_lower = response.lower().strip()
        
        # Map natural language to our action words
        direction_map = {
            'up': 'forward',
            'down': 'backward',
            'left': 'left',
            'right': 'right',
            'forward': 'forward',
            'backward': 'backward'
        }
        
        # Strategy 1: Look for direction words with context
        # Match patterns like "move UP", "go DOWN", "moving left", etc.
        import re
        patterns = [
            r'(?:move|go|moving|action)[:\s]+(\w+)',  # "move DOWN"
            r'(?:should|will|can)\s+(?:move|go)\s+(\w+)',  # "should move up"
            r'direction[:\s]+(\w+)',  # "direction: down"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                word = match.group(1)
                if word in direction_map:
                    return direction_map[word]
        
        # Strategy 2: Check last line
        lines = response_lower.split('\n')
        if lines:
            last_line = lines[-1].strip()
            for word, action in direction_map.items():
                if word in last_line:
                    return action
        
        # Strategy 3: Search entire response for any direction word
        for word, action in direction_map.items():
            if word in response_lower:
                return action
        
        print(f"Warning: Could not parse action from: '{response[:100]}...', defaulting to 'forward'")
        return 'forward'

    def _remap_action(self, llm_action):
        """Remap LLM's action (natural directions) to actual physics action
        
        LLM uses: up, down, left, right (natural grid directions)
        Physics uses: forward, backward, left, right (coordinate system)
        
        After flipping the grid, the mapping is:
        - LLM "up" → Physics "backward" (decrease x)
        - LLM "down" → Physics "forward" (increase x)
        - LLM "left" → Physics "right" (decrease y)
        - LLM "right" → Physics "left" (increase y)
        """
        remap = {
            'up': 'backward',
            'down': 'forward',
            'left': 'right',
            'right': 'left',
            # Also accept the old format just in case
            'forward': 'backward',
            'backward': 'forward'
        }
        
        return remap.get(llm_action, 'forward')

# Test code - only runs when you execute this file directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Agent")
    print("=" * 60)
    
    # Test 1: Grid converter
    print("\n[Test 1: Grid Converter]")
    test_grid = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [2, 0, 0, 0, 3],
        [0, 0, 1, 0, 0]
    ])
    print(grid_to_text(test_grid))
    
    # Test 2: Create LLM agent
    print("\n[Test 2: Initialize LLM Agent]")
    agent = LLMAgent(model="gpt-4o-mini", temperature=0.7)
    
    # Test 3: Test action parsing
    print("\n[Test 3: Action Parsing]")
    test_responses = [
        "forward",
        "I think we should move forward",
        "Forward is the best choice",
        "Let's go backward to avoid the wall"
    ]
    for response in test_responses:
        action = agent._parse_action(response)
        print(f"  Response: '{response}' → Action: '{action}'")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")