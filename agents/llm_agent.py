# FILE: agents/llm_agent.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()


def grid_to_text(grid):
    """Convert numpy grid array to text representation for LLM
    
    This creates a visual representation where:
    - Top of display = Higher X values (North/Forward)
    - Bottom of display = Lower X values (South/Backward)
    - Left of display = Lower Y values (West)
    - Right of display = Higher Y values (East)
    
    The LLM will use natural directions (up/down/left/right) based on
    what it SEES in this visual representation.
    
    Args:
        grid: numpy array where:
              0 = empty space
              1 = wall  
              2 = cube (you are here)
              3 = target (goal)
    
    Returns:
        str: Visual text grid
    """
    # Step 1: Define symbols
    symbols = {
        0: '.',  # empty space
        1: '#',  # wall
        2: 'C',  # cube (you)
        3: 'T'   # target (goal)
    }
    
    # Step 2: Flip vertically so high X is at top
    # This makes the visual match intuition:
    # - "up" in the visual = forward in environment (increasing X)
    # - "down" in the visual = backward in environment (decreasing X)
    grid_flipped = np.flipud(grid)
    
    # Step 3: Convert to text row by row
    rows = []
    for row in grid_flipped:
        row_symbols = [symbols[cell] for cell in row]
        row_text = ' '.join(row_symbols)
        rows.append(row_text)
    
    return '\n'.join(rows)


class LLMAgent:
    """An agent that uses an LLM (GPT) to navigate by looking at the grid visually"""
    
    def __init__(self, model="gpt-5-mini", temperature=0.0):
        """Initialize the LLM agent
        
        Args:
            model: Which OpenAI model to use
            temperature: How creative/random (0.0 = deterministic)
        """
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables!")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        
        # Statistics tracking
        self.num_calls = 0
        self.total_tokens = 0
        
        # Track last action for feedback
        self.last_action = None
        self.last_pos = None
        self.action_failed = False
        
        print(f"✓ LLM Agent initialized with model: {model}")
        
    def get_action(self, env, verbose=False):
        """Get the next action from the LLM based on visual grid"""
        # Get current state
        grid = env.get_grid_state()
        cube_pos = env.get_cube_grid_pos()
        target_pos = env.get_target_grid_pos()
        
        # Check if already at goal
        if cube_pos == target_pos:
            return 'forward'
        
        # Check if last action failed (hit wall)
        if self.last_pos is not None:
            if cube_pos == self.last_pos:
                self.action_failed = True
            else:
                self.action_failed = False
        
        # Convert grid to visual text
        grid_text = grid_to_text(grid)
        
        '''# DEBUG: Show what we're sending
        print(f"\n=== Visual Grid for LLM ===")
        print(grid_text)
        print(f"\nCube position: {cube_pos}")
        print(f"Target position: {target_pos}")
        print(f"=== End Visual ===\n")'''
        
        # Create prompt
        prompt = self._create_prompt(grid_text, cube_pos, target_pos)
        
        if prompt is None:
            return 'forward'
        
        if verbose:
            print(f"\n=== FULL PROMPT ===")
            print(prompt)
            print(f"=== END PROMPT ===\n")
        
        # Call LLM
        response = self._call_llm(prompt)
        
        '''# Show reasoning
        print(f"\n--- LLM Reasoning ---")
        print(response)
        print(f"--- End Reasoning ---\n")'''
        
        # Parse action from response
        visual_action = self._parse_action(response)
        
        # Map visual action to environment action
        actual_action = self._map_visual_to_env(visual_action)
        
        '''print(f"LLM chose: '{visual_action}' (visual) → '{actual_action}' (environment)")'''
        
        # Remember for next time
        self.last_action = visual_action
        self.last_pos = cube_pos
        
        return actual_action
    
    def _create_prompt(self, grid_text, cube_pos, target_pos):
        """Create a simple, visual navigation prompt"""
        
        if cube_pos == target_pos:
            return None
        
        # Build feedback if last action failed
        feedback = ""
        if self.action_failed:
            feedback = f"\n⚠️ Your last move '{self.last_action}' FAILED - you hit a wall!\nTry a different direction.\n"
        
        prompt = f"""You are navigating a maze. Look at the grid below and choose the best direction.

GRID (look at this carefully):
{grid_text}

LEGEND:
- C = You (current position)
- T = Goal (target to reach)
- # = Wall (cannot pass through)
- . = Empty space (can move through)

DIRECTIONS YOU CAN MOVE:
- "up" = move toward the top of the grid
- "down" = move toward the bottom of the grid  
- "left" = move toward the left side of the grid
- "right" = move toward the right side of the grid

YOUR TASK:
Navigate from C to T by choosing one direction at a time.
{feedback}

THINK STEP BY STEP:
1. Where is T relative to C in the visual grid?
2. Are there any walls blocking the direct path?
3. What is the best single direction to get closer to T?

Your response should be:
First, explain your thinking in 1-2 sentences.
Then, on a new line, write ONLY ONE of these words: up, down, left, or right

Your response:"""

        return prompt

    def _call_llm(self, prompt):
        """Call the OpenAI API"""
        try:
            self.num_calls += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at navigating grid mazes visually. You look at the grid, identify where you are (C) and where the goal is (T), then choose the best direction."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            self.total_tokens += response.usage.total_tokens
            
            return response_text
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            import random
            return random.choice(['up', 'down', 'left', 'right'])
    
    def _parse_action(self, response):
        """Extract direction from LLM's response
        
        The LLM should respond with: up, down, left, or right
        based on what it sees in the visual grid.
        """
        response_lower = response.lower().strip()
        
        # Valid visual directions
        valid_directions = ['up', 'down', 'left', 'right']
        
        # Strategy 1: Check last line (where we asked it to put the answer)
        lines = response_lower.split('\n')
        if lines:
            last_line = lines[-1].strip()
            for direction in valid_directions:
                if direction in last_line:
                    return direction
        
        # Strategy 2: Look for direction words with common phrases
        import re
        patterns = [
            r'(?:move|go|choose|pick)\s+(\w+)',
            r'direction[:\s]+(\w+)',
            r'(?:i will|i choose|i pick)\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                word = match.group(1)
                if word in valid_directions:
                    return word
        
        # Strategy 3: Just find any direction word in the response
        for direction in valid_directions:
            if direction in response_lower:
                return direction
        
        # Default: move up (safest guess)
        print(f"Warning: Could not parse direction from '{response[:100]}...', defaulting to 'up'")
        return 'up'

    def _map_visual_to_env(self, visual_action):
        """Map visual direction to environment action
        
        COORDINATE SYSTEM:
        - Environment uses X, Y coordinates where:
          * X axis: forward/backward
          * Y axis: left/right
        
        - Visual grid after flip shows:
          * "up" in visual = higher rows in flipped grid = higher X in env = FORWARD
          * "down" in visual = lower rows in flipped grid = lower X in env = BACKWARD
          * "left" in visual = lower columns = lower Y in env = ... needs checking
          * "right" in visual = higher columns = higher Y in env = ... needs checking
        
        HOWEVER, based on your note about needing to flip left/right but not up/down,
        let me implement what you need:
        """
        # Based on your feedback: flip left/right, keep up/down as is
        mapping = {
            'up': 'forward',       # Keep up → forward
            'down': 'backward',    # Keep down → backward  
            'left': 'right',       # FLIP left → right
            'right': 'left'        # FLIP right → left
        }
        
        return mapping.get(visual_action, 'forward')


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fixed LLM Agent")
    print("=" * 60)
    
    # Test 1: Grid converter
    print("\n[Test 1: Grid to Visual Text]")
    test_grid = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [2, 0, 0, 0, 3],
        [0, 0, 1, 0, 0]
    ])
    print("Original grid array:")
    print(test_grid)
    print("\nVisual representation:")
    print(grid_to_text(test_grid))
    
    # Test 2: Create agent
    print("\n[Test 2: Initialize Agent]")
    agent = LLMAgent()
    
    # Test 3: Action parsing
    print("\n[Test 3: Parse Directions]")
    test_responses = [
        "I should move up to get closer",
        "down",
        "The best direction is left",
        "right seems clear"
    ]
    for resp in test_responses:
        visual = agent._parse_action(resp)
        env = agent._map_visual_to_env(visual)
        print(f"  '{resp}' → visual='{visual}' → env='{env}'")
    
    print("\n" + "=" * 60)
    print("✓ Tests complete!")
