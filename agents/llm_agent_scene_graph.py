# FILE: agents/llm_agent_scene_graph.py
"""
LLM Agent using Text-Based Scene Graph

Instead of showing a visual grid (which LLMs struggle with),
this agent describes the environment textually:
- Current position
- Target position
- Nearby walls
- Available moves

This plays to LLM's strength (language) not weakness (spatial vision).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import re

load_dotenv()


class LLMSceneGraphAgent:
    """LLM Agent that uses textual scene description instead of visual grid"""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0, max_replans=3):
        """Initialize the scene graph agent
        
        Args:
            model: OpenAI model (gpt-4o-mini is cheapest)
            temperature: Sampling temperature (0.0 = deterministic)
            max_replans: Maximum replanning attempts
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found!")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_replans = max_replans
        
        # Statistics
        self.num_calls = 0
        self.total_tokens = 0
        
        # Path planning state
        self.planned_path = []
        self.path_index = 0
        self.replan_count = 0
        self.last_pos = None
        
        print(f"✓ LLM Scene Graph Agent initialized")
        print(f"  Model: {model}")
        print(f"  Max replans: {max_replans}")
    
    def get_action(self, env, verbose=False):
        """Get next action from planned path, or replan if needed"""
        
        cube_pos = env.get_cube_grid_pos()
        target_pos = env.get_target_grid_pos()
        
        # Already at goal?
        if cube_pos == target_pos:
            return 'forward'
        
        # Check if stuck
        action_failed = False
        if self.last_pos is not None and cube_pos == self.last_pos:
            action_failed = True
            print(f"  ⚠️  Last action FAILED - stuck at {cube_pos}")
        
        # Need to plan?
        need_to_plan = False
        
        if len(self.planned_path) == 0:
            need_to_plan = True
            print(f"  📋 No path yet - planning...")
        elif self.path_index >= len(self.planned_path):
            need_to_plan = True
            print(f"  📋 Path exhausted - replanning...")
        elif action_failed:
            need_to_plan = True
            print(f"  📋 Hit obstacle - replanning...")
        
        # Plan if needed
        if need_to_plan:
            if self.replan_count >= self.max_replans:
                print(f"  ❌ Replan limit reached ({self.max_replans})!")
                import random
                return random.choice(['forward', 'backward', 'left', 'right'])
            
            self.replan_count += 1
            print(f"  🔄 Replan #{self.replan_count}/{self.max_replans}")
            
            # Build scene graph description
            scene_description = self._build_scene_graph(env, cube_pos, target_pos)
            
            if verbose:
                print(f"\n=== SCENE GRAPH ===")
                print(scene_description)
                print(f"===================\n")
            
            # Create prompt
            prompt = self._create_prompt(scene_description, cube_pos, target_pos)
            
            # Call LLM
            response = self._call_llm(prompt)
            
            print(f"\n--- LLM Response ---")
            print(response)
            print(f"--- End Response ---\n")
            
            # Parse path
            self.planned_path = self._parse_path(response)
            self.path_index = 0
            
            if not self.planned_path:
                print(f"  ❌ Failed to parse path!")
                return 'forward'
            
            print(f"  ✓ New path: {self.planned_path} ({len(self.planned_path)} steps)")
        
        # Get next action
        if self.path_index < len(self.planned_path):
            action = self.planned_path[self.path_index]
            self.path_index += 1
            
            print(f"  Step {self.path_index}/{len(self.planned_path)}: {action}")
            
            self.last_pos = cube_pos
            return action
        else:
            return 'forward'
    
    def _build_scene_graph(self, env, cube_pos, target_pos):
        """Build textual scene description
        
        This is the KEY improvement: describe the environment with text,
        not a visual grid that LLMs struggle to parse.
        """
        grid = env.grid
        grid_size = env.grid_size
        
        cx, cy = cube_pos
        tx, ty = target_pos
        
        # Calculate direction to target
        dx = tx - cx  # Positive = need to increase X (go forward)
        dy = ty - cy  # Positive = need to increase Y (go left)
        
        # Build description
        lines = []
        
        # 1. Current state
        lines.append(f"CURRENT POSITION: ({cx}, {cy})")
        lines.append(f"TARGET POSITION: ({tx}, {ty})")
        lines.append(f"GRID SIZE: {grid_size} x {grid_size}")
        lines.append("")
        
        # 2. Direction to target
        lines.append("DIRECTION TO TARGET:")
        if dx > 0:
            lines.append(f"  - Need to go FORWARD {dx} steps (increase X from {cx} to {tx})")
        elif dx < 0:
            lines.append(f"  - Need to go BACKWARD {abs(dx)} steps (decrease X from {cx} to {tx})")
        else:
            lines.append(f"  - Same X coordinate (no forward/backward needed)")
        
        if dy > 0:
            lines.append(f"  - Need to go LEFT {dy} steps (increase Y from {cy} to {ty})")
        elif dy < 0:
            lines.append(f"  - Need to go RIGHT {abs(dy)} steps (decrease Y from {cy} to {ty})")
        else:
            lines.append(f"  - Same Y coordinate (no left/right needed)")
        lines.append("")
        
        # 3. Available moves from current position
        lines.append("AVAILABLE MOVES FROM CURRENT POSITION:")
        
        # Check forward (X+1)
        if cx + 1 >= grid_size:
            lines.append(f"  - forward: BLOCKED (edge of grid)")
        elif grid[cx + 1, cy] == 1:
            lines.append(f"  - forward: BLOCKED (wall at ({cx+1}, {cy}))")
        else:
            lines.append(f"  - forward: CLEAR (goes to ({cx+1}, {cy}))")
        
        # Check backward (X-1)
        if cx - 1 < 0:
            lines.append(f"  - backward: BLOCKED (edge of grid)")
        elif grid[cx - 1, cy] == 1:
            lines.append(f"  - backward: BLOCKED (wall at ({cx-1}, {cy}))")
        else:
            lines.append(f"  - backward: CLEAR (goes to ({cx-1}, {cy}))")
        
        # Check left (Y+1)
        if cy + 1 >= grid_size:
            lines.append(f"  - left: BLOCKED (edge of grid)")
        elif grid[cx, cy + 1] == 1:
            lines.append(f"  - left: BLOCKED (wall at ({cx}, {cy+1}))")
        else:
            lines.append(f"  - left: CLEAR (goes to ({cx}, {cy+1}))")
        
        # Check right (Y-1)
        if cy - 1 < 0:
            lines.append(f"  - right: BLOCKED (edge of grid)")
        elif grid[cx, cy - 1] == 1:
            lines.append(f"  - right: BLOCKED (wall at ({cx}, {cy-1}))")
        else:
            lines.append(f"  - right: CLEAR (goes to ({cx}, {cy-1}))")
        lines.append("")
        
        # 4. Walls along direct path to target
        lines.append("WALLS BLOCKING DIRECT PATH:")
        walls_found = []
        
        # Check walls between current and target X
        x_start, x_end = min(cx, tx), max(cx, tx)
        for x in range(x_start, x_end + 1):
            if grid[x, cy] == 1:
                walls_found.append(f"({x}, {cy})")
        
        # Check walls between current and target Y
        y_start, y_end = min(cy, ty), max(cy, ty)
        for y in range(y_start, y_end + 1):
            if grid[cx, y] == 1:
                walls_found.append(f"({cx}, {y})")
            if grid[tx, y] == 1:
                walls_found.append(f"({tx}, {y})")
        
        if walls_found:
            for wall in set(walls_found):
                lines.append(f"  - Wall at {wall}")
        else:
            lines.append(f"  - No walls blocking direct path!")
        lines.append("")
        
        # 5. Find nearby walls (within 2 steps)
        lines.append("NEARBY WALLS (within 2 steps):")
        nearby_walls = []
        for x in range(max(0, cx-2), min(grid_size, cx+3)):
            for y in range(max(0, cy-2), min(grid_size, cy+3)):
                if grid[x, y] == 1:
                    nearby_walls.append(f"({x}, {y})")
        
        if nearby_walls:
            for wall in nearby_walls[:10]:  # Limit to 10
                lines.append(f"  - Wall at {wall}")
        else:
            lines.append(f"  - No walls nearby")
        
        return '\n'.join(lines)
    
    def _create_prompt(self, scene_description, cube_pos, target_pos):
        """Create prompt using scene graph description"""
        
        prompt = f"""You are navigating a grid maze. Plan the COMPLETE path from your current position to the target.

{scene_description}

MOVEMENT RULES:
- "forward" = increase X by 1
- "backward" = decrease X by 1
- "left" = increase Y by 1
- "right" = decrease Y by 1

TASK:
1. Analyze the scene description above
2. Plan a complete path avoiding walls
3. Use ONLY these actions: forward, backward, left, right

REASONING:
First explain your plan in 2-3 sentences.

JSON:
Then output your path as JSON:
{{"path": ["forward", "left", "left", "forward"]}}

Your response:"""
        
        return prompt
    
    def _call_llm(self, prompt):
        """Call OpenAI API"""
        try:
            self.num_calls += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert path planner. You analyze textual scene descriptions and plan optimal paths through mazes. You always respond with reasoning followed by a JSON path."
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
            print(f"❌ Error calling LLM: {e}")
            return '{"path": []}'
    
    def _parse_path(self, response):
        """Extract path from JSON in response"""
        
        # Look for JSON object
        json_match = re.search(r'\{[^{}]*"path"\s*:\s*\[[^\]]*\][^{}]*\}', response)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                path = data.get('path', [])
                
                # Validate actions
                valid_actions = {'forward', 'backward', 'left', 'right'}
                path = [a.lower() for a in path if a.lower() in valid_actions]
                
                return path
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parse error: {e}")
                return []
        else:
            print(f"  ❌ Could not find JSON in response")
            return []


# Test code
if __name__ == "__main__":
    print("="*70)
    print("Testing Scene Graph LLM Agent")
    print("="*70)
    
    # Just test initialization
    agent = LLMSceneGraphAgent()
    
    print("\n✓ Agent created successfully!")
    print("\nKey differences from visual grid approach:")
    print("1. No visual grid parsing (LLMs are bad at this)")
    print("2. Explicit textual descriptions")
    print("3. Direct coordinate system (no flipping)")
    print("4. Clear obstacle information")
    print("5. Available moves listed explicitly")
