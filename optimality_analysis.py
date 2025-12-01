"""
A* Pathfinding for Optimal Path Calculation

Computes optimal path length for each maze to measure optimality gap
"""
import numpy as np
from collections import deque
import heapq


def a_star_pathfinding(grid, start, goal):
    """
    A* algorithm to find optimal path
    
    Args:
        grid: 2D numpy array (0=free, 1=wall)
        start: (x, y) tuple
        goal: (x, y) tuple
    
    Returns:
        path: List of (x,y) positions from start to goal
        length: Number of steps in optimal path
    """
    
    def heuristic(pos):
        """Manhattan distance to goal"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Priority queue: (f_score, g_score, position, path)
    open_set = [(heuristic(start), 0, start, [start])]
    visited = set()
    
    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check if reached goal
        if current == goal:
            return path, len(path) - 1  # -1 because we count steps, not positions
        
        # Explore neighbors
        x, y = current
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)
            
            # Check bounds and walls
            if (0 <= nx < grid.shape[0] and 
                0 <= ny < grid.shape[1] and 
                grid[nx, ny] == 0 and 
                neighbor not in visited):
                
                new_g_score = g_score + 1
                new_f_score = new_g_score + heuristic(neighbor)
                new_path = path + [neighbor]
                
                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))
    
    # No path found
    return None, float('inf')


def calculate_optimality_metrics(results, test_mazes):
    """
    Calculate optimality gap for agent performance
    
    IMPORTANT: Only calculates on SUCCESSFUL episodes
    Failures (timeouts) are excluded as they didn't reach the goal
    
    Args:
        results: Dict with 'steps' list and 'success_flags' list from test_agent()
        test_mazes: List of maze templates
    
    Returns:
        metrics: Dict with optimality statistics
    """
    optimal_lengths = []
    agent_lengths = []
    optimality_gaps = []
    optimality_ratios = []
    
    for i, maze in enumerate(test_mazes):
        # Only calculate optimality for successful episodes
        if not results['success_flags'][i]:
            continue  # Skip failed episodes
        
        # Get A* optimal path
        _, optimal_len = a_star_pathfinding(
            maze['grid'],
            maze['start_pos'],
            maze['target_pos']
        )
        
        # Get agent's actual path length (for successful episode only)
        agent_len = results['steps'][i]
        
        # Calculate metrics
        if optimal_len != float('inf'):
            optimal_lengths.append(optimal_len)
            agent_lengths.append(agent_len)
            gap = agent_len - optimal_len
            ratio = agent_len / optimal_len if optimal_len > 0 else float('inf')
            
            optimality_gaps.append(gap)
            optimality_ratios.append(ratio)
    
    return {
        'avg_optimal_length': np.mean(optimal_lengths) if optimal_lengths else 0,
        'avg_agent_length': np.mean(agent_lengths) if agent_lengths else 0,
        'avg_optimality_gap': np.mean(optimality_gaps) if optimality_gaps else 0,
        'avg_optimality_ratio': np.mean(optimality_ratios) if optimality_ratios else 0,
        'num_analyzed': len(optimal_lengths),  # How many successful episodes
        'optimal_lengths': optimal_lengths,
        'optimality_gaps': optimality_gaps,
        'optimality_ratios': optimality_ratios
    }


def print_optimality_comparison(baseline_results, llm_results, test_mazes):
    """Print optimality metrics comparison (successful episodes only)"""
    
    print("\n" + "="*60)
    print("OPTIMALITY ANALYSIS (vs A* Optimal Path)")
    print("NOTE: Calculated on SUCCESSFUL episodes only")
    print("="*60)
    
    baseline_opt = calculate_optimality_metrics(baseline_results, test_mazes)
    llm_opt = calculate_optimality_metrics(llm_results, test_mazes)
    
    print(f"\nOptimal Path (A*):")
    print(f"  Average length: {baseline_opt['avg_optimal_length']:.1f} steps")
    
    print(f"\nBaseline Agent ({baseline_opt['num_analyzed']} successful episodes):")
    print(f"  Average length: {baseline_opt['avg_agent_length']:.1f} steps")
    print(f"  Optimality gap: +{baseline_opt['avg_optimality_gap']:.1f} steps")
    print(f"  Optimality ratio: {baseline_opt['avg_optimality_ratio']:.2f}x optimal")
    
    print(f"\nWith LLM Guidance ({llm_opt['num_analyzed']} successful episodes):")
    print(f"  Average length: {llm_opt['avg_agent_length']:.1f} steps")
    print(f"  Optimality gap: +{llm_opt['avg_optimality_gap']:.1f} steps")
    print(f"  Optimality ratio: {llm_opt['avg_optimality_ratio']:.2f}x optimal")
    
    # Calculate improvement
    gap_reduction = baseline_opt['avg_optimality_gap'] - llm_opt['avg_optimality_gap']
    ratio_improvement = baseline_opt['avg_optimality_ratio'] - llm_opt['avg_optimality_ratio']
    percent_improvement = (ratio_improvement / baseline_opt['avg_optimality_ratio'] * 100) if baseline_opt['avg_optimality_ratio'] > 0 else 0
    
    print(f"\nLLM Impact on Optimality:")
    print(f"  Gap reduction: {gap_reduction:.1f} steps closer to optimal")
    print(f"  Ratio improvement: {ratio_improvement:.2f}x ({percent_improvement:.1f}% closer to optimal)")
    
    # Additional insight
    if gap_reduction > 0:
        print(f"\n✓ LLM guidance produces more optimal paths on successful episodes")
    else:
        print(f"\n→ Both agents achieve similar path optimality on successful episodes")
    
    print("="*60)
    
    return baseline_opt, llm_opt


if __name__ == "__main__":
    # Test A* on simple maze
    grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ])
    
    path, length = a_star_pathfinding(grid, (0, 0), (4, 4))
    print(f"Optimal path: {path}")
    print(f"Optimal length: {length} steps")