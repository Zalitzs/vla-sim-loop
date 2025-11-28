import numpy as np

def get_maze_template(name, grid_size=20):
    """Return a predefined maze template
    
    Returns:
        dict with 'grid', 'start_pos', 'target_pos'
    """
    
    mazes = {
        'corridor': {
            'description': 'Long corridor with walls - tests forward planning',
            'grid_size': 10,
            'walls': [(i, 3) for i in range(2, 8)] + [(i, 5) for i in range(2, 8)],
            'start': (1, 4),
            'target': (8, 4)
        },
        
        'u_shape': {
            'description': 'U-shaped path - tests ability to go around',
            'grid_size': 10,
            'walls': [(2, i) for i in range(2, 8)] + [(i, 7) for i in range(3, 10)]+ [(i, 2) for i in range(3, 10)],
            'start': (6, 1),
            'target': (6, 8)
        },
        
        'narrow_gap': {
            'description': 'Tests if agent can find small opening',
            'grid_size': 10,
            'walls': [(4, i) for i in range(10) if i != 5],  # Wall with one gap
            'start': (2, 2),
            'target': (7, 7)
        },
        
        'spiral': {
            'description': 'Spiral path - tests multi-turn navigation',
            'grid_size': 10,
            'walls': ([(2, i) for i in range(2, 8)] + 
                     [(i, 7) for i in range(2, 8)] +
                     [(8, i) for i in range(2, 8)] +
                     [(i, 2) for i in range(4, 8)] +
                     [(4, i) for i in range(2, 6)] +
                     [(5, 5), (6, 5), (6, 4)]
                     ),
            'start': (1, 1),
            'target': (5, 4)
        },
        
        'maze_hard': {
            'description': 'Hard maze with one optimal path',
            'grid_size': 10,
            'walls': [(0, 0), (1, 2), (2, 9),(6, 9)] + 
            [(9, i) for i in range(5, 9)] + [(4, i) for i in range(3, 8)] + [(2, i) for i in range(8)] + 
            [(i, 5) for i in range(6, 9)] +[(i, 1) for i in range(2, 9)] + [(i, 3) for i in range(4, 9)]+ [(i, 8) for i in range(4, 9) if i != 7],
            'start': (1, 0),
            'target': (8, 6)
        },
        
        'maze_hard_2': {
            'description': 'maze harder',
            'grid_size': 10,
            'walls': [(0, 3), (0, 7), (1, 1), (1, 3), (1, 5), (1, 7), (1, 8), (2, 1), (2, 2), (2, 3), (2, 5), (2, 6), (2, 7), (2, 8), (3, 3), (4, 0), (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 9), (5, 1), (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)],
            'start': (9, 0),
            'target': (1, 6)
        },
        
        'maze_hard_3': {
            'description': 'Hard maze with one optimal path',
            'grid_size': 10,
            'walls': [(0, 0), (1, 2), (2, 9),(6, 9)] + 
            [(9, i) for i in range(5, 9)] + [(4, i) for i in range(3, 8)] + [(2, i) for i in range(8)] + 
            [(i, 5) for i in range(6, 9)] +[(i, 1) for i in range(2, 9)] + [(i, 3) for i in range(4, 9)]+ [(i, 8) for i in range(4, 9) if i != 7],
            'start': (1, 0),
            'target': (8, 6)
        },
        
        'maze_simple': {
            'description': 'Simple maze with multiple optimal paths',
            'grid_size': 10,
            'walls': [(i, 2) for i in range(0, 3)] + 
            [(i, 3) for i in range(0, 3)] + [(2,i) for i in range(6,9)] + 
            [(3, i) for i in range(6, 9)] + [(i,6) for i in range(6,9)] + 
            [(i,7) for i in range(6,9)] + [(6,i) for i in range(0,3)] + 
            [(7,i) for i in range(0,3)],
            'start': (0, 0),
            'target': (9, 9)
        }
    }
    
    if name not in mazes:
        raise ValueError(f"Unknown maze: {name}. Available: {list(mazes.keys())}")
    
    template = mazes[name]
    
    # Create grid
    grid = np.zeros((template['grid_size'], template['grid_size']), dtype=int)
    for x, y in template['walls']:
        if 0 <= x < template['grid_size'] and 0 <= y < template['grid_size']:
            grid[x, y] = 1
    
    return {
        'grid': grid,
        'start_pos': template['start'],
        'target_pos': template['target'],
        'description': template['description'],
        'grid_size': template['grid_size']
    }