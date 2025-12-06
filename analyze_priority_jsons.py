"""
Analyze Priority Mazes from JSON Data

Calculates statistics, patterns, and insights from all analyzed mazes
"""
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def load_all_jsons(directory='priority_analysis'):
    """Load all JSON files from analysis"""
    json_files = glob.glob(f'{directory}/*.json')
    
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            maze_data = json.load(f)
            
        # Extract metadata from filename
        filename = Path(json_file).stem
        parts = filename.split('_')
        maze_num = parts[0].replace('Maze', '')
        boost_type = parts[1]
        
        maze_data['filename'] = filename
        maze_data['maze_num'] = int(maze_num)
        maze_data['boost_type'] = boost_type
        
        data.append(maze_data)
    
    return data

def analyze_loops(queries):
    """Detect position loops in queries"""
    positions = [tuple(q['position']) for q in queries]
    
    # Count position repetitions
    from collections import Counter
    position_counts = Counter(positions)
    
    # Find loops (positions visited 3+ times)
    loops = {pos: count for pos, count in position_counts.items() if count >= 3}
    
    return {
        'unique_positions': len(position_counts),
        'total_positions': len(positions),
        'revisit_rate': 1 - (len(position_counts) / len(positions)),
        'loop_positions': loops,
        'max_revisits': max(position_counts.values()) if position_counts else 0
    }

def analyze_follow_patterns(queries):
    """Analyze when agent follows vs ignores LLM"""
    followed = [q for q in queries if q['followed']]
    ignored = [q for q in queries if not q['followed']]
    
    patterns = {
        'follow_count': len(followed),
        'ignore_count': len(ignored),
        'follow_rate': len(followed) / len(queries) if queries else 0
    }
    
    # When does agent follow?
    if followed:
        patterns['follow_avg_entropy'] = np.mean([q['entropy'] for q in followed])
        patterns['follow_max_prob'] = np.mean([max(q['probabilities'].values()) for q in followed])
    
    # When does agent ignore?
    if ignored:
        patterns['ignore_avg_entropy'] = np.mean([q['entropy'] for q in ignored])
        patterns['ignore_max_prob'] = np.mean([max(q['probabilities'].values()) for q in ignored])
    
    return patterns

def analyze_entropy_patterns(queries):
    """Analyze entropy distribution"""
    entropies = [q['entropy'] for q in queries]
    
    return {
        'avg_entropy': np.mean(entropies),
        'median_entropy': np.median(entropies),
        'min_entropy': np.min(entropies),
        'max_entropy': np.max(entropies),
        'std_entropy': np.std(entropies),
        'below_threshold': sum(1 for e in entropies if e < 1.0),
        'high_uncertainty': sum(1 for e in entropies if e > 1.3)
    }

def create_summary_report(all_data):
    """Create comprehensive summary"""
    
    print("="*80)
    print("COMPREHENSIVE PRIORITY MAZE ANALYSIS")
    print("="*80)
    
    # Group by priority
    priority1_mult = [d for d in all_data if d['maze_num'] in [19, 30, 41] and d['boost_type'] == 'mult']
    priority1_add = [d for d in all_data if d['maze_num'] in [19, 30, 41] and d['boost_type'] == 'add']
    priority2_mult = [d for d in all_data if d['maze_num'] in [1, 35, 60, 74] and d['boost_type'] == 'mult']
    priority3_add = [d for d in all_data if d['maze_num'] in [96, 101, 109, 122] and d['boost_type'] == 'add']
    
    print(f"\nDataset Summary:")
    print(f"  Priority 1 (Both work): {len(priority1_mult)} mult, {len(priority1_add)} add")
    print(f"  Priority 2 (Mult only): {len(priority2_mult)} mult")
    print(f"  Priority 3 (Add only):  {len(priority3_add)} add")
    
    # Detailed analysis by priority
    groups = {
        'Priority 1: Multiplicative': priority1_mult,
        'Priority 1: Additive': priority1_add,
        'Priority 2: Multiplicative only': priority2_mult,
        'Priority 3: Additive only': priority3_add
    }
    
    for group_name, group_data in groups.items():
        if not group_data:
            continue
            
        print(f"\n{'='*80}")
        print(group_name)
        print(f"{'='*80}")
        
        for maze in group_data:
            print(f"\nMaze #{maze['maze_num']} ({maze['boost_type']}):")
            print(f"  Wall Density: {maze['maze_info']['wall_density']:.2f}")
            print(f"  Total Queries: {maze['maze_info']['total_queries']}")
            
            # Follow patterns
            follow = analyze_follow_patterns(maze['queries'])
            print(f"  Follow Rate: {follow['follow_rate']*100:.1f}% ({follow['follow_count']}/{maze['maze_info']['total_queries']})")
            
            # Entropy
            entropy = analyze_entropy_patterns(maze['queries'])
            print(f"  Entropy: avg={entropy['avg_entropy']:.3f}, range=[{entropy['min_entropy']:.3f}, {entropy['max_entropy']:.3f}]")
            print(f"  Below threshold (<1.0): {entropy['below_threshold']} queries")
            print(f"  High uncertainty (>1.3): {entropy['high_uncertainty']} queries")
            
            # Loops
            loops = analyze_loops(maze['queries'])
            print(f"  Position Revisits: {loops['revisit_rate']*100:.1f}% (max: {loops['max_revisits']}x)")
            if loops['loop_positions']:
                print(f"  Loop Detected: {len(loops['loop_positions'])} positions visited 3+ times")
    
    # Comparative statistics
    print(f"\n{'='*80}")
    print("COMPARATIVE STATISTICS")
    print(f"{'='*80}")
    
    if priority1_mult and priority1_add:
        print("\nPriority 1: Multiplicative vs Additive (same mazes)")
        print("-" * 80)
        
        mult_follow = np.mean([analyze_follow_patterns(m['queries'])['follow_rate'] for m in priority1_mult])
        add_follow = np.mean([analyze_follow_patterns(m['queries'])['follow_rate'] for m in priority1_add])
        
        mult_queries = np.mean([m['maze_info']['total_queries'] for m in priority1_mult])
        add_queries = np.mean([m['maze_info']['total_queries'] for m in priority1_add])
        
        mult_entropy = np.mean([analyze_entropy_patterns(m['queries'])['avg_entropy'] for m in priority1_mult])
        add_entropy = np.mean([analyze_entropy_patterns(m['queries'])['avg_entropy'] for m in priority1_add])
        
        print(f"  Average Follow Rate:")
        print(f"    Multiplicative: {mult_follow*100:.1f}%")
        print(f"    Additive:       {add_follow*100:.1f}%")
        
        print(f"  Average Queries:")
        print(f"    Multiplicative: {mult_queries:.1f}")
        print(f"    Additive:       {add_queries:.1f}")
        
        print(f"  Average Entropy:")
        print(f"    Multiplicative: {mult_entropy:.3f}")
        print(f"    Additive:       {add_entropy:.3f}")
    
    # Export to CSV
    csv_data = []
    for maze in all_data:
        follow = analyze_follow_patterns(maze['queries'])
        entropy = analyze_entropy_patterns(maze['queries'])
        loops = analyze_loops(maze['queries'])
        
        csv_data.append({
            'maze_num': maze['maze_num'],
            'boost_type': maze['boost_type'],
            'wall_density': maze['maze_info']['wall_density'],
            'total_queries': maze['maze_info']['total_queries'],
            'follow_rate': follow['follow_rate'],
            'avg_entropy': entropy['avg_entropy'],
            'max_entropy': entropy['max_entropy'],
            'below_threshold': entropy['below_threshold'],
            'revisit_rate': loops['revisit_rate'],
            'max_revisits': loops['max_revisits']
        })
    
    df = pd.DataFrame(csv_data)
    df = df.sort_values(['maze_num', 'boost_type'])
    df.to_csv('priority_analysis_summary.csv', index=False)
    
    print(f"\n{'='*80}")
    print("[OK] Exported summary to: priority_analysis_summary.csv")
    print(f"{'='*80}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze priority maze JSONs')
    parser.add_argument('--dir', type=str, default='priority_analysis',
                       help='Directory containing JSON files')
    
    args = parser.parse_args()
    
    print("Loading JSON files...")
    all_data = load_all_jsons(args.dir)
    
    if not all_data:
        print(f"[X] No JSON files found in {args.dir}/")
        exit(1)
    
    print(f"[OK] Loaded {len(all_data)} maze analyses\n")
    
    df = create_summary_report(all_data)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("\nSee priority_analysis_summary.csv for full data")