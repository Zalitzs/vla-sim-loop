"""
Batch Analysis Script - Run All Priority Mazes

Analyzes key mazes for publication:
- Priority 1: Success stories (both configs work)
- Priority 2: Multiplicative-only successes
- Priority 3: Additive-only successes
"""
import subprocess
import os

# Define mazes to analyze
priority_mazes = {
    "Priority 1: Both configs solved (success stories)": [
        # Maze #19 (idx 18)
        {"idx": 18, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze19_mult"},
        {"idx": 18, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze19_add"},
        # Maze #30 (idx 29)
        {"idx": 29, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze30_mult"},
        {"idx": 29, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze30_add"},
        # Maze #41 (idx 40)
        {"idx": 40, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze41_mult"},
        {"idx": 40, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze41_add"},
    ],
    
    "Priority 2: Multiplicative-only successes": [
        # Maze #1 (idx 0)
        {"idx": 0, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze1_mult"},
        # Maze #35 (idx 34)
        {"idx": 34, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze35_mult"},
        # Maze #60 (idx 59)
        {"idx": 59, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze60_mult"},
        # Maze #74 (idx 73) - Hardest!
        {"idx": 73, "threshold": 0.8, "boost": 2.0, "type": "multiplicative", "name": "Maze74_mult"},
    ],
    
    "Priority 3: Additive-only successes": [
        # Maze #96 (idx 95)
        {"idx": 95, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze96_add"},
        # Maze #101 (idx 100)
        {"idx": 100, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze101_add"},
        # Maze #109 (idx 108)
        {"idx": 108, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze109_add"},
        # Maze #122 (idx 121)
        {"idx": 121, "threshold": 1.0, "boost": 0.3, "type": "additive", "name": "Maze122_add"},
    ],
}

# Create output directory
output_dir = "priority_analysis"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("BATCH ANALYSIS - PRIORITY MAZES")
print("="*70)
print(f"Output directory: {output_dir}/")
print(f"Total analyses: {sum(len(mazes) for mazes in priority_mazes.values())}")
print("="*70)

# Run all analyses
total_count = 0
for priority_name, mazes in priority_mazes.items():
    print(f"\n{'='*70}")
    print(priority_name)
    print(f"{'='*70}")
    
    for maze_config in mazes:
        total_count += 1
        idx = maze_config["idx"]
        threshold = maze_config["threshold"]
        boost = maze_config["boost"]
        boost_type = maze_config["type"]
        output_name = f"{output_dir}/{maze_config['name']}_table.png"
        
        print(f"\n[{total_count}] Analyzing Maze #{idx+1} ({boost_type}, boost={boost})...")
        
        # Build command
        cmd = [
            "python", "analyze_action_probs_table.py",
            "--maze-idx", str(idx),
            "--threshold", str(threshold),
            "--boost", str(boost),
            "--boost-type", boost_type,
            "--output", output_name
        ]
        
        # Run analysis
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"  [OK] Success! Saved to {output_name}")
            else:
                print(f"  [X] Failed!")
                print(f"  Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  [!] Timeout (>2 min)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

print("\n" + "="*70)
print("BATCH ANALYSIS COMPLETE!")
print("="*70)
print(f"Check {output_dir}/ for all visualizations")
print(f"Total files created: {total_count}")
print("="*70)

# Print summary
print("\n" + "="*70)
print("SUMMARY OF ANALYSES")
print("="*70)
for priority_name, mazes in priority_mazes.items():
    print(f"\n{priority_name}:")
    for maze_config in mazes:
        print(f"  - {maze_config['name']}_table.png")