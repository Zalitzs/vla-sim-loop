import pandas as pd
from tqdm import trange

def run_episodes(env, agent, episodes=50, difficulty="medium", rounded_feedback=False, agent_name="agent"):
    rows = []
    for ep in trange(episodes, desc=f"{agent_name}"):
        obs = env.reset(difficulty=difficulty)
        step = 0
        done = False
        while not done:
            feedback = make_feedback(obs, rounded=rounded_feedback)
            action = agent.act(feedback, obs)
            next_obs, reward, done, info = env.step(action)
            rows.append({
                "episode": ep,
                "step": step,
                "agent": agent_name,
                "cube_x": obs["cube"][0],
                "cube_y": obs["cube"][1],
                "target_x": obs["target"][0],
                "target_y": obs["target"][1],
                "dist": obs["dist"],
                "action": action,
                "done": done,
                "success": info.get("success", False)
            })
            obs = next_obs
            step += 1
    df = pd.DataFrame(rows)
    return df

def make_feedback(obs, rounded=False):
    cx, cy = obs["cube"]
    tx, ty = obs["target"]
    dist = obs["dist"]
    if rounded:
        cx, cy, tx, ty, dist = [round(v,1) for v in [cx, cy, tx, ty, dist]]
    return (f"The cube is at (x={cx:.2f}, y={cy:.2f}). " 
            f"The target is at (x={tx:.2f}, y={ty:.2f}). " 
            f"The current distance is {dist:.2f} m. " 
            "Choose exactly one action from {forward, backward, left, right} and output only that single word.")
