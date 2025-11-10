import pandas as pd
import numpy as np

def summarize(df: pd.DataFrame):
    # per-episode success is last row (max step) where done==True
    last_steps = df.sort_values(['episode','step']).groupby('episode').tail(1)
    success_rate = last_steps['success'].mean()
    steps_success = last_steps.loc[last_steps['success'], 'step'].mean()
    final_dist = last_steps['dist'].mean()
    return {
        'success_rate': float(success_rate),
        'mean_steps_success': float(steps_success) if not np.isnan(steps_success) else None,
        'mean_final_dist': float(final_dist)
    }
