import pandas as pd
import matplotlib.pyplot as plt

def plot_success_bar(summary_csv='logs/summary.csv'):
    df = pd.read_csv(summary_csv)
    plt.figure()
    plt.bar(df['agent'], df['success_rate'])
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Agent')
    plt.savefig('assets/success_rate.png', bbox_inches='tight')

def plot_distance_curves(*csv_paths):
    plt.figure()
    for path in csv_paths:
        df = pd.read_csv(path)
        # average distance per step
        g = df.groupby('step')['dist'].mean()
        plt.plot(g.index, g.values, label=path.split('/')[-1].replace('_episodes.csv',''))
    plt.xlabel('Step')
    plt.ylabel('Mean Distance')
    plt.title('Mean Distance vs Step')
    plt.legend()
    plt.savefig('assets/distance_vs_step.png', bbox_inches='tight')

if __name__ == '__main__':
    try:
        plot_success_bar()
    except Exception as e:
        print('Bar plot error:', e)
    try:
        plot_distance_curves('logs/random_episodes.csv','logs/heuristic_episodes.csv')
    except Exception as e:
        print('Line plot error:', e)
