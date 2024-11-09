
import pandas as pd
import matplotlib.pyplot as plt

NAMES = ['epoch_loss.csv']

for name in NAMES:
    print(f'Showing {name}')
    df = pd.read_csv("output_csvs/" + name, delimiter=';')
    for run_name, group in df.groupby('run'):
        plt.plot(group['step'], group['value'], label=run_name)

    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()