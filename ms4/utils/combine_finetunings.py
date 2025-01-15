import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_runs(file_path):
    # Load the data
    data = pd.read_csv(file_path, sep=';', header=0)

    # Ensure numeric conversion
    data['epoch'] = data['step'].astype(int)
    data['value'] = data['value'].astype(float)

    # Adjust the epochs for a continuous plot
    cumulative_offset = 0
    data['adjusted_epoch'] = 0

    for run in data['run'].unique():
        run_data = data[data['run'] == run]
        data.loc[run_data.index, 'adjusted_epoch'] = run_data['epoch'] + cumulative_offset
        cumulative_offset += run_data['epoch'].max() + 1  # Add offset for the next run

    # Plot the combined run
    plt.figure(figsize=(12, 6))
    plt.plot(data['adjusted_epoch'], data['value'], label='Combined Run')

    # Highlight the top value
    max_value = data['value'].max()
    max_value_epoch = data[data['value'] == max_value]['adjusted_epoch'].values[0]
    plt.scatter(max_value_epoch, max_value, color='black')  # Highlight the point
    plt.text(max_value_epoch, max_value, f'{max_value:.4f}', fontsize=12, ha='right', color='black')  # Add text box

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Combined Finetuning Run for Triplet Loss')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = 'output_csvs/epoch_accuracy.csv'
    plot_combined_runs(file_path)