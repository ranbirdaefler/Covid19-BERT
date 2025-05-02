# src/plot_model_comparison.py

import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance():
    datasets = ['Validation', 'Test']
    
    # Your final metrics
    accuracies = {
        'BERT': [0.9860, 0.9855],
        'Static': [0.9794, 0.9797]
    }
    
    macro_f1s = {
        'BERT': [0.9593, 0.9600],
        'Static': [0.9294, 0.9304]
    }

    x = np.arange(len(datasets))  # the label locations
    width = 0.35  # width of the bars

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy plot
    axs[0].bar(x - width/2, accuracies['BERT'], width, label='BERT XGBoost')
    axs[0].bar(x + width/2, accuracies['Static'], width, label='Static XGBoost')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracy Comparison')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(datasets)
    axs[0].legend()

    # Macro F1 plot
    axs[1].bar(x - width/2, macro_f1s['BERT'], width, label='BERT XGBoost')
    axs[1].bar(x + width/2, macro_f1s['Static'], width, label='Static XGBoost')
    axs[1].set_ylabel('Macro F1 Score')
    axs[1].set_title('Macro F1 Comparison')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(datasets)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_model_performance()
