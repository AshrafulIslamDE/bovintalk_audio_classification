import json

import seaborn as sns
import matplotlib.pyplot as plt
import os

from config import EPOCHS


def draw_confusion_matrix(cm, model_name, class_names=None):
    """
    Plots and saves a confusion matrix heatmap.
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 8))

    # If you have specific class names (e.g., ['Cat', 'Dog']), pass them here
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')

    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    cm_path = os.path.join("plots", f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")

def draw_f1_score(model_name, f1_scores):
        os.makedirs("plots", exist_ok=True)
        os.makedirs("plot_data", exist_ok=True)

        # --- 1. Save the actual Image ---
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, EPOCHS + 1), f1_scores, marker='o', color='b', label='F1 Score')
        plt.title(f"F1 Score per Epoch - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend(loc='lower right')
        plt.xticks(range(1, EPOCHS + 1))
        plt.ylim(0, 1)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        plot_path = os.path.join("plots", f"{model_name}_f1_plot.png")
        plt.savefig(plot_path)
        plt.close()  # Close figure to free up memory
        print(f"Plot image saved to: {plot_path}")

        # --- 2. Save Raw Data for later use in IDE ---
        data_path = os.path.join("plot_data", f"{model_name}_f1_values.json")
        with open(data_path, 'w') as f: json.dump(f1_scores, f)
        print(f"Raw F1 data saved to: {data_path}")
        f1_scores.clear()