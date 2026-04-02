import matplotlib.pyplot as plt

def weighted_average(metrics):
    """
    Aggregation function for the Flower server.
    Calculates the weighted average of accuracy across all clients.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric
    return {"accuracy": sum(accuracies) / sum(examples)}

def plot_results(history):
    """
    Extracts accuracy from history and plots a professional line graph.
    """
    # 1. Extract the accuracy values
    accuracies = [val for _, val in history.metrics_distributed["accuracy"]]
    rounds = range(1, len(accuracies) + 1)
    
    # 2. Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, [a * 100 for a in accuracies], marker='o', linestyle='-', color='b', linewidth=2)
    
    # 3. Formatting for a Research Paper look
    plt.title('F-NIDS Global Convergence: Accuracy over Federated Rounds', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Global Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(90, 100) # Standardizes the view
    
    # 4. Save the plot to the images folder
    plt.savefig('images/accuracy_graph.png')
    print(" Results graph saved to images/accuracy_graph.png")
    plt.show()
