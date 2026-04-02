import flwr as fl
from src.preprocessing import load_and_preprocess_data, partition_data
from src.model import create_initial_model
from src.client import NIDSClient
from src.utils import weighted_average, plot_results

def main():
    # 1. Load and Prepare the Big Data (2.5M Rows)
    # Assumes the file is in the data/ folder as per your structure
    X, y = load_and_preprocess_data('data/cicids2017_cleaned.csv')
    
    # 2. Partition data for 5 clients (The experiment we discussed!)
    NUM_CLIENTS = 5
    X_clients, y_clients = partition_data(X, y, num_clients=NUM_CLIENTS)

    # 3. Define the Virtual Client Engine logic
    def client_fn(cid: str):
        # Create a fresh "brain" for this client
        model = create_initial_model()
        
        # Give the client its specific slice of the 2.5M rows
        idx = int(cid)
        return NIDSClient(model, X_clients[idx], y_clients[idx]).to_client()

    # 4. Configure the Federation Strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # 5. Kick off the Simulation
    print(f"🚀 Starting Federated Learning with {NUM_CLIENTS} clients...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # 6. Export the Results
    print("📊 Simulation Complete. Generating results...")
    plot_results(history)

if __name__ == "__main__":
    main()
