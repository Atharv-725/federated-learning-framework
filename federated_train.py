import torch
import torchvision.transforms as transforms
import torchvision
from models.cnn_model import CNN
from clients.fl_client import FederatedClient
from server.fl_server import FederatedServer
from utils.data_utils import load_cifar10, split_data_iid, get_client_dataloader
# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLIENTS = 10
NUM_ROUNDS  = 5
LOCAL_EPOCHS = 3

# Load data
print("\nLoading data...")
trainset, testset = load_cifar10()
client_data = split_data_iid(trainset, num_clients=NUM_CLIENTS)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Create clients
clients = []
for i in range(NUM_CLIENTS):
    dataloader = get_client_dataloader(trainset, client_data[i])
    clients.append(FederatedClient(client_id=i, dataloader=dataloader, device=device))

# Create global model and server
global_model = CNN().to(device)
server = FederatedServer(global_model)

print(f"\nStarting Federated Learning with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds...\n")

# Federated training loop
for round_num in range(1, NUM_ROUNDS + 1):
    print(f"--- Round {round_num}/{NUM_ROUNDS} ---")

    # Each client trains locally
    client_weights = []
    for client in clients:
        weights = client.train(global_model.state_dict(), epochs=LOCAL_EPOCHS)
        client_weights.append(weights)

    # Server aggregates
    server.aggregate(client_weights)

    # Evaluate global model
    accuracy = server.evaluate(testloader, device)
    print(f"  🌍 Global Model Accuracy after Round {round_num}: {accuracy:.2f}%\n")

print("✅ Federated Training Complete!")
print(f"\nBaseline Accuracy (centralized): 73.55%")
print(f"Federated Accuracy (final round): {server.round_accuracies[-1]:.2f}%")