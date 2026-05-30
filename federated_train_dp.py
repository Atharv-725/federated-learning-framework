import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from models.cnn_model import CNN
from utils.data_utils import load_cifar10, split_data_iid, get_client_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_CLIENTS  = 10
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 3
MAX_GRAD_NORM = 1.0
EPSILON = 50.0   # privacy budget

def train_with_dp(model, dataloader, epochs, device):
    model = model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=EPSILON,
        target_delta=1e-5,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model._module.state_dict()

def aggregate(client_weights_list):
    avg_weights = client_weights_list[0].copy()
    for key in avg_weights.keys():
        for i in range(1, len(client_weights_list)):
            avg_weights[key] += client_weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights_list))
    return avg_weights

def evaluate(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Load data
print("\nLoading data...")
trainset, testset = load_cifar10()
client_data = split_data_iid(trainset, num_clients=NUM_CLIENTS)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Global model
global_model = CNN().to(device)

print(f"\n🔒 Starting Federated Learning WITH Differential Privacy")
print(f"   Privacy Budget (epsilon): {EPSILON}")
print(f"   Max Grad Norm: {MAX_GRAD_NORM}\n")

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"--- Round {round_num}/{NUM_ROUNDS} ---")
    client_weights = []

    for i in range(NUM_CLIENTS):
        dataloader = get_client_dataloader(trainset, client_data[i])
        local_model = CNN().to(device)
        local_model.load_state_dict(global_model.state_dict())
        weights = train_with_dp(local_model, dataloader, LOCAL_EPOCHS, device)
        client_weights.append(weights)
        print(f"  ✅ Client {i} finished (with DP)")

    avg_weights = aggregate(client_weights)
    global_model.load_state_dict(avg_weights)

    accuracy = evaluate(global_model, testloader, device)
    print(f"  🔒 Global Accuracy after Round {round_num}: {accuracy:.2f}%\n")

print("✅ Federated Training with Differential Privacy Complete!")
print(f"\n📊 Final Results:")
print(f"   Centralized Baseline:         73.55%")
print(f"   Federated (no DP):            63.69%")
print(f"   Federated (with DP, ε={EPSILON}):  {accuracy:.2f}%")