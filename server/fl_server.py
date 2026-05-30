import torch
import copy

class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.round_accuracies = []

    def aggregate(self, client_weights_list):
        """FedAvg: Average all client weights"""
        avg_weights = copy.deepcopy(client_weights_list[0])

        for key in avg_weights.keys():
            for i in range(1, len(client_weights_list)):
                avg_weights[key] += client_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(client_weights_list))

        self.global_model.load_state_dict(avg_weights)
        print(f"  🔗 Server aggregated weights from {len(client_weights_list)} clients")
        return avg_weights

    def evaluate(self, testloader, device):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self.round_accuracies.append(accuracy)
        return accuracy