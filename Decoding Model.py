# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def relu(x):
    return torch.maximum(torch.tensor(0.0), x)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))



class GroupCode(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(GroupCode, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, output_size) 
        self.leake_relu = nn.LeakyReLU()

    def forward(self, a):
        out1 = self.fc1(a)
        out1 = self.leake_relu(out1)  

        out2 = self.fc2(out1)
        out2 = self.leake_relu(out2)  

        out3 = self.fc3(out2)
        out3 = (torch.sin(out3) + 1) / 2 
        return out3



# Custom Dataset to Generate Binary Data
class BinaryDataset(Dataset):
    def __init__(self, input_size, v, num_samples, device):
        self.input_size = input_size
        self.v = v
        self.num_samples = num_samples 
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        a = torch.randint(0, 2, (self.input_size,), dtype=torch.float32).to(self.device)
        x = torch.sum(a * self.v, dim=0).to(torch.int) % 2
        x = x.to(self.device)
        return a, x


# Generate Random Test Samples
def generate_random_test_inputs(dim, num_samples, device):
    random_inputs = torch.randint(0, 2, (num_samples, dim), dtype=torch.float32).to(device)
    return random_inputs



def train_model(network, dim, v, epochs=1, learning_rate=0.0001, hidden_size1=640, hidden_size2=320, batch_size=64, min_loss=1e-5):

    input_size = dim
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    
    output_dir = "training_output"
    os.makedirs(output_dir, exist_ok=True)
    loss_file_path = os.path.join(output_dir, "epoch_loss.txt")
    loss_file = open(loss_file_path, "w")
    loss_file.write("Epoch, Avg Loss\n")


    dataset = BinaryDataset(input_size, v, num_samples=2**len(v), device=device)  # Generate 2**len(v) data samples
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    loss_values = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        epoch_loss = 0
        num_batches = 0  
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}") as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1).float() 
                
                optimizer.zero_grad()
                outputs = network(inputs)

                loss_val = loss_fn(outputs, targets)
                epoch_loss += loss_val.item()
                num_batches += 1
    
                loss_val.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss_val.item())
                pbar.update(1)

        
        avg_epoch_loss = epoch_loss / num_batches
        loss_values.append(avg_epoch_loss)
        loss_file.write(f"{epoch + 1}, {avg_epoch_loss:.6f}\n")

        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.6f}")
        
        if avg_epoch_loss < min_loss:
            print(f"Average loss has reached the target value ({min_loss}). Stopping training.")
            break
        torch.cuda.empty_cache()

    loss_file.close()


    plt.plot(range(1, len(loss_values) + 1), loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss during Training')
    plt.grid(True)
    loss_image_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_image_path)

    return network



def test_model(network, dim, v, num_samples=10, cmax=0.995, cmin=0.005):
    print("Make predictions:")

    test_input = generate_random_test_inputs(dim, num_samples, device)

    
    x_test = torch.sum(test_input * v, dim=1).to(torch.int)
    x_test = x_test % 2
    predictions = network(test_input)
    print("Prediction results:")
    predictions_values = predictions.detach().cpu().numpy()
    correct_count = 0
    for i, (pred, true_value) in enumerate(zip(predictions_values, x_test.cpu().numpy())):
        if pred[0] >= cmax:
            output_value = 1
        elif pred[0] <= cmin:
            output_value = 0
        else:
            output_value = 1 if pred[0] > 0.5 else 0

        print(f"Sample {i + 1} predicted value: {output_value}, true value: {true_value}")
        if output_value == true_value:
            correct_count += 1

    accuracy = correct_count / num_samples
    print(f"Test accuracy: {accuracy * 100:.2f}%")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the vector v
v = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],dtype=torch.float32).to(device)


network = GroupCode(input_size=len(v), hidden_size1=640, hidden_size2=320, output_size=1).to(device)

trained_network = train_model(network, dim=len(v), v=v, epochs=20, learning_rate=0.0001, min_loss=1e-3)

test_model(trained_network, dim=len(v), v=v, num_samples=10, cmax=0.995, cmin=0.005)


