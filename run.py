import torch

def create_samples(n, device):
    # Step 1: Create a tensor of zeros with size (n, 100, 3)
    tensor = torch.zeros((n, 100, 3), device = device, dtype=torch.bool)

    # Step 2: Randomly assign one of each of the size 100 dim to have a 1
    indices = torch.randint(0, 100, (n, 3), device = device)
    tensor[torch.arange(n).unsqueeze(1), indices, torch.arange(3)] = 1

    # Step 3: Randomly zero out one row along the size 3 dim
    zero_indices = torch.randint(0, 100, (n,), device = device)
    tensor[torch.arange(n), zero_indices, :] = 0

    return tensor

def filter_samples(tensor):
    # Check if the first 99 rows along the 100 dim within the 3 dim at index 0 contain 99 zeros
    condition = (tensor[:, :99, 0] == 0).sum(dim=1) == 99

    # Filter out the tensors that meet the condition
    filtered_tensor = tensor[condition]

    return filtered_tensor

# usage
n = 10_000_000
result_tensor = create_samples(n, 0)
print(result_tensor[0,:,:])

result_tensor = filter_samples(result_tensor)
print(result_tensor[0,:,:])

# Get the average
print(result_tensor[:,99,0].float().mean())
