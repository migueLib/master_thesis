# External
import torch
import numpy as np


def predict(model, dataloader, device):
    # Get softmax
    sm = torch.nn.Softmax()
    
    # Initialize output arrays
    trues = np.array([])
    predictions = np.array([])
    probabilities = np.array([])

    # Iterate over data
    for i, (inputs, labels) in enumerate(dataloader):
        # Move the data to the specified device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.set_grad_enabled(False):

            # Inception-v3 can output an auxiliary layer
            try:
                outputs, aux = model(inputs)
            except ValueError:
                outputs = model(inputs)
            
            # Use the soft max to extract the probabilites
            prob = sm(outputs)
            _, predicted = torch.max(outputs, 1)
            
        # Append and output everything
        trues = np.append(trues, labels.cpu().detach().numpy())
        predictions = np.append(predictions, predicted.cpu().detach().numpy())
        probabilities = np.append(probabilities, prob.cpu().detach().numpy())
    
    return trues, predictions, probabilities

if __name__ == '__main__':
    pass