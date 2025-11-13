import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = r'dino_v3/outputs_dino/training_20250928_010043.log'

epochs = []
losses = []

# Regex to match epoch and loss
pattern = re.compile(r'Epoch (\d+) completed\. Average loss: ([0-9.]+)')

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.tight_layout()
plt.show()
