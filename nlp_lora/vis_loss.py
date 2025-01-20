#visualize the loss of the model

import os
import json
import matplotlib.pyplot as plt

json_file = "/home/paneah/Desktop/data-selection-for-bat/dsbat/models/nlp/lora/arc-c/results.json"

with open(json_file, 'r') as f:
    data = json.load(f)

train_losses = data["train_losses"]
eval_losses = data["eval_losses"]

#put each loss in separate plot
# Create two separate subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the training losses
ax1.plot(train_losses, color="blue", label="Training Loss")
ax1.set_title("Training Losses")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot the evaluation losses
ax2.plot(eval_losses, color="red", label="Evaluation Loss")
ax2.set_title("Evaluation Losses")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()

#save the plot

output_dir = os.path.dirname(json_file)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, "loss_plot.png"))