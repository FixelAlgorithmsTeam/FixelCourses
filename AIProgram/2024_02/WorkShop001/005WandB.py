import wandb
import random
import plotly.express as px
import pandas as pd

# start a new wandb run to track this script
wandb.init(
    project="ml-workshop",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# Simulate training
epochs = 10
offset = random.random() / 5
metrics = []

for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    metrics.append({'epoch': epoch, 'acc': acc, 'loss': loss})

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# Create a DataFrame
df = pd.DataFrame(metrics)

# Create a plot using plotly express
fig = px.line(df, x='epoch', y=['acc', 'loss'], title='Accuracy and Loss over Epochs', labels={
    'value': 'Metric Value', 'variable': 'Metrics'
})

# Show plot in the notebook (optional if you're running this in a notebook)
# fig.show()

# save fig
# fig.write_image('fig.png')

# Log the plot to wandb
# wandb.log({"Accuracy and Loss Plot": wandb.Plotly(fig)})

# Optional: finish the wandb run, necessary in notebooks
wandb.finish()
