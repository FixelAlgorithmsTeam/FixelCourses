#%%
# PART 1 - Log model in a running experiment
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Initialize a W&B run
run = wandb.init(entity="validit", project="mnist-workshop", config=config)

# Training algorithm
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Save model
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# Log the model
run.log_model(path=full_path, name="MNIST")

# Explicitly tell W&B to end the run.
run.finish()

#%% PART 2
# Create the model
run = wandb.init(entity="validit", project="mnist-workshop", config=config)
run.link_model(path=full_path, registered_model_name="MNIST")
run.finish()


#%% PART 3
# use the model
entity = "validit"
project = "mnist-workshop"
alias = "latest"  # semantic nickname or identifier for the model version
model_artifact_name = "MNIST"

# Initialize a run
run = wandb.init()
# Access and download model. Returns path to downloaded artifact

downloaded_model_path = run.use_model(name=f"{entity}/{project}/{model_artifact_name}:{alias}")
print(downloaded_model_path)

# %%
