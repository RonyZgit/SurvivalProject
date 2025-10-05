import pandas as pd
import numpy as np
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
from pycox.models.data import pair_rank_mat
import scipy.integrate
import random


torch.manual_seed(18)
random.seed(18)

# Loading normalized CSV
df = pd.read_csv("Primary_Biliary_Cirrhosis_transformed_with_age.csv")
df = df.dropna()
print(df.head())

# Separate features, time, event
y_time = df['time'].values
y_event = df['status'].values
X = df.drop(columns=['time', 'status']).values.astype(np.float32)
events = ['transplanted', 'death']

# Split to training set (0.8) and test (0.2)
X_trainval_old, X_test_old, E_trainval_old, E_test_old, T_trainval_old, T_test_old = train_test_split(
    X, y_event, y_time, test_size=0.20)

# Split the training set into train (0.8) and validation set (0.2)
X_train_old, X_val_old, E_train_old, E_val_old, T_train_old, T_val_old = train_test_split(
    X_trainval_old, E_trainval_old, T_trainval_old, test_size=0.20)

print(f'Training set size {len(X_train_old)}')
print(f'Validation set size {len(X_val_old)}')
print(f'Test set size {len(X_test_old)}')

### Discretizing time ###
class TransformLable(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


num_time_steps = 10
label_transform = TransformLable(num_time_steps, scheme='quantiles')  # Split time by quantiles

T_train_discrete, E_train_discrete = label_transform.fit_transform(T_train_old, E_train_old)
T_val_discrete, E_val_discrete = label_transform.transform(T_val_old, E_val_old)

time_grid_train = label_transform.cuts
output_num_time_steps = len(time_grid_train)
print(f'Number of time steps in the model: {output_num_time_steps}')
print('Time grid:', time_grid_train)

# Prepare for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train_old, dtype=torch.float32, device=device)
T_train = torch.tensor(T_train_discrete, dtype=torch.int64, device=device)
E_train = torch.tensor(E_train_discrete, dtype=torch.int32, device=device)
train_data = list(zip(X_train, T_train, E_train))

X_val = torch.tensor(X_val_old, dtype=torch.float32, device=device)
T_val = torch.tensor(T_val_discrete, dtype=torch.int64, device=device)
E_val = torch.tensor(E_val_discrete, dtype=torch.int32, device=device)
val_data = list(zip(X_val, T_val, E_val))


### DeepHit model ###
class DeepHit_model(nn.Module):
    def __init__(self, num_covariates, num_events, num_hidden):
        super(DeepHit_model, self).__init__()

        self.num_critical_events = num_events
        self.mlp = nn.Sequential(
            nn.Linear(num_covariates, num_hidden),  # input to hidden layers
            nn.ReLU(),  # activation function
            nn.Linear(num_hidden, num_events * output_num_time_steps)  # output layer
            ).to(device)

    def forward(self, inputs):  # Neural Network method
        mlp_output = self.mlp(inputs)
        return mlp_output.view(inputs.size(0), self.num_critical_events, -1)

num_input_features = X_train.size(1)
base_nn = DeepHit_model(num_input_features, 2, 5).to(device)
deephit_model = DeepHit(base_nn, alpha=0.9, sigma=0.1, device=device,
                        duration_index=time_grid_train)
deephit_loss = deephit_model.loss  # Loss function

# Minibatch gradient descent
num_epochs = 50
batch_size = 16
learning_rate = 0.005

train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling the training set for minibatch
val_loader = DataLoader(val_data, batch_size, shuffle=False)  # not shuffling the validation set

optimizer = torch.optim.Adam(base_nn.parameters(), lr=learning_rate)
train_epoch_losses = []
val_epoch_losses = []
best_val_loss = float('inf')
best_params = None
best_epoch_ind = None

for epoch_ind in range(num_epochs):
    base_nn.train()
    for X_batch, T_batch, E_batch in train_loader:
        nn_output = base_nn(X_batch)
        rank_matrix = pair_rank_mat(T_batch.cpu().numpy(), E_batch.cpu().numpy())
        rank_matrix = torch.tensor(rank_matrix, dtype=torch.int, device=device)
        loss_batch = deephit_loss(nn_output, T_batch, E_batch, rank_matrix)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

    # Loss functions of training and validation sets
    base_nn.eval()
    with torch.no_grad():
        train_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0
        for X_batch, T_batch, E_batch in train_loader:
            batch_num_points = X_batch.size(0)
            nn_output = base_nn(X_batch)
            rank_matrix = pair_rank_mat(T_batch.cpu().numpy(), E_batch.cpu().numpy())
            rank_matrix = torch.tensor(rank_matrix, dtype=torch.int, device=device)
            train_loss += deephit_loss(nn_output, T_batch, E_batch, rank_matrix) * batch_num_points
            num_points += batch_num_points
        train_loss = float(train_loss / num_points)
        train_epoch_losses.append(train_loss)

        val_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0
        for X_batch, T_batch, E_batch in val_loader:
            batch_num_points = X_batch.size(0)
            nn_output = base_nn(X_batch)
            rank_matrix = pair_rank_mat(T_batch.cpu().numpy(), E_batch.cpu().numpy())
            rank_matrix = torch.tensor(rank_matrix, dtype=torch.int, device=device)
            val_loss += deephit_loss(nn_output, T_batch, E_batch, rank_matrix) * batch_num_points
            num_points += batch_num_points
        val_loss = float(val_loss / num_points)
        val_epoch_losses.append(val_loss)

        if val_loss < best_val_loss: # Update best epoch by loss
            best_val_loss = val_loss
            best_epoch_ind = epoch_ind
            best_params = deepcopy(base_nn.state_dict())
print(f'Lowest DeepHit loss for validation set is {round(best_val_loss, 4)}, epoch {best_epoch_ind + 1}')
base_nn.load_state_dict(best_params)

# Plot of loss as a function of epochs
plt.plot(range(1, num_epochs + 1), train_epoch_losses, label='Training')
plt.plot(range(1, num_epochs + 1), val_epoch_losses, '--', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DeepHit Loss as a Function of Epochs')
plt.legend()
plt.show()

# Predicting
cif_test = deephit_model.predict_cif(X_test_old, batch_size=batch_size, to_cpu=True, numpy=True)

# C-index for death in test set
cif_df = pd.DataFrame(1 - cif_test[1, :, :], index=time_grid_train)
eval_pycox = EvalSurv(cif_df, T_test_old, E_test_old == 2)
c_td = eval_pycox.concordance_td('antolini')  # Antolini method for time dependent C-index
print(f'C-index for death: {round(c_td, 4)}')

# Brier Score for death in test set
eval_pycox = EvalSurv(cif_df, T_test_old, E_test_old == 2)
eval_pycox.add_km_censor()
brier_scores = eval_pycox.brier_score(time_grid_train)

# Integrated Brier Score (IBS) for death in test set
scipy.integrate.simps = scipy.integrate.simpson
ibs = eval_pycox.integrated_brier_score(time_grid_train)
print(f'Integrated Brier Score for death: {round(ibs, 4)}')
plt.plot(time_grid_train, brier_scores)
plt.xlabel('Time')
plt.ylabel('Brier Score')
plt.title('Brier Score Over Time for Death')
plt.grid(True)
plt.show()
