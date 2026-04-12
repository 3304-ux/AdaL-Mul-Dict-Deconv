import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os
from DATA import load_data
from Model_supervised import AdaLISTA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_multi_dict_model(model, train_loader, test_loader, epochs, lr, weight_decay, step_size, gamma):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_train = []
    loss_test = []
    for epoch in range(epochs):
        model.current_epoch = epoch
        model.train()
        train_loss = 0

        for batch_idx, (b_y, b_D, b_x, b_weiguiyi_y, b_dict_indices) in enumerate(train_loader):
            optimizer.zero_grad()
            b_y = b_y.to(device);
            b_D = b_D.to(device);
            b_x = b_x.to(device);
            b_dict_indices = b_dict_indices.to(device)
            x_hat = model(b_y, b_D, b_dict_indices)
            mse_loss = torch.nn.functional.mse_loss(x_hat, b_x)
            l1_loss = torch.norm(x_hat, p=1)

            A = A  # weight value
            B = B  # weight value
            loss = A * mse_loss + B * l1_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        loss_train.append(train_loss / len(train_loader))
        model.eval()
        test_loss = 0


        with torch.no_grad():
            for b_y, b_D, b_x, b_weiguiyi_y, b_dict_indices in test_loader:
                b_y = b_y.to(device);
                b_D = b_D.to(device);
                b_x = b_x.to(device);
                b_dict_indices = b_dict_indices.to(device)
                x_hat = model(b_y, b_D, b_dict_indices)
                mse_loss = torch.nn.functional.mse_loss(x_hat, b_x)
                l1_loss = torch.norm(x_hat, p=1)
                A = A  # weight value
                B = B  # weight value
                loss = A * mse_loss + B * l1_loss
                test_loss += loss.item()
        loss_test.append(test_loss / len(test_loader))
        scheduler.step()
        print(f'Epoch {epoch}, Train Loss: {loss_train[-1]:.4f}, Test Loss: {loss_test[-1]:.4f}')
    return model



def main():
    train_loader, D, train_inputs = load_data()

    n_features, n_atoms = D.shape
    model = AdaLISTA(n_features, n_atoms, max_iter=100, lambd=1.0, num_dicts=5)
    model = model.to(device)

    trained_model = train_multi_dict_model(
    )

if __name__ == "__main__":
    main()
