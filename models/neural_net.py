import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau


class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.config = config
        self.input_size = config.INPUT_SIZE
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.output_size = config.OUTPUT_SIZE
        self.dropout_rate = config.DROPOUT_RATE

        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=config.ATTENTION_HEADS, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=config.LR_PATIENCE, factor=config.LR_FACTOR, min_lr=config.MIN_LEARNING_RATE)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(x)
        x = self.dropout(x)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.view(x.size(0), -1, self.hidden_size)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual connection
        x = F.relu(self.fc1(attn_out.squeeze(1)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def train_step(self, state, action, reward):
        self.optimizer.zero_grad()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        predicted_q_values = self(state_tensor)
        target_q_values = predicted_q_values.clone().detach()
        target_q_values[0, action.value] = reward
        loss = F.mse_loss(predicted_q_values, target_q_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def get_action_probabilities(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self(state_tensor)
            return F.softmax(q_values, dim=1).squeeze()

    def update_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train_batch(self, states, actions, rewards):
        self.optimizer.zero_grad()
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        predicted_q_values = self(states_tensor)
        q_values = predicted_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, rewards_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.MAX_GRAD_NORM)
        self.optimizer.step()
        self.scheduler.step(loss)
        return loss.item()
