#! /usr/bin/env python3
import torch
import numpy as np


####################
# RNN
####################
class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):

        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        # Validate the nonlinearity option
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        # Define linear transformations
        self.x2h = torch.nn.Linear(input_size, hidden_size, bias=bias)  # Input to hidden
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)  # Hidden to hidden
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        # If no hidden state is provided, initialize with zeros
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)
        # Combine input and hidden state
        hy = self.x2h(input) + self.h2h(hx)
        # Apply nonlinearity
        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)
        return hy


class RNN(torch.nn.Module):
    def __init__(self, channels=5, img_size=80, hidden_size=500, num_layers=3, bias=True, output_size=3, activation='tanh'):
        super(RNN, self).__init__()
        self.name = "RNN"
        self.input_size = 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.out_ch = channels

        """
        convolutional NN
        """  
        self.channels = channels
        self.img_shape = (img_size, img_size)  
        # layers
        self.conv1 = torch.nn.Conv2d(self.channels, self.out_ch, 3)
        self.max_pool = torch.nn.MaxPool2d(7, 3)

        #width=((W-F+2*P )/S)+1 
        p_w = (((img_size-2-7)/3)+1) // 1
        p_h = (((img_size-2-7)/3)+1) // 1
        print("p_w", p_w)
        print("p_h", p_h)
        flat_size = int(p_w * p_h + 2*img_size)
        print("flat_size", flat_size)
        self.input_size = flat_size
        print("self.input_size (ch): ", self.input_size)

        # Create a list to hold RNN cells
        self.rnn_cell_list = torch.nn.ModuleList()
        # Initialize RNN cells based on activation function
        if activation == 'tanh':
            # First layer takes input_size, rest take hidden_size as input
            self.rnn_cell_list.append(RNNCell(self.input_size, self.hidden_size, self.bias, "tanh"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.hidden_size, self.hidden_size, self.bias, "tanh"))
        elif activation == 'relu':
            self.rnn_cell_list.append(RNNCell(self.input_size, self.hidden_size, self.bias, "relu"))
            for l in range(1, self.num_layers):
                self.rnn_cell_list.append(RNNCell(self.hidden_size, self.hidden_size, self.bias, "relu"))
        else:
            raise ValueError("Invalid activation.")

        #This is not used but trained model have it so can't run without this line
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)#Do not delete

        self.flat1 = torch.nn.Linear(self.hidden_size, 120)
        self.flat2 = torch.nn.Linear(120, 80)
        self.out = torch.nn.Linear(80, 3)

        self.dropout_20 = torch.nn.Dropout(p=0.2)
        self.dropout_15 = torch.nn.Dropout(p=0.15)
        self.dropout_10 = torch.nn.Dropout(p=0.1)

    def forward(self, x, hx=None):
        # vector: d, th
        vect = x[:, :, -2:, :]  # [batch_s, channel, col, row]
        vect = torch.flatten(vect, 2)
        # image
        # [batch_s, channel, col, row]
        x = x[:, :, :-2]

        x = self.conv1(x)
        x = self.max_pool(x)
        x = torch.nn.functional.relu(x)
        
        x = torch.flatten(x, 2)
        input = torch.cat((x, vect), 2)
        """
        Forward pass of the RNN.
        
        Args:
            input: Input tensor of shape (batch_size, sequence length, input_size)
            hx: Initial hidden state (optional)
        
        Returns:
            out: Output tensor of shape (batch_size, output_size)
        """
        if hx is None:
            h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda()
        else:
            h0 = hx

        outs = []
        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])
        # Process each time step
        for t in range(input.size(1)):
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                hidden[layer] = hidden_l
            outs.append(hidden_l)
        # Take only last time step
        out = outs[-1].squeeze()
        # Pass through final fully connected layer
        x = self.flat1(out)
        x = torch.nn.functional.relu(x)
        x = self.dropout_20(x)
        x = self.flat2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout_15(x)
        x = self.out(x)
        x = torch.nn.functional.tanh(x)
        return x