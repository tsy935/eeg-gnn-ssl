import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle
import utils


class LSTMModel(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(LSTMModel, self).__init__()
        
        num_nodes = args.num_nodes
        rnn_units = args.rnn_units  
        num_rnn_layers = args.num_rnn_layers             
        input_dim = args.input_dim
        
        self._input_dim = input_dim
        self._num_nodes = num_nodes
        self._num_rnn_layers = num_rnn_layers
        self._rnn_units = rnn_units
        self._num_classes = num_classes
        self._device = device
                
        self.lstm = nn.LSTM(input_dim * num_nodes, 
                          rnn_units, 
                          num_rnn_layers,
                          batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout) # dropout layer before final FC
        self.fc = nn.Linear(rnn_units, num_classes) # final FC layer
        self.relu = nn.ReLU()  
    
    def forward(self, inputs, seq_lengths):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_nodes, input_dim)
            seq_lengths: (batch_size, )
        """
        batch_size, max_seq_len, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))  # (batch_size, max_seq_len, num_nodes*input_dim)
        
        # initialize hidden states
        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

        # LSTM
        output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)

        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True) # (batch_size, rnn_units)
        last_out = last_out.to(self._device)
        
        # Dropout -> ReLU -> FC
        logits = self.fc(self.relu(self.dropout(last_out))) # (batch_size, num_classes)       
        
        return logits    

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_().to(self._device)
        return hidden, cell