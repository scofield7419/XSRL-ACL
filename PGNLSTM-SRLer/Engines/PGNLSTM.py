import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
            np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        pass
    else:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def lstm_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden
    gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def left2right_lstm(params, input, masks, initial, drop_masks):
    w_ih, w_hh, b_ih, b_hh = params
    max_time = input.size(0)
    output = []
    hx = initial
    for time in range(max_time):
        h_next, c_next = lstm_cell(input[time], hx, w_ih, w_hh, b_ih, b_hh)
        h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
        c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
        output.append(h_next)
        if drop_masks is not None: h_next = h_next * drop_masks
        hx = (h_next, c_next)
    output = torch.stack(output, 0)
    return output, hx


def right2left_lstm(params, input, masks, initial, drop_masks):
    w_ih, w_hh, b_ih, b_hh = params
    max_time = input.size(0)
    output = []
    hx = initial
    for time in reversed(range(max_time)):
        h_next, c_next = lstm_cell(input[time], hx, w_ih, w_hh, b_ih, b_hh)
        h_next = h_next * masks[time] + initial[0] * (1 - masks[time])
        c_next = c_next * masks[time] + initial[1] * (1 - masks[time])
        output.append(h_next)
        if drop_masks is not None: h_next = h_next * drop_masks
        hx = (h_next, c_next)
    output.reverse()
    output = torch.stack(output, 0)
    return output, hx


class PGNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, task_dim_size, \
                 num_layers=1, batch_first=False, \
                 bidirectional=False, dropout_in=0, dropout_out=0):
        super(PGNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.num_directions = 2 if bidirectional else 1
        self.task_dim_size = task_dim_size

        self._all_name_weights, self._all_weights = [], []
        self.fparam_indices = [-1 for idx in range(num_layers)]
        self.bparam_indices = [-1 for idx in range(num_layers)]
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            self.fparam_indices[layer] = len(self._all_weights)

            param_ih = Parameter(torch.Tensor(4 * hidden_size, layer_input_size, task_dim_size))
            param_ih_name = 'fweights_ih_l{}'.format(layer)
            self._all_name_weights.append(param_ih_name)
            self._all_weights.append(param_ih)
            setattr(self, param_ih_name, param_ih)
            param_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size, task_dim_size))
            param_hh_name = 'fweights_hh_l{}'.format(layer)
            self._all_name_weights.append(param_hh_name)
            self._all_weights.append(param_hh)
            setattr(self, param_hh_name, param_hh)
            bias_ih = Parameter(torch.Tensor(4 * hidden_size, task_dim_size))
            bias_ih_name = 'fbias_ih_l{}'.format(layer)
            self._all_name_weights.append(bias_ih_name)
            self._all_weights.append(bias_ih)
            setattr(self, bias_ih_name, bias_ih)
            bias_hh = Parameter(torch.Tensor(4 * hidden_size, task_dim_size))
            bias_hh_name = 'fbias_hh_l{}'.format(layer)
            self._all_name_weights.append(bias_hh_name)
            self._all_weights.append(bias_hh)
            setattr(self, bias_hh_name, bias_hh)

            if self.bidirectional:
                self.bparam_indices[layer] = len(self._all_weights)

                param_ih = Parameter(torch.Tensor(4 * hidden_size, layer_input_size, task_dim_size))
                param_ih_name = 'bweights_ih_l{}'.format(layer)
                self._all_name_weights.append(param_ih_name)
                self._all_weights.append(param_ih)
                setattr(self, param_ih_name, param_ih)
                param_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size, task_dim_size))
                param_hh_name = 'bweights_hh_l{}'.format(layer)
                self._all_name_weights.append(param_hh_name)
                self._all_weights.append(param_hh)
                setattr(self, param_hh_name, param_hh)
                bias_ih = Parameter(torch.Tensor(4 * hidden_size, task_dim_size))
                bias_ih_name = 'bbias_ih_l{}'.format(layer)
                self._all_name_weights.append(bias_ih_name)
                self._all_weights.append(bias_ih)
                setattr(self, bias_ih_name, bias_ih)
                bias_hh = Parameter(torch.Tensor(4 * hidden_size, task_dim_size))
                bias_hh_name = 'bbias_hh_l{}'.format(layer)
                self._all_name_weights.append(bias_hh_name)
                self._all_weights.append(bias_hh)
                setattr(self, bias_hh_name, bias_hh)

            self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
            param_ih_name = 'fweights_ih_l{}'.format(layer)
            param_hh_name = 'fweights_hh_l{}'.format(layer)
            param_ih = self.__getattr__(param_ih_name)
            param_hh = self.__getattr__(param_hh_name)
            W = orthonormal_initializer(self.hidden_size, self.hidden_size + layer_input_size)
            W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
            py_w_h = torch.from_numpy(np.concatenate([W_h] * 4, 0)).unsqueeze(dim=2)
            py_w_h = py_w_h.expand(-1, -1, self.task_dim_size)
            py_w_x = torch.from_numpy(np.concatenate([W_x] * 4, 0)).unsqueeze(dim=2)
            py_w_x = py_w_x.expand(-1, -1, self.task_dim_size)
            param_hh.data.copy_(py_w_h)
            param_ih.data.copy_(py_w_x)

            if self.bidirectional:
                param_ih_name = 'bweights_ih_l{}'.format(layer)
                param_hh_name = 'bweights_hh_l{}'.format(layer)
                param_ih = self.__getattr__(param_ih_name)
                param_hh = self.__getattr__(param_hh_name)
                W = orthonormal_initializer(self.hidden_size, self.hidden_size + layer_input_size)
                W_h, W_x = W[:, :self.hidden_size], W[:, self.hidden_size:]
                py_w_h = torch.from_numpy(np.concatenate([W_h] * 4, 0)).unsqueeze(dim=2)
                py_w_h = py_w_h.expand(-1, -1, self.task_dim_size)
                py_w_x = torch.from_numpy(np.concatenate([W_x] * 4, 0)).unsqueeze(dim=2)
                py_w_x = py_w_x.expand(-1, -1, self.task_dim_size)
                param_hh.data.copy_(py_w_h)
                param_ih.data.copy_(py_w_x)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(self.__getattr__(name), 0)

    def forward(self, task_emb, input, masks, initial=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = torch.unsqueeze(masks.transpose(0, 1), dim=2)
        max_time, batch_size, _ = input.size()
        masks = masks.expand(-1, -1, self.hidden_size)

        if initial is None:
            initial = Variable(input.data.new(batch_size, self.hidden_size).zero_())
            initial = (initial, initial)
        h_n = []
        c_n = []

        all_weights = []
        task_emb = task_emb.reshape(-1, 1)
        for cur_name_weight, cur_weight in zip(self._all_name_weights, self._all_weights):
            domain_task_lstm = torch.mm(cur_weight.reshape(-1, self.task_dim_size), task_emb)
            if "bias" in cur_name_weight:
                domain_task_lstm = domain_task_lstm.reshape(4 * self.hidden_size)
            else:
                domain_task_lstm = domain_task_lstm.reshape(4 * self.hidden_size, -1)
            all_weights.append(domain_task_lstm)

        for layer in range(self.num_layers):
            max_time, batch_size, input_size = input.size()
            input_mask, hidden_mask = None, None
            if self.training:
                input_mask = input.data.new(batch_size, input_size).fill_(1 - self.dropout_in)
                input_mask = Variable(torch.bernoulli(input_mask), requires_grad=False)
                input_mask = input_mask / (1 - self.dropout_in)
                input_mask = torch.unsqueeze(input_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
                input = input * input_mask

                hidden_mask = input.data.new(batch_size, self.hidden_size).fill_(1 - self.dropout_out)
                hidden_mask = Variable(torch.bernoulli(hidden_mask), requires_grad=False)
                hidden_mask = hidden_mask / (1 - self.dropout_out)

            findex = self.fparam_indices[layer]
            fparams = all_weights[findex], all_weights[findex + 1], \
                      all_weights[findex + 2], all_weights[findex + 3]
            layer_output, (layer_h_n, layer_c_n) = left2right_lstm(params=fparams, \
                                                                   input=input, masks=masks, initial=initial,
                                                                   drop_masks=hidden_mask)

            if self.bidirectional:
                bindex = self.bparam_indices[layer]
                bparams = all_weights[bindex], all_weights[bindex + 1], \
                          all_weights[bindex + 2], all_weights[bindex + 3]
                blayer_output, (blayer_h_n, blayer_c_n) = right2left_lstm(params=bparams, \
                                                                          input=input, masks=masks, initial=initial,
                                                                          drop_masks=hidden_mask)

            h_n.append(torch.cat([layer_h_n, blayer_h_n], 1) if self.bidirectional else layer_h_n)
            c_n.append(torch.cat([layer_c_n, blayer_c_n], 1) if self.bidirectional else layer_c_n)
            input = torch.cat([layer_output, blayer_output], 2) if self.bidirectional else layer_output

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return input, (h_n, c_n)

