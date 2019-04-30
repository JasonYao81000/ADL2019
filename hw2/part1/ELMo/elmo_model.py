from typing import Optional, Tuple, List, Callable, Union
import h5py
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
import itertools

from .utils import util

RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # pylint: disable=invalid-name
RnnStateStorage = Tuple[torch.Tensor, ...]  # pylint: disable=invalid-name


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask
    
def block_orthogonal(tensor: torch.Tensor,
                     split_sizes: List[int],
                     gain: float = 1.0) -> None:
        if isinstance(tensor, Variable):
            sizes = list(tensor.size())
            if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
                raise ConfigurationError("tensor dimensions must be divisible by their respective "
                                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
            indexes = [list(range(0, max_size, split))
                       for max_size, split in zip(sizes, split_sizes)]
            for block_start_indices in itertools.product(*indexes):
                index_and_step_tuples = zip(block_start_indices, split_sizes)
                block_slice = tuple([slice(start_index, start_index + step)
                                     for start_index, step in index_and_step_tuples])
                tensor[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].contiguous(), gain=gain)



class Projection(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None) -> None:
        super(Projection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability

        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size, bias=True)

        self.state_projection = torch.nn.Linear(cell_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size].fill_(1.0)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.FloatTensor,
                batch_lengths: List[int],
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):

        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]

        output_accumulator = Variable(inputs.data.new(batch_size,
                                                      total_timesteps,
                                                      self.hidden_size).fill_(0))
        if initial_state is None:
            full_batch_previous_memory = Variable(inputs.data.new(batch_size,
                                                                  self.cell_size).fill_(0))
            full_batch_previous_state = Variable(inputs.data.new(batch_size,
                                                                 self.hidden_size).fill_(0))
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)

        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability,
                                            full_batch_previous_state)
        else:
            dropout_mask = None

        for timestep in range(total_timesteps):
            index = timestep if self.go_forward else total_timesteps - timestep - 1


            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < (len(batch_lengths) - 1) and \
                                batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1

            previous_memory = full_batch_previous_memory[0: current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0: current_length_index + 1].clone()
            timestep_input = inputs[0: current_length_index + 1, index]

            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)

            input_gate = torch.sigmoid(projected_input[:, (0 * self.cell_size):(1 * self.cell_size)] +
                                       projected_state[:, (0 * self.cell_size):(1 * self.cell_size)])
            forget_gate = torch.sigmoid(projected_input[:, (1 * self.cell_size):(2 * self.cell_size)] +
                                        projected_state[:, (1 * self.cell_size):(2 * self.cell_size)])
            memory_init = torch.tanh(projected_input[:, (2 * self.cell_size):(3 * self.cell_size)] +
                                     projected_state[:, (2 * self.cell_size):(3 * self.cell_size)])
            output_gate = torch.sigmoid(projected_input[:, (3 * self.cell_size):(4 * self.cell_size)] +
                                        projected_state[:, (3 * self.cell_size):(4 * self.cell_size)])
            memory = input_gate * memory_init + forget_gate * previous_memory

            if self.memory_cell_clip_value:
                memory = torch.clamp(memory, -self.memory_cell_clip_value, self.memory_cell_clip_value)

            pre_projection_timestep_output = output_gate * torch.tanh(memory)

            timestep_output = self.state_projection(pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output,
                                              -self.state_projection_clip_value,
                                              self.state_projection_clip_value)

            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0: current_length_index + 1]

            full_batch_previous_memory = Variable(full_batch_previous_memory.data.clone())
            full_batch_previous_state = Variable(full_batch_previous_state.data.clone())
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1] = timestep_output
            output_accumulator[0:current_length_index + 1, index] = timestep_output

        final_state = (full_batch_previous_state.unsqueeze(0),
                       full_batch_previous_memory.unsqueeze(0))

        return output_accumulator, final_state



class ElmobiLm(util):
  def __init__(self, config, use_cuda=False):
    super(ElmobiLm, self).__init__(stateful=True)
    self.config = config
    self.use_cuda = use_cuda
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']
    cell_size = config['encoder']['dim']
    num_layers = config['encoder']['n_layers']
    memory_cell_clip_value = config['encoder']['cell_clip']
    state_projection_clip_value = config['encoder']['proj_clip']
    recurrent_dropout_probability = config['dropout']

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.cell_size = cell_size
    
    forward_layers = []
    backward_layers = []

    lstm_input_size = input_size
    go_forward = True
    for layer_index in range(num_layers):
      forward_layer = Projection(lstm_input_size,
                                             hidden_size,
                                             cell_size,
                                             go_forward,
                                             recurrent_dropout_probability,
                                             memory_cell_clip_value,
                                             state_projection_clip_value)
      backward_layer = Projection(lstm_input_size,
                                              hidden_size,
                                              cell_size,
                                              not go_forward,
                                              recurrent_dropout_probability,
                                              memory_cell_clip_value,
                                              state_projection_clip_value)
      lstm_input_size = hidden_size

      self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
      self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
      forward_layers.append(forward_layer)
      backward_layers.append(backward_layer)
    self.forward_layers = forward_layers
    self.backward_layers = backward_layers

  def forward(self, inputs, mask):
    batch_size, total_sequence_length = mask.size()
    stacked_sequence_output, final_states, restoration_indices = \
      self.sort_and_run_forward(self._lstm_forward, inputs, mask)

    num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
    if num_valid < batch_size:
      zeros = stacked_sequence_output.data.new(num_layers,
                                               batch_size - num_valid,
                                               returned_timesteps,
                                               encoder_dim).fill_(0)
      zeros = Variable(zeros)
      stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

      new_states = []
      for state in final_states:
        state_dim = state.size(-1)
        zeros = state.data.new(num_layers, batch_size - num_valid, state_dim).fill_(0)
        zeros = Variable(zeros)
        new_states.append(torch.cat([state, zeros], 1))
      final_states = new_states

    sequence_length_difference = total_sequence_length - returned_timesteps
    if sequence_length_difference > 0:
      zeros = stacked_sequence_output.data.new(num_layers,
                                               batch_size,
                                               sequence_length_difference,
                                               stacked_sequence_output[0].size(-1)).fill_(0)
      zeros = Variable(zeros)
      stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

    self._update_states(final_states, restoration_indices)

    return stacked_sequence_output.index_select(1, restoration_indices)


  def _lstm_forward(self, 
                    inputs: PackedSequence,
                    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
      Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
          
    if initial_state is None:
      hidden_states: List[Optional[Tuple[torch.Tensor,
                                   torch.Tensor]]] = [None] * len(self.forward_layers)
    elif initial_state[0].size()[0] != len(self.forward_layers):
      raise Exception("Initial states were passed to forward() but the number of "
                               "initial states does not match the number of layers.")
    else:
      hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

    inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
    forward_output_sequence = inputs
    backward_output_sequence = inputs

    final_states = []
    sequence_outputs = []
    for layer_index, state in enumerate(hidden_states):
      forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
      backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

      forward_cache = forward_output_sequence
      backward_cache = backward_output_sequence

      if state is not None:
        forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
        forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
        forward_state = (forward_hidden_state, forward_memory_state)
        backward_state = (backward_hidden_state, backward_memory_state)
      else:
        forward_state = None
        backward_state = None

      forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                             batch_lengths,
                                                             forward_state)
      backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                batch_lengths,
                                                                backward_state)
      if layer_index != 0:
        forward_output_sequence += forward_cache
        backward_output_sequence += backward_cache

      sequence_outputs.append(torch.cat([forward_output_sequence,
                                         backward_output_sequence], -1))
      final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                           torch.cat([forward_state[1], backward_state[1]], -1)))

    stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
    final_hidden_states, final_memory_states = zip(*final_states)
    final_state_tuple: Tuple[torch.FloatTensor,
                             torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                   torch.cat(final_memory_states, 0))
    return stacked_sequence_outputs, final_state_tuple