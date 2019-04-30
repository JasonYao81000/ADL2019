"""
An LSTM with Recurrent Dropout, a hidden_state which is projected and
clipping on both the hidden state and the memory state of the LSTM.
"""

from typing import Optional, Tuple, List
import itertools
import torch
from torch.autograd import Variable


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



class LstmCellWithProjection(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 go_forward: bool = True,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None) -> None:
        super(LstmCellWithProjection, self).__init__()
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
