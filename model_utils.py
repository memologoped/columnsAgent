import torch
import torch.nn as nn


# transformer attention head with in_size equal to transformer input size and
# hidden size equal to in_size/h (number of attention heads)
class TAttention(nn.Module):
    def __init__(self, in_size, h):
        super(TAttention, self).__init__()
        self.att_size = int(in_size / h)
        # create the Query, Key and Value parts of the attention head
        self.Q = nn.Linear(in_size, in_size, bias=False)
        self.K = nn.Linear(in_size, in_size, bias=False)
        self.V = nn.Linear(in_size, in_size, bias=False)
        # att block linear output layer
        self.fc = nn.Linear(in_size, in_size, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.h = h
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        # scaling factor for the attention scores
        scale = torch.sqrt(torch.FloatTensor([self.h])).item()
        batch_size = q.size(0)
        # apply the linear transform to the query, key and value and reshape
        # the result into h attention heads
        Q = self.Q(q).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)
        K = self.K(k).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)
        V = self.V(v).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)
        # multiply and scale q and v to get the attention scores
        self.alpha = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # apply mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
            self.alpha = self.alpha.masked_fill(mask == 0, -1e9)
        # apply softmax to the (masked)attention scores and apply them to V
        self.alpha = self.softmax(self.alpha)
        att_applied = torch.matmul(self.dropout(self.alpha), V)

        # reshape the attention heads and finally pass them through a fully
        # connected layer
        att = att_applied.transpose(1, 2).reshape(batch_size, -1,
                                                  self.att_size * self.h)
        output = self.fc(att)
        return output


# the linear layer block (feed forward) of the transformer applies 2 linear
# layers with relu activation in between
class TFeedForward(nn.Module):
    def __init__(self, in_size, fc_size):
        super(TFeedForward, self).__init__()
        self.ff_1 = nn.Linear(in_size, fc_size)
        self.ff_2 = nn.Linear(fc_size, in_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.ff_2(self.relu(self.ff_1(input)))

        return output


# single encoder transformer cell with h attention heads fully connected layer
# block and residual connections
class TEncoderCell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(TEncoderCell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # the encoder applies an attention block and a linear block, with
        # a residual connection and layer normalisation after each block
        self.att_heads = TAttention(in_size, h)

        self.ff = TFeedForward(in_size, fc_size)

        self.norm_att = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input, mask=None):
        # attention block, for encoder q = k = v = input
        att = self.att_heads(input, input, input, mask)
        norm_att = self.norm_att(self.dropout(att) + input)

        lin = self.ff(norm_att)
        out = self.norm_ff(self.dropout(lin) + norm_att)
        return out


class TDecoderCell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(TDecoderCell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # the decoder applies two attention blocks, followed by linear block
        # with a residual connection and layer normalisation after each block
        self.att_one = TAttention(in_size, h)
        self.att_two = TAttention(in_size, h)

        self.ff = TFeedForward(in_size, fc_size)

        self.norm_att_one = nn.LayerNorm(in_size)
        self.norm_att_two = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)

        self.dropout = nn.Dropout(0.1)

    # the decoder has different masks for the first and second attention layer.
    # the encoder input is optional, if not provided the decoder acts
    # basically as an encoder with two attention layers.
    def forward(self, input, dec_mask=None, enc_mask=None,
                enc_input=None):
        # in the first layer q = k = v = input.
        att = self.att_one(input, input, input, dec_mask)
        norm_att = self.norm_att_one(self.dropout(att) + input)

        # in the second att block, q is the intermediate dec output, k and v
        # are the final states of the encoder.
        if enc_input is None:
            # if no enc_input is provided default to using the intermediate
            # dec output and the decoder mask
            enc_input = norm_att
            enc_mask = dec_mask
        att_2 = self.att_two(norm_att, enc_input, enc_input, enc_mask)
        norm_att_2 = self.norm_att_two(self.dropout(att_2) + norm_att)

        lin = self.ff(norm_att_2)
        out = self.norm_ff(self.dropout(lin) + norm_att_2)

        return out


# the transformer encoder capable of stacking multiple transformer cells.
class TEncoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(TEncoder, self).__init__()
        # create one or more multi-head attention layers
        self.tf_stack = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.tf_stack.append(TEncoderCell(in_size, fc_size, h))

    def forward(self, input, mask=None):
        # apply the (stacked) transformer
        for tf in self.tf_stack:
            input = tf(self.dropout(input), mask)
        return input


# the transformer decoder capable of stacking multiple transformer cells.
class TDecoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(TDecoder, self).__init__()
        # create one or more multi-head attention layers
        self.tf_stack = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.tf_stack.append(TDecoderCell(in_size, fc_size, h))

    def forward(self, input, dec_mask=None, enc_mask=None, enc_input=None):
        # apply the (stacked) transformer
        for tf in self.tf_stack:
            input = tf(self.dropout(input), dec_mask, enc_mask, enc_input)
        return input
