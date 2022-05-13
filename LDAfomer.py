import math
import torch
import torch.nn as nn

# Data Embedding

class valueEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear, value_sqrt):
        super(valueEmbedding, self).__init__()

        self.value_linear = value_linear
        self.value_sqrt = value_sqrt
        self.d_model = d_model
        self.inputLinear = nn.Linear(d_input, d_model)

    def forward(self, x):
        if self.value_linear:
            x = self.inputLinear(x)
        if self.value_sqrt:
            x = x * math.sqrt(self.d_model)
        return x

class positionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positionalEmbedding, self).__init__()
        
        pos_emb = torch.zeros(max_len, d_model).float()
        pos_emb.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        return self.pos_emb[:, :x.size(1)]

class dataEmbedding(nn.Module):
    def __init__(self, d_input, d_model, value_linear, value_sqrt, posi_emb, input_dropout=0.05):
        super(dataEmbedding, self).__init__()
        
        self.posi_emb = posi_emb
        self.value_embedding = valueEmbedding(d_input=d_input, d_model=d_model, value_linear=value_linear, value_sqrt=value_sqrt)
        self.positional_embedding = positionalEmbedding(d_model=d_model)
        self.inputDropout = nn.Dropout(input_dropout)

    def forward(self, x):
        if self.posi_emb:
            x = self.value_embedding(x) + self.positional_embedding(x)
        else:
            x = self.value_embedding(x)
        return self.inputDropout(x)

# Scaled Dot-Product Attention

class scaledDotProductAttention(nn.Module):
    def __init__(self):
        super(scaledDotProductAttention, self).__init__()
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape  # (batch_size, seq_len, n_heads, d_model)
        _, S, _, D = values.shape

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) # matmul
        scale = 1./math.sqrt(E) # scale
        # no mask
        A = torch.softmax(scale * scores, dim=-1)   # softmax
        V = torch.einsum("bhls, bshd->blhd", A, values) # matmul
        
        return V.contiguous()

# Attention Layer

class attentionLayer(nn.Module):
    def __init__(self, scaled_dot_product_attention, d_model, n_heads):
        super(attentionLayer, self).__init__()

        d_values = d_model // n_heads

        self.scaled_dot_product_attention = scaled_dot_product_attention
        self.query_linear = nn.Linear(d_model, d_values * n_heads)
        self.key_linear = nn.Linear(d_model, d_values * n_heads)
        self.value_linear = nn.Linear(d_model, d_values * n_heads)
        self.out_linear = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_linear(queries).view(B, L, H, -1)
        keys = self.key_linear(keys).view(B, S, H, -1)
        values = self.value_linear(values).view(B, S, H, -1)

        out = self.scaled_dot_product_attention(queries, keys, values)
        out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_linear(out)


# Encoder Layer

class encoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, add, norm, ff, encoder_dropout=0.05):
        super(encoderLayer, self).__init__()
        
        d_ff = int(d_ff * d_model)
        self.attention_layer = attention_layer

        self.add = add
        self.norm = norm
        self.ff = ff
        
        # point wise feed forward network
        self.feedForward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(encoder_dropout)

        self.norm1 = nn.LayerNorm(d_model)    # (batch_size, seq_len, d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        new_x = self.attention_layer(x, x, x)
        
        if self.add:
            if self.norm:
                out1 = self.norm1(x + self.dropout(new_x))
                out2 = self.norm2(out1 + self.dropout(self.feedForward(out1)))
            else:
                out1 = x + self.dropout(new_x)
                out2 = out1 +  self.dropout(self.feedForward(out1))
        else:
            if self.norm:
                out1 = self.norm1(self.dropout(new_x))
                out2 = self.norm2(self.dropout(self.feedForward(out1)))
            else:
                out1 = self.dropout(new_x)
                out2 = self.dropout(self.feedForward(out1))
        
        if self.ff:
            return out2
        else:
            return out1

# Encoder

class encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        # x [B, L, D]
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

# LDAformer

class LDAformer(nn.Module):
    def __init__(self, seq_len, d_input, d_model=12, n_heads=1, e_layers=4, d_ff=0.5, pos_emb=False, value_linear=True, value_sqrt=False, add=True, norm=True, ff=True, dropout=0.05):
        super(LDAformer, self).__init__()

        # Encoding
        self.data_embedding = dataEmbedding(d_input, d_model, posi_emb=pos_emb, value_linear=value_linear, value_sqrt=value_sqrt, input_dropout=dropout)
        # Encoder
        self.encoder = encoder(
            [
                encoderLayer(
                    attentionLayer(scaledDotProductAttention(), d_model, n_heads),
                    d_model,
                    d_ff,
                    encoder_dropout=dropout,
                    add=add,
                    norm=norm,
                    ff=ff
                ) for l in range(e_layers)
            ]
        )
        self.prediction = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_embedding = self.data_embedding(x)
        enc_out = self.encoder(x_embedding)
        
        return self.prediction(enc_out).squeeze(-1)


class CNN(nn.Module):
    def __init__(self, seq_len, d_input, d_model, dropout_rate, out_channel1, out_channel2, kernel_size1, kernel_size2, maxpool_size, emb=True):
        super(CNN, self).__init__()

        self.data_embedding = dataEmbedding(d_input, d_model, posi_emb=False, value_linear=True, value_sqrt=False, input_dropout=dropout_rate)

        if emb:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=out_channel1, kernel_size=(kernel_size1, d_model)),
                nn.MaxPool2d(kernel_size=(maxpool_size, 1), stride=(maxpool_size, 1))
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=out_channel1, kernel_size=(kernel_size1, d_input)),
                nn.MaxPool2d(kernel_size=(maxpool_size, 1), stride=(maxpool_size, 1))
            )
        self.emb = emb
        tmp_dim = (seq_len - kernel_size1 + 1) // maxpool_size

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=(kernel_size2, 1)),
            nn.MaxPool2d(kernel_size=(maxpool_size, 1), stride=(maxpool_size, 1))
        )
        tmp_dim = (tmp_dim - kernel_size2 + 1) // maxpool_size

        self.prediction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channel2 * tmp_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        if self.emb:
            x = self.data_embedding(x)

        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.squeeze(-1)
        x = self.prediction(x).squeeze(-1)

        return x

class NN(nn.Module):
    def __init__(self, seq_len, d_input, d_model, dropout_rate, emb=True):
        super(NN, self).__init__()

        self.data_embedding = dataEmbedding(d_input, d_model, posi_emb=False, value_linear=True, value_sqrt=False, input_dropout=dropout_rate)

        if emb:
            self.nnLayers1 = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.ReLU()
            )
        else:
            self.nnLayers1 = nn.Sequential(
                nn.Linear(d_input, 1),
                nn.ReLU()
            )
        self.emb = emb

        self.nnLayers2 = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            # nn.Linear(seq_len // 2, seq_len // 2),
            nn.Linear(seq_len // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        if self.emb:
            x = self.data_embedding(x)
        x = self.nnLayers1(x).squeeze(-1)
        x = self.nnLayers2(x).squeeze(-1)

        return x
