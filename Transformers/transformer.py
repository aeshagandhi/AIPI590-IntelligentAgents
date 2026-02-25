# Based on "Attention Is All You Need" - Vaswani et al., 2017
# https://arxiv.org/abs/1706.03762

import numpy as np


def softmax(x, axis=-1):
    # convert vector of raw scores into probabilities 
    # subtract max for numerical stability
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class LayerNorm:
    # re center and rescale values at each position to have mean 0 and variance 1
    # apply learnable parameters gamma (for scaling) and beta (for shifting)
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps # small number to avoid divide by zero
        self.gamma = np.ones(d_model) # scale param
        self.beta = np.zeros(d_model) # shift param

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class PositionalEncoding:
    # no sense of word order in self attention, sees all tokens at once
    # add sin and cosine to each position so model can learn to tell relative positions
    def __init__(self, d_model, max_len=5000):
        pe = np.zeros((max_len, d_model))
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(positions * div_term) # even indices get sine
        pe[:, 1::2] = np.cos(positions * div_term) # odd indices get cosine
        self.pe = pe

    def forward(self, x):
        # fixed process of adding positional embedding to encodings, not learned
        seq_len = x.shape[1]
        return x + self.pe[np.newaxis, :seq_len, :]


class MultiHeadAttention:
    # splits Q, K, V into multiple heads, runs attention in parallel, then concatenates
    # learn for every token which other tokens to attend to and how much for each head
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        scale = np.sqrt(2.0 / (d_model + d_model))
        # initialize learnable weight matrices as normal to avoid exploding from early on
        self.W_Q = np.random.normal(0, scale, (d_model, d_model))
        self.W_K = np.random.normal(0, scale, (d_model, d_model))
        self.W_V = np.random.normal(0, scale, (d_model, d_model))
        self.W_O = np.random.normal(0, scale, (d_model, d_model))
        self.b_Q = np.zeros(d_model)
        self.b_K = np.zeros(d_model)
        self.b_V = np.zeros(d_model)
        self.b_O = np.zeros(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # scores shape: (batch, heads, seq_q, seq_k)
        # dot product of Q and all K, scaled by sqrt(d_k) to keep gradients table, apply softmax to get weights
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attn_weights = softmax(scores, axis=-1)
        # multiply weights with V to get weighted sum of values
        return np.matmul(attn_weights, V)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project inputs to Q, K, V
        Q = query @ self.W_Q + self.b_Q
        K = key   @ self.W_K + self.b_K
        V = value @ self.W_V + self.b_V

        # reshape into (batch, heads, seq, d_k) for parallel attention
        def split_heads(x, seq_len):
            x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
            return x.transpose(0, 2, 1, 3)

        Q = split_heads(Q, seq_q)
        K = split_heads(K, seq_k)
        V = split_heads(V, seq_k)

        attn_out = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project back to d_model
        attn_out = attn_out.transpose(0, 2, 1, 3)
        attn_out = attn_out.reshape(batch_size, seq_q, self.d_model)

        return attn_out @ self.W_O + self.b_O


class FeedForward:
    # simple 2 layer NN: linear --> ReLU --> linear
    # after attention, pass each token's representation through two layer neural network independently 
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.normal(0, np.sqrt(2.0 / d_model), (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.normal(0, np.sqrt(2.0 / d_ff), (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


class EncoderBlock:
    # full encoder layer with self attention, add input back as residual connection, normalize
    # then feed forward, add input back, normalize again
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # pass x as query, key, value for self attention, where every token is looking at every other token
        x = self.norm1.forward(x + self.self_attn.forward(x, x, x, mask=src_mask))
        x = self.norm2.forward(x + self.ffn.forward(x))
        return x


class Encoder:
    # Stack of N encoder blocks
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer.forward(x, src_mask)
        return x


class DecoderBlock:
    # decoder layer with masked self attention, cross attention, feed forward NN
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads) # token can only attend to previous tokens, can't look ahead
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # tokens attend to encoder output
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.norm1.forward(x + self.self_attn.forward(x, x, x, mask=tgt_mask)) # target mask is the causal mask for future tokens
        # query comes from x but keys and values come from the encoder output
        # each target token looks at all source tokens and decide which are most relevant
        x = self.norm2.forward(x + self.cross_attn.forward(x, enc_output, enc_output, mask=src_mask)) 
        x = self.norm3.forward(x + self.ffn.forward(x))
        return x


class Decoder:
    # Stack of N decoder blocks
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer.forward(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer:
    # puts everything together
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000):
        self.d_model = d_model
        scale = np.sqrt(1.0 / d_model)

        # map tokens to integer index first in word embeddings
        # this is a word embedding table for source and target, lookup tables where each row is learned vector for one token
        self.src_embedding = np.random.normal(0, scale, (src_vocab_size, d_model))
        self.tgt_embedding = np.random.normal(0, scale, (tgt_vocab_size, d_model))
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        self.output_projection = np.random.normal(0, scale, (d_model, tgt_vocab_size))
        self.output_bias = np.zeros(tgt_vocab_size)

    def make_padding_mask(self, token_ids, pad_idx=0):
        # 1 for real tokens, 0 for padding — shape (batch, 1, 1, seq_len)
        mask = (token_ids != pad_idx).astype(np.float32)
        return mask[:, np.newaxis, np.newaxis, :]

    def make_causal_mask(self, seq_len):
        # lower-triangular matrix so each position can only attend to past positions
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return mask[np.newaxis, np.newaxis, :, :]

    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        
        # make masks
        src_mask = self.make_padding_mask(src, src_pad_idx)
        # Combine padding mask and causal mask for the target
        tgt_mask = self.make_padding_mask(tgt, tgt_pad_idx) * self.make_causal_mask(tgt.shape[1])

        # embed and add positional encoding, scaled by sqrt(d_model)
        src_emb = self.pos_encoding.forward(self.src_embedding[src] * np.sqrt(self.d_model))
        tgt_emb = self.pos_encoding.forward(self.tgt_embedding[tgt] * np.sqrt(self.d_model))


        # run encoder on source
        enc_output = self.encoder.forward(src_emb, src_mask)
        # run decoder on target and encoder output
        dec_output = self.decoder.forward(tgt_emb, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

        # Project decoder output to vocabulary logits
        return dec_output @ self.output_projection + self.output_bias


if __name__ == "__main__":
    np.random.seed(10)

    # model = Transformer(
    #     src_vocab_size=100,
    #     tgt_vocab_size=100,
    #     d_model=32,
    #     num_heads=4,
    #     num_layers=2,
    #     d_ff=64
    # )

    # src = np.array([[1, 2, 3, 4, 5]])
    # tgt = np.array([[1, 7, 3]])

    # logits = model.forward(src, tgt)

    # print("Source tokens:   ", src)
    # print("Target tokens:   ", tgt)
    # print("Output shape:    ", logits.shape)
    # print("Predicted tokens:", np.argmax(logits, axis=-1))

    # Build a simple vocabulary: word -> index
    vocab = {
        "<pad>": 0, "<sos>": 1, "<eos>": 2,
        "the": 3, "cat": 4, "sat": 5, "on": 6, "mat": 7,
        "a": 8, "dog": 9, "ran": 10, "fast": 11
    }
    idx_to_word = {v: k for k, v in vocab.items()}

    def encode(sentence):
        # wrap sentence with start and end tokens
        tokens = [vocab["<sos>"]] + [vocab[w] for w in sentence.split()] + [vocab["<eos>"]]
        return np.array([tokens])  # add batch dimension

    def decode(indices):
        # map predicted indices back to words
        return " ".join(idx_to_word[i] for i in indices[0])

    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64
    )

    # pass source and target sentence through model
    src = encode("the cat sat on the mat")
    tgt = encode("a dog ran fast")

    logits = model.forward(src, tgt)
    # take argmax of logits to get highest scoring token at each position and decode back to words
    predicted_indices = np.argmax(logits, axis=-1)

    print("Source:           ", "the cat sat on the mat")
    print("Source tokens:    ", src)
    print("Target:           ", "a dog ran fast")
    print("Target tokens:    ", tgt)
    print("Output shape:     ", logits.shape)
    print("Predicted tokens: ", predicted_indices)
    print("Predicted words:  ", decode(predicted_indices))
    
    # predictions are nonsense because model randomly initialized but correct shapes and working forward pass