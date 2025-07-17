import numpy as np


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(query, key, value):
    """
    Compute 'Scaled Dot Product Attention'
    Args:
        query: (batch_size, sequence_length, embedding_dim)
        key: (batch_size, sequence_length, embedding_dim)
        value: (batch_size, sequence_length, embedding_dim)
    Returns:
        context: (batch_size, sequence_length, embedding_dim)
        weights: (batch_size, sequence_length, sequence_length)
    """
    # Reshape for proper matrix multiplication
    Q = query.reshape(-1, query.shape[1], query.shape[2])
    K = key.reshape(-1, key.shape[1], key.shape[2])
    V = value.reshape(-1, value.shape[1], value.shape[2])

    # Scale dot product attention
    d_k = query.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Get attention weights
    weights = softmax(scores, axis=-1)

    # Calculate weighted sum
    context = np.matmul(weights, V)

    return context.reshape(query.shape), weights


class MultiHeadAttention:
    def __init__(self, num_heads=8, embedding_dim=128):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def split_heads(self, x):
        """Split the last dimension into (heads, depth)"""
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        x = x.reshape(batch_size, sequence_length, self.num_heads, -1)
        return x.transpose((0, 2, 1, 3))

    def combine_heads(self, x):
        """Combine heads dimension"""
        batch_size = x.shape[0]
        sequence_length = x.shape[2]
        x = x.transpose((0, 2, 1, 3)).reshape(batch_size, sequence_length, -1)
        return x

    def forward(self, query, key, value):
        # Split heads
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)

        # Apply attention
        context, weights = scaled_dot_product_attention(q, k, v)

        # Combine heads
        return self.combine_heads(context), weights


class FeedForwardNetwork:
    def __init__(self, embedding_dim=128, hidden_dim=256):
        self.W1 = np.random.randn(embedding_dim, hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embedding_dim)

    def forward(self, x):
        return np.matmul(np.maximum(np.matmul(x, self.W1), 0), self.W2)


class TransformerLayer:

    self_attn: MultiHeadAttention
    feed_forward: FeedForwardNetwork

    def __init__(self, num_heads=8, embedding_dim=128, dropout=0.1):
        self.self_attn = MultiHeadAttention(num_heads, embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim)
        self.dropout = dropout

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn.forward(x, x, x)

        # Feed-forward network
        ff_output = self.feed_forward.forward(attn_output + x)

        return ff_output


# Example usage
np.random.seed(42)
batch_size = 32
sequence_length = 50
embedding_dim = 128

transformer_layer = TransformerLayer(num_heads=8, embedding_dim=embedding_dim)

input_data = np.random.randn(batch_size, sequence_length, embedding_dim)
output = transformer_layer.forward(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
