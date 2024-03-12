import torch, os
from torch import nn, Tensor
import torch.nn.functional as F
from train_model import get_loss, decode, character_set

# determine the device to train on (this is mac inclusive too :D)
mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cuda = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
device = mps or cuda or "cpu"
device = "cpu"
print("Device is", device, "\n")

# ----- Global Parameters -----
CONTEXT_LENGTH = 8
BATCH_SIZE = 4
VOCAB_SIZE = len(character_set)
HIDDEN_LAYER = 256
NO_OF_HEADS = 4
NO_OF_BLOCKS = 3
DROPOUT_RATE = (
    0.2  # added right before the residual connection back to prevent overfitting
)
lr = 1e-3
training_rounds = 1000
# -----------------------------


# making a head of self-attention *squeals*
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, in_features):
        super().__init__()
        self.head_size = head_size
        # each block of self-attention will have 3 values: Query, Key, Value
        self.queries = nn.Linear(in_features=in_features, out_features=head_size)
        self.keys = nn.Linear(in_features=in_features, out_features=head_size)
        self.values = nn.Linear(in_features=in_features, out_features=head_size)

        # add dropout
        self.do = nn.Dropout(DROPOUT_RATE)

        # then create the mask
        self.register_buffer(
            "tril", torch.tril(torch.ones((CONTEXT_LENGTH, CONTEXT_LENGTH)))
        )

    def forward(self, x):
        # process the shape of the inputs so that the masked fill can occur even if the input size is less than CONTEXT LENGTH
        B, T, C = x.shape

        # generate the Q and K values for each input
        Q: Tensor = self.queries(x)
        K: Tensor = self.keys(x)
        # create the weights RMB TO SCALE!
        wei = Q @ K.transpose(-1, -2) * self.head_size**-0.5
        # **** this line with T is kind of sketchy welps, will have to dig further later ****
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, -1)
        wei = self.do(wei)
        # produce the V
        V = self.values(x)

        # determine the weighted result
        out = wei @ V
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        self.multi_headed_attention_block = nn.ModuleList(
            [
                SelfAttentionHead(head_size, head_size * number_of_heads)
                for _ in range(number_of_heads)
            ]
        )
        self.proj = nn.Linear(
            in_features=head_size * number_of_heads,
            out_features=head_size * number_of_heads,
        )

    def forward(self, x):
        # since the multi-attention head requires the splitting of the channels among the heads, this joins back the channels
        out = torch.cat(
            [attn_head(x) for attn_head in self.multi_headed_attention_block], dim=-1
        )
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """This block gives tokens the time to think"""

    def __init__(self, in_features, out_features):
        super().__init__()
        # the *4 here is just me following Karpathy
        self.ff = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features * 4),
            nn.ReLU(),
            nn.Linear(in_features=in_features * 4, out_features=out_features),
        )

    def forward(self, x):
        out = self.ff(x)
        return out


class Block(nn.Module):
    """Enables communication (multi-headed self-attention), then computation (feed-forward)"""

    def __init__(self, hidden_layer_size, no_of_heads):
        super().__init__()
        multi_attn_head_size = hidden_layer_size // no_of_heads
        self.multi_attn = MultiHeadedAttention(no_of_heads, multi_attn_head_size)
        self.ffwd = FeedForward(hidden_layer_size, hidden_layer_size)

        # adding layernorm
        self.ln1 = nn.LayerNorm(hidden_layer_size)
        self.ln2 = nn.LayerNorm(hidden_layer_size)

        # adding dropout, before connection back from the residual pathway
        self.do1 = nn.Dropout(DROPOUT_RATE)
        self.do2 = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        # more common to layernorm before applying attention and ffwd now
        multi_attention_r = x + self.do1(self.multi_attn(self.ln1(x)))
        ffwd_r = multi_attention_r + self.do2(self.ffwd(self.ln2(multi_attention_r)))
        return ffwd_r


# creation of the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_LAYER,
        )
        # besides the identity of the tokens, we also wish to embed the postition of the tokens
        self.position_embedding_table = nn.Embedding(
            num_embeddings=CONTEXT_LENGTH,
            embedding_dim=HIDDEN_LAYER,
        )
        self.blocks = nn.Sequential(
            *[
                Block(hidden_layer_size=HIDDEN_LAYER, no_of_heads=NO_OF_HEADS)
                for _ in range(NO_OF_BLOCKS)
            ]
        )
        self.ln = nn.LayerNorm(HIDDEN_LAYER)
        self.lm_head = nn.Linear(in_features=HIDDEN_LAYER, out_features=VOCAB_SIZE)

    def forward(self, x, targets=None):
        """Do a forward pass of the NN. Returns loss if targets are given"""
        # get the shape of the input so that the postitional outputs can be properly computed when the input size is less than 8
        B, T = x.shape

        token_logits: Tensor = self.token_embedding_table(x)
        postition_logits: Tensor = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        input_tokens = token_logits + postition_logits
        feed_forward = self.blocks(input_tokens)
        normalized = self.ln(feed_forward)
        logits: Tensor = self.lm_head(normalized)

        # if there are targets present, perform a calculation of the loss
        if targets != None:
            # B: number of batches, T: the context length, C: the number of outputs
            B, T, C = logits.shape
            # order the logits in the form: (minibatch,C) to enable cross_entropy calculation
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        return logits

    def generate(self, x, max_tokens=10):
        """Receives a (B, T) input token and creates tokens based on the input"""
        for _ in range(max_tokens):
            # get the last CONTEXT LENGTH number of tokens from the input sequence
            context = x[:, -CONTEXT_LENGTH:]
            logits = self(context)
            # we are only interested in the logits of the last character
            last_char_logits = logits[:, -1, :]  # (1, 88)
            # take the last embedding, since that is all that we are concerned about
            probs = F.softmax(
                last_char_logits, dim=1
            )  # the result of this is that all values along dim 1 will sum to 1
            next_char = torch.multinomial(probs, 1)
            x = torch.cat(
                [x, next_char], dim=1
            )  # addition is perfomed along dim=1 here because they all belong in the same batch
        return x


if __name__ == "__main__":
    if not os.path.exists("model.pth"):
        raise Exception("No previous configuration of the saved model exists !")

    m: Model = torch.load("model.pth")
    m.eval()

    get_loss(m, 500)
    print(
        decode(
            m.generate(
                torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=300
            )[0].tolist()
        )
    )
