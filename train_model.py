import torch, os
from torch import nn, Tensor
import torch.nn.functional as F
from tokenizer.tokenizer import Tokenizer

# Data processing and extraction
data = open("data/essays.txt", "r", encoding="utf-8").read()

# load the tokenizer from config
trained_tokenizer = Tokenizer(saved_config_folder="tokenizer/trained_config")

# ----- Global Parameters -----
CONTEXT_LENGTH = 256
BATCH_SIZE = 64
VOCAB_SIZE = 256 + len(trained_tokenizer.merges)
HIDDEN_LAYER = 4096
NO_OF_HEADS = 64
NO_OF_BLOCKS = 8
DROPOUT_RATE = (
    0.3  # added right before the residual connection back to prevent overfitting
)
lr = 3e-4
training_rounds = 7000
# -----------------------------

# determine the device to train on (this is mac inclusive too :D)
mps = "mps" if torch.backends.mps.is_available() else None
cuda = "cuda" if torch.cuda.is_available() else None
device = mps or cuda or "cpu"
print("Device is", device, "\n")


# preparing the training and testing splits
n = int(len(data) * 0.9)
train = torch.tensor(trained_tokenizer.encode(data[:n]))
val = torch.tensor(trained_tokenizer.encode(data[n:]))


def get_batch(data_set):
    """Generate a batch of learning examples and targets from the data set"""
    # get a BATCH_SIZEW number of random integers
    indexes = torch.randint(len(data_set) - CONTEXT_LENGTH, (BATCH_SIZE,))
    # Get BATCH_SIZE length of inputs to mask later.
    Xb = torch.stack([data_set[i : i + CONTEXT_LENGTH] for i in indexes], dim=0)
    # Get the corresponding targets
    Yb = torch.stack([data_set[i + 1 : i + CONTEXT_LENGTH + 1] for i in indexes], dim=0)
    Xb, Yb = Xb.to(device), Yb.to(device)
    return Xb, Yb


# give the model a way to evaluate its loss
@torch.no_grad
def get_loss(model, loss_evals):
    """Evaluate the mean loss for both training and validation batches"""
    model.eval()
    losses_tr, losses_val = [], []
    for _ in range(loss_evals):
        Xtr, Ytr = get_batch(train)
        Xval, Yval = get_batch(val)

        logits_tr, loss_tr = model(Xtr, Ytr)
        losses_tr.append(loss_tr.item())

        logits_val, loss_val = model(Xval, Yval)
        losses_val.append(loss_val.item())

    tr_avg_loss = torch.tensor(losses_tr, dtype=torch.float).mean()
    val_avg_loss = torch.tensor(losses_val, dtype=torch.float).mean()
    print(f"Train loss: {tr_avg_loss:.5f} | Val loss: {val_avg_loss:.5f}")
    model.train()


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


def save_model_weights(model: Model, path):
    torch.save(model, path)


if __name__ == "__main__":
    # clear the old model file
    if os.path.exists("model.pth"):
        os.remove("model.pth")

    model = Model()
    m = model.to(device)
    optimizer = torch.optim.AdamW(params=m.parameters(), lr=lr)
    m.train()

    for i in range(training_rounds):
        Xtr, Ytr = get_batch(train)
        # forward pass
        logits, loss = m(Xtr, Ytr)

        # print the loss at regular intervals
        if i % 1000 == 0:
            get_loss(m, 200)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_model_weights(m, "model.pth")

    m.eval()
    print("Training completed...")
    get_loss(m, 500)
    print(
        trained_tokenizer.decode(
            m.generate(
                torch.zeros((1, 1), dtype=torch.long, device=device), max_tokens=300
            )[0].tolist()
        )
    )
