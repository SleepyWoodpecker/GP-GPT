# making a tokenizer, based on GPT-4's regex pattern
import regex as re
import pickle
import os

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, text, saved_config_folder=None):
        # load trained tokenizer if there is a config path for the merge and reverse merge provided
        if saved_config_folder and os.path.exists(saved_config_folder):
            with open(f"{saved_config_folder}/merges.pkl", "rb") as file:
                self.merges = pickle.load(file)

            with open(f"{saved_config_folder}/reverse_merges.pkl", "rb") as file:
                self.reverse_merges = pickle.load(file)

        elif saved_config_folder == None:
            self.merges = {}
            self.reverse_merges = {}
            self.vocab = {idx: bytes([idx]) for idx in range(256)}

            chunks = re.findall(GPT4_SPLIT_PATTERN, text)
            self.encoded_chunks = []
            for chunk in chunks:
                encoded_chunk = []
                for character in chunk:
                    encoded_chunk += list((character).encode("utf-8"))

                self.encoded_chunks.append(encoded_chunk)

        else:
            raise Exception("Tokenizer config file cannot be found!")

    def _get_ranks(self, chunks):
        ranks = {}
        for chunk in chunks:
            for pair in zip(chunk, chunk[1:]):
                ranks[pair] = ranks.get(pair, 0) + 1

        return sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    def _merge(self, chunks, target_pair, new_token):
        merged_sequence = []
        for chunk in chunks:
            new_chunk = []
            skip_next = False
            for pair in zip(chunk, chunk[1:]):
                if skip_next:
                    skip_next = False
                    continue

                if pair == target_pair:
                    # print("HI")
                    new_chunk.append(new_token)
                    skip_next = True

                else:
                    new_chunk.append(pair[0])

            new_chunk += [chunk[-1]] if not skip_next else []
            merged_sequence.append(new_chunk)

        return merged_sequence

    def train(self, vocab_size, verbose=False):
        if vocab_size < 256:
            raise Exception("No training is going to happen!")

        next_token = 256

        while next_token <= vocab_size:
            most_frequent_pair_and_count = self._get_ranks(self.encoded_chunks)[0]

            # halt training if pairs appear no more than once
            if most_frequent_pair_and_count[1] < 2:
                print("Stopping training, no more pairs to compress")
                break

            self.encoded_chunks = self._merge(
                self.encoded_chunks, most_frequent_pair_and_count[0], next_token
            )
            self.merges[most_frequent_pair_and_count[0]] = next_token

            if verbose:
                print(f"Merged {most_frequent_pair_and_count[0]} into {next_token}")

            next_token += 1

        new_to_original = {
            new_token: original_token
            for original_token, new_token in self.merges.items()
        }
        self.reverse_merges = list(reversed(new_to_original.items()))

        # record the trained dictionaries
        with open("trained_config/merges.pkl", "wb") as file:
            pickle.dump(self.merges, file)

        with open("trained_config/reverse_merges.pkl", "wb") as file:
            pickle.dump(self.reverse_merges, file)

    def encode(self, input_text):
        if not self.merges or not self.reverse_merges:
            raise Exception("Train the tokenizer first!")

        input_text_chunks = re.findall(GPT4_SPLIT_PATTERN, input_text)
        input_chunks = []

        for text_chunk in input_text_chunks:
            input_chunks.append(list(text_chunk.encode("utf-8")))

        for target, new_token in self.merges.items():
            input_chunks = self._merge(input_chunks, target, new_token)

        return input_chunks

    def decode(self, input_tokens):
        if not self.merges or not self.reverse_merges:
            raise Exception("Train the tokenizer first!")

        # unmerge the tokens
        for new_token, original_tokens in self.reverse_merges:
            new_sequence = []
            for chunk in input_tokens:
                new_chunk = []
                for token in chunk:
                    if token == new_token:
                        new_chunk.extend(original_tokens)
                    else:
                        new_chunk.append(token)

                new_sequence.append(new_chunk)
                input_tokens = new_sequence

        return (b"".join([bytes(i) for i in input_tokens])).decode("utf-8")


if __name__ == "__main__":
    txt = open("../data/essays2.txt", "r").read()

    t = Tokenizer(txt, "trained_config")

    r = t.encode("Economic powerhouse, let them rise. It is time we made a move")
    print(r)
    print(t.decode(r))
