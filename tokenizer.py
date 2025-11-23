import pickle
import os


class TokenizerV1:
    def __init__(self):
        self.token_to_id = {"<start>": 1, "<end_of_text>": 2, " ": 3, "<unk>": 4}
        self.id_to_token = {1: "<start>", 2: "<end_of_text>", 3: " ", 4: "<unk>"}
        self.vocab = set(["<start>", "<end_of_text>", "<unk>", " "])
        self.bp_merges = {}

    def train(self, text):
        assert len(text) > 0, "You must input a non-empty text"
        text_characters = []
        for char in text:
            if char not in self.vocab:
                self.vocab.add(char)
                id = len(self.vocab)
                self.token_to_id[char] = id
                self.id_to_token[id] = char
            text_characters.append(char)

        while True:
            frequency = {}
            # Contruct frequency from the text_characters
            for i in range(1, len(text_characters)):
                pair = (text_characters[i - 1], text_characters[i])
                if pair not in frequency:
                    frequency[pair] = 0
                frequency[pair] += 1
            if not frequency:
                break

            most_commmon_pair, occurence = max(
                frequency.items(), key=lambda item: item[1]
            )
            if occurence > 1:
                new_token = "".join(most_commmon_pair)
                self.vocab.add(new_token)
                id = len(self.vocab)
                self.token_to_id[new_token] = id
                self.id_to_token[id] = new_token
                self.bp_merges[most_commmon_pair] = (
                    id  # id here is the rank for our pair
                )
                # Merge those tokens inside the text_characters
                new_text_characters = []

                index = 0
                while index < len(text_characters):
                    if (
                        text_characters[index] == most_commmon_pair[0]
                        and index < len(text_characters) - 1
                        and text_characters[index + 1] == most_commmon_pair[1]
                    ):
                        new_text_characters.append(new_token)
                        index += 2  # Skip the next character
                    else:
                        new_text_characters.append(text_characters[index])
                        index += 1
                text_characters = new_text_characters
            else:
                break

    def encode(self, text: str, add_special_tokens=True):
        assert self.bp_merges, "You must train your tokenizer first!"
        tokens = list(text)
        # Merge based on the rank that a pair was constructed
        while True:
            best_rank = float("inf")
            best_pair = None
            candidate_index = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bp_merges.get(pair, None)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
                    candidate_index = i
            if best_pair is None:
                break
            # Merge pair with lowest rank
            tokens[candidate_index] = "".join(best_pair)
            del tokens[candidate_index + 1]

        ids = [
            self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in tokens
        ]
        if add_special_tokens:
            ids = (
                [self.token_to_id["<start>"]]
                + ids
                + [self.token_to_id["<end_of_text>"]]
            )

        return ids

    def decode(self, inputs):
        string = "".join(
            [self.id_to_token.get(id, self.id_to_token[4]) for id in inputs]
        )
        return string

    # Save tokenizer da

    def save_tokenizer(self):

        tokenizer_path = "tokenizer"
        if not os.path.isdir(tokenizer_path):
            os.mkdir(tokenizer_path)

        with open(f"{tokenizer_path}/token_to_id.pkl", "wb") as f:
            pickle.dump(f, self.token_to_id)

        with open(f"{tokenizer_path}/id_to_token.pkl", "wb") as f:
            pickle.dump(f, self.id_to_token)

        with open(f"{tokenizer_path}/bp_merges.pkl", "wb") as f:
            pickle.dump(f, self.bp_merges)

    def load_tokenizer(self):

        tokenizer_path = "tokenizer"
        with open(f"{tokenizer_path}/token_to_id.pkl", "wb") as f:
            self.token_to_id = pickle.load(f)

        with open(f"{tokenizer_path}/id_to_token.pkl", "wb") as f:
            self.id_to_token = pickle.load(f)

        with open(f"{tokenizer_path}/bp_merges.pkl", "wb") as f:
            self.bp_merges = pickle.load(f)


from collections import Counter, deque
from functools import lru_cache
import json


class TokenizerV2:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        # For the official OpenAI GPT-2 merges, use a rank dict:
        #  of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|start|>", "<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text)) if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text, allowed_special=None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.

        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.

        Returns:
            List of token IDs.
        """
        import re

        token_ids = []

        # If special token handling is enabled
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = (
                "("
                + "|".join(
                    re.escape(tok)
                    for tok in sorted(allowed_special, key=len, reverse=True)
                )
                + ")"
            )

            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index : match.start()]
                token_ids.extend(
                    self.encode(prefix, allowed_special=None)
                )  # Encode prefix without special handling

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(
                        f"Special token {special_token} not found in vocabulary."
                    )
                last_index = match.end()

            text = text[last_index:]  # Remaining part to process normally

            # Check if any disallowed special tokens are in the remainder
            disallowed = [
                tok
                for tok in self.inverse_vocab
                if tok.startswith("<|")
                and tok.endswith("|>")
                and tok in text
                and tok not in allowed_special
            ]
            if disallowed:
                raise ValueError(
                    f"Disallowed special tokens encountered in text: {disallowed}"
                )

        # If no special tokens, or remaining text after special token split:
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "  # Add space if not present before a newline
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [
                {"pair": list(pair), "new_id": new_id}
                for pair, new_id in self.bpe_merges.items()
            ]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Uncomment for educational purposes:
                        # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of that pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
