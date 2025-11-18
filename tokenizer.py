class Tokenizer:
    def __init__(self):
        self.token_to_id = {'<start>' : 1, '<end_of_text>' : 2, ' ' : 3, '<unk>' : 4}
        self.id_to_token = {1 : '<start>', 2 : '<end_of_text>', 3 : ' ', 4: "<unk>"}
        self.vocab = set(['<start>', '<end_of_text>', '<unk>', ' '])
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
                pair = (text_characters[i-1], text_characters[i])
                if pair not in frequency:
                    frequency[pair] = 0
                frequency[pair] += 1
            if not frequency: break

            most_commmon_pair, occurence = max(frequency.items(), key = lambda item: item[1])
            if occurence > 1:
                new_token = ''.join(most_commmon_pair)
                self.vocab.add(new_token)
                id = len(self.vocab)
                self.token_to_id[new_token] = id
                self.id_to_token[id] = new_token
                self.bp_merges[most_commmon_pair] = id # id here is the rank for our pair
                # Merge those tokens inside the text_characters
                new_text_characters = []

                index = 0
                while (index < len(text_characters)):
                    if (text_characters[index] == most_commmon_pair[0] and index < len(text_characters) - 1 and text_characters[index+1] == most_commmon_pair[1]):
                        new_text_characters.append(new_token)
                        index += 2 # Skip the next character
                    else:
                        new_text_characters.append(text_characters[index])
                        index += 1
                text_characters = new_text_characters
            else:
                break

    def encode(self, text: str, add_special_tokens = True):
        assert self.bp_merges, "You must train your tokenizer first!"
        tokens = list(text)
        # Merge based on the rank that a pair was constructed
        while True:
            best_rank = float("inf")
            best_pair = None
            candidate_index = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.bp_merges.get(pair, None)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
                    candidate_index = i
            if best_pair is None: break
            # Merge pair with lowest rank
            tokens[candidate_index] = ''.join(best_pair)
            del tokens[candidate_index + 1]

        ids = [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
        if add_special_tokens:
            ids = [self.token_to_id["<start>"]] + ids + [self.token_to_id['<end_of_text>']]

        return ids

    def decode(self, inputs):
        string =  "".join([self.id_to_token.get(id, self.id_to_token[4]) for id in inputs])
        return string