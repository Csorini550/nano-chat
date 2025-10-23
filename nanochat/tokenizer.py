import os  # import os to interact with filesystem paths and directories
import copy  # import copy so we can duplicate nested data structures safely
from functools import lru_cache  # bring in lru_cache to memoize results of helper functions

SPECIAL_TOKENS = [  # define the ordered list of reserved tokens used during conversation encoding
    "<|bos|>",  # represent the beginning-of-sequence delimiter inserted at document start
    "<|user_start|>",  # represent the marker that precedes user message content
    "<|user_end|>",  # represent the marker that follows user message content
    "<|assistant_start|>",  # represent the marker that precedes assistant message content
    "<|assistant_end|>",  # represent the marker that follows assistant message content
    "<|python_start|>",  # represent the marker that indicates the beginning of a python tool call
    "<|python_end|>",  # represent the marker that indicates the end of a python tool call
    "<|output_start|>",  # represent the marker that introduces text returned by the python tool
    "<|output_end|>",  # represent the marker that terminates text returned by the python tool
]  # close the definition of the special token collection

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # provide the regex used to pre-tokenize text in a GPT-4-like manner

from tokenizers import Tokenizer as HFTokenizer  # import the HuggingFace Tokenizer implementation and alias it
from tokenizers import pre_tokenizers, decoders, Regex  # import helpers for configuring tokenization behavior
from tokenizers.models import BPE  # import the BPE model class used during training
from tokenizers.trainers import BpeTrainer  # import the BpeTrainer class responsible for learning merges

class HuggingFaceTokenizer:  # define a wrapper class that exposes a friendlier API over HuggingFace tokenizers
    def __init__(self, tokenizer):  # initialize the wrapper with an existing tokenizer instance
        self.tokenizer = tokenizer  # store the underlying tokenizer for all subsequent operations

    @classmethod  # declare the following method as a class-level constructor
    def from_pretrained(cls, hf_path):  # build a tokenizer from a pretrained HuggingFace checkpoint path
        tokenizer = HFTokenizer.from_pretrained(hf_path)  # load the tokenizer configuration and merges from the hub
        return cls(tokenizer)  # wrap the loaded tokenizer inside our helper class and return it

    @classmethod  # declare another class-level constructor
    def from_directory(cls, tokenizer_dir):  # build a tokenizer by reading files from a directory on disk
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")  # compute the path to the tokenizer json file
        tokenizer = HFTokenizer.from_file(tokenizer_path)  # load the tokenizer definition from the json file
        return cls(tokenizer)  # wrap the tokenizer and expose the helper methods

    @classmethod  # declare a class-level factory for training
    def train_from_iterator(cls, text_iterator, vocab_size):  # create and train a tokenizer from a stream of text
        tokenizer = HFTokenizer(BPE(  # instantiate a HuggingFace tokenizer using a BPE model configuration
            byte_fallback=True,  # enable byte fallback so unknown bytes map to byte tokens
            unk_token=None,  # disable an unknown token because BPE with byte fallback should cover all bytes
            fuse_unk=False,  # prevent unknown tokens from being fused during tokenization
        ))  # finish constructing the tokenizer model configuration
        tokenizer.normalizer = None  # explicitly disable any normalization so text passes through unchanged
        gpt4_split_regex = Regex(SPLIT_PATTERN)  # compile the GPT-4 style split pattern into a Regex object
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([  # configure a sequence of pre-tokenizers to prepare text
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),  # split text using the GPT-4 inspired regex into isolated tokens
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)  # apply byte-level splitting without prefix spaces to mirror GPT behavior
        ])  # finish setting up the pre-tokenizer pipeline
        tokenizer.decoder = decoders.ByteLevel()  # set the decoder to convert byte-level tokens back into strings
        tokenizer.post_processor = None  # avoid applying any post-processing so encoded ids remain untouched
        trainer = BpeTrainer(  # configure the BPE trainer that will learn the vocabulary merges
            vocab_size=vocab_size,  # set the desired vocabulary size including special tokens
            show_progress=True,  # enable progress reporting while the trainer processes text
            min_frequency=0,  # allow merges even if symbols only appear once
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # seed the trainer with the byte-level alphabet
            special_tokens=SPECIAL_TOKENS,  # reserve slots for all special tokens during training
        )  # complete the trainer configuration
        tokenizer.train_from_iterator(text_iterator, trainer)  # run BPE training over the provided text iterator
        return cls(tokenizer)  # wrap the freshly trained tokenizer into the helper class

    def get_vocab_size(self):  # expose the size of the tokenizer vocabulary
        return self.tokenizer.get_vocab_size()  # delegate to the underlying HuggingFace tokenizer implementation

    def get_special_tokens(self):  # expose the list of special tokens tracked by the tokenizer
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()  # obtain the mapping from token ids to token objects for added tokens
        special_tokens = [w.content for w in special_tokens_map.values()]  # extract the string value for each added token
        return special_tokens  # return the ordered collection of special token strings

    def id_to_token(self, id):  # convert a token id back into its string form
        return self.tokenizer.id_to_token(id)  # rely on the HuggingFace tokenizer to perform the lookup

    def _encode_one(self, text, prepend=None, append=None):  # encode a single piece of text with optional surrounding tokens
        assert isinstance(text, str)  # ensure the provided text is a string so encoding is well-defined
        ids = []  # start with an empty list that will accumulate token ids
        if prepend is not None:  # check whether a prefix token should be inserted
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)  # convert the prepend value into a token id
            ids.append(prepend_id)  # push the prepended token id to the front of the sequence
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)  # encode the text without automatically adding special tokens and extend the list
        if append is not None:  # check whether a suffix token should be appended
            append_id = append if isinstance(append, int) else self.encode_special(append)  # convert the append value into a token id
            ids.append(append_id)  # add the appended token id to the end of the sequence
        return ids  # return the final list of token ids for this text snippet

    def encode_special(self, text):  # encode a special token string via exact match
        return self.tokenizer.token_to_id(text)  # fetch the numeric id associated with the requested special token

    def get_bos_token_id(self):  # fetch the id of the beginning-of-sequence token
        bos = self.encode_special("<|bos|>")  # resolve the BOS token into its numeric id
        return bos  # return the BOS id for callers that need it

    def encode(self, text, *args, **kwargs):  # encode text while accepting positional and keyword options
        if isinstance(text, str):  # check if the provided input is a single string
            return self._encode_one(text, *args, **kwargs)  # delegate to the helper that encodes one string
        elif isinstance(text, list):  # otherwise check if the input is a list of strings
            return [self._encode_one(t, *args, **kwargs) for t in text]  # encode each string in the list and return the results
        else:  # handle unsupported types
            raise ValueError(f"Invalid input type: {type(text)}")  # raise an informative error about the unexpected input type

    def __call__(self, *args, **kwargs):  # allow the tokenizer object to be called like a function
        return self.encode(*args, **kwargs)  # forward call semantics to the encode method

    def decode(self, ids):  # convert a list of token ids back into text
        return self.tokenizer.decode(ids, skip_special_tokens=False)  # decode ids while preserving special tokens in the output

    def save(self, tokenizer_dir):  # persist the tokenizer configuration to disk
        os.makedirs(tokenizer_dir, exist_ok=True)  # ensure the destination directory exists before writing files
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")  # compute the path where the tokenizer json will be saved
        self.tokenizer.save(tokenizer_path)  # write the tokenizer configuration to disk
        print(f"Saved tokenizer to {tokenizer_path}")  # inform the caller where the tokenizer was stored

import pickle  # import pickle to serialize the rustbpe encoder to disk
import rustbpe  # import rustbpe to access the fast BPE trainer implemented in Rust
import tiktoken  # import tiktoken to leverage efficient runtime tokenization

class RustBPETokenizer:  # define a tokenizer that trains with rustbpe and runs inference with tiktoken
    def __init__(self, enc, bos_token):  # initialize the tokenizer wrapper with a tiktoken encoding
        self.enc = enc  # store the tiktoken encoding object for later reuse
        self.bos_token_id = self.encode_special(bos_token)  # cache the numeric id for the provided BOS token string

    @classmethod  # mark the following as a class-level constructor for training
    def train_from_iterator(cls, text_iterator, vocab_size):  # train a rustbpe tokenizer and wrap it for inference
        tokenizer = rustbpe.Tokenizer()  # instantiate a new rustbpe tokenizer object
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)  # compute the number of mergeable tokens excluding reserved specials
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"  # ensure the reduced vocabulary still supports byte coverage
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)  # fit the tokenizer using the GPT-4 style split pattern
        pattern = tokenizer.get_pattern()  # retrieve the regex pattern rustbpe learned for tokenization
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()  # fetch the list of mergeable byte pair ranks from training
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}  # convert the list of merges into the dictionary format expected by tiktoken
        tokens_offset = len(mergeable_ranks)  # determine the first id available for special tokens after the base merges
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}  # assign consecutive ids to each special token after mergeable ranks
        enc = tiktoken.Encoding(  # build the tiktoken Encoding object that mirrors the rustbpe model
            name="rustbpe",  # name the encoding so it can be identified
            pat_str=pattern,  # provide the regex pattern describing how to split text into initial tokens
            mergeable_ranks=mergeable_ranks,  # supply the merge ranks mapping bytes to their priority indices
            special_tokens=special_tokens,  # register the mapping from special token strings to ids
        )  # finish constructing the encoding object
        return cls(enc, "<|bos|>")  # wrap the encoding in the RustBPETokenizer using the BOS token as the anchor

    @classmethod  # declare a class-level constructor that loads from disk
    def from_directory(cls, tokenizer_dir):  # restore a previously saved tokenizer from a directory
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")  # compute the path to the serialized encoding file
        with open(pickle_path, "rb") as f:  # open the pickle file in binary read mode
            enc = pickle.load(f)  # deserialize the tiktoken encoding object from disk
        return cls(enc, "<|bos|>")  # wrap the loaded encoding with the BOS token label used during training

    @classmethod  # provide a class-level method to use existing tiktoken presets
    def from_pretrained(cls, tiktoken_name):  # construct the tokenizer by looking up a named tiktoken encoding
        enc = tiktoken.get_encoding(tiktoken_name)  # fetch the encoding associated with the provided name
        return cls(enc, "<|endoftext|>")  # wrap the encoding while using the end-of-text token as the BOS surrogate

    def get_vocab_size(self):  # expose the vocabulary size for the tiktoken encoding
        return self.enc.n_vocab  # return the number of tokens tracked by the underlying encoder

    def get_special_tokens(self):  # return the set of registered special tokens
        return self.enc.special_tokens_set  # delegate to the tiktoken encoding to retrieve the special token strings

    def id_to_token(self, id):  # convert a token id into its human-readable token string
        return self.enc.decode([id])  # decode a single-element list to leverage tiktoken's decode behavior

    @lru_cache(maxsize=32)  # cache special token encoding results to avoid repeated lookups
    def encode_special(self, text):  # translate a special token string into its id via the encoder
        return self.enc.encode_single_token(text)  # use tiktoken's helper to encode a single special token

    def get_bos_token_id(self):  # fetch the cached beginning-of-sequence id
        return self.bos_token_id  # return the stored BOS id to callers

    def encode(self, text, prepend=None, append=None, num_threads=8):  # encode text with optional prefix and suffix tokens
        if prepend is not None:  # check if the caller requested a token to add to the front
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)  # translate the prepend value into a numeric id
        if append is not None:  # check if the caller requested a token to add to the back
            append_id = append if isinstance(append, int) else self.encode_special(append)  # translate the append value into a numeric id
        if isinstance(text, str):  # handle encoding of a single string input
            ids = self.enc.encode_ordinary(text)  # encode the string without adding implicit special tokens
            if prepend is not None:  # if a prepend id was provided for the single string
                ids.insert(0, prepend_id)  # insert the prepend id at the beginning of the sequence
            if append is not None:  # if an append id was provided for the single string
                ids.append(append_id)  # append the token id to the end of the sequence
        elif isinstance(text, list):  # handle encoding of a batch of strings
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)  # encode all strings in parallel using the specified thread count
            if prepend is not None:  # if a prepend id was defined for the batch
                for ids_row in ids:  # iterate through each encoded sequence in the batch
                    ids_row.insert(0, prepend_id)  # insert the prepend id at the beginning of each sequence
            if append is not None:  # if an append id was defined for the batch
                for ids_row in ids:  # iterate through each encoded sequence in the batch
                    ids_row.append(append_id)  # append the token id to the end of each sequence
        else:  # handle unsupported input types
            raise ValueError(f"Invalid input type: {type(text)}")  # raise an error when the input is neither a string nor a list of strings
        return ids  # return the encoded ids or batch of ids back to the caller

    def __call__(self, *args, **kwargs):  # make the tokenizer callable like a function
        return self.encode(*args, **kwargs)  # route calls to the encode method for convenience

    def decode(self, ids):  # convert ids back into text using the tiktoken encoder
        return self.enc.decode(ids)  # delegate decoding to the underlying encoding implementation

    def save(self, tokenizer_dir):  # persist the encoder to disk so it can be reused later
        os.makedirs(tokenizer_dir, exist_ok=True)  # ensure the directory exists before writing files
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")  # determine the destination path for the serialized encoding
        with open(pickle_path, "wb") as f:  # open the target file for binary writing
            pickle.dump(self.enc, f)  # serialize the encoding object to disk using pickle
        print(f"Saved tokenizer encoding to {pickle_path}")  # notify the caller where the encoding was stored

    def render_conversation(self, conversation, max_tokens=2048):  # convert a structured conversation into token ids and supervision masks
        ids, mask = [], []  # prepare empty lists that will accumulate token ids and attention mask values
        def add_tokens(token_ids, mask_val):  # define a helper to add tokens along with their corresponding mask value
            if isinstance(token_ids, int):  # handle the case where a single token id is provided
                token_ids = [token_ids]  # wrap the single token id in a list for uniform processing
            ids.extend(token_ids)  # extend the ids list with the provided token ids
            mask.extend([mask_val] * len(token_ids))  # extend the mask list with repeated mask values for each token id
        if conversation["messages"][0]["role"] == "system":  # detect if the first message is a system directive
            conversation = copy.deepcopy(conversation)  # copy the conversation so the original object is not mutated
            messages = conversation["messages"]  # alias the list of messages after the copy
            assert messages[1]["role"] == "user", "System message must be followed by a user message"  # ensure the system message is immediately followed by a user entry
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]  # merge the system content into the first user message separated by blank lines
            messages = messages[1:]  # drop the original system message from the working list
        else:  # handle conversations that start with a user message directly
            messages = conversation["messages"]  # reuse the original messages without modification
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"  # ensure there is at least one message to process
        bos = self.get_bos_token_id()  # look up the beginning-of-sequence token id once for reuse
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")  # resolve user boundary tokens into ids
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")  # resolve assistant boundary tokens into ids
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")  # resolve python tool call boundary tokens into ids
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")  # resolve python output boundary tokens into ids
        add_tokens(bos, 0)  # prepend the BOS token and mark it as unsupervised for the assistant
        for i, message in enumerate(messages):  # iterate through the messages while tracking their index
            must_be_from = "user" if i % 2 == 0 else "assistant"  # determine which role should own the current message index
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"  # verify the conversation alternates as expected
            content = message["content"]  # extract the content field from the current message for inspection
            if message["role"] == "user":  # branch for user-authored messages
                assert isinstance(content, str), "User messages are simply expected to be strings"  # confirm user content is plain text
                value_ids = self.encode(content)  # tokenize the user text into ids
                add_tokens(user_start, 0)  # insert the user_start token without supervising it
                add_tokens(value_ids, 0)  # add user content tokens as context without supervision
                add_tokens(user_end, 0)  # append the user_end token without supervision
            elif message["role"] == "assistant":  # branch for assistant-authored messages
                add_tokens(assistant_start, 0)  # insert the assistant_start token without supervision
                if isinstance(content, str):  # handle assistant responses that are plain text
                    value_ids = self.encode(content)  # tokenize the assistant text into ids
                    add_tokens(value_ids, 1)  # supervise the assistant-generated tokens for training
                elif isinstance(content, list):  # handle assistant responses broken into structured parts
                    for part in content:  # iterate through each part to process them sequentially
                        value_ids = self.encode(part["text"])  # tokenize the textual portion of the current part
                        if part["type"] == "text":  # handle text parts authored by the assistant
                            add_tokens(value_ids, 1)  # supervise assistant text tokens for learning
                        elif part["type"] == "python":  # handle python tool invocations emitted by the assistant
                            add_tokens(python_start, 1)  # add the python_start token and supervise it
                            add_tokens(value_ids, 1)  # supervise the tokens representing the python code request
                            add_tokens(python_end, 1)  # add the python_end token and supervise it
                        elif part["type"] == "python_output":  # handle outputs returned from the python tool
                            add_tokens(output_start, 0)  # add the output_start token without supervision because it is environment produced
                            add_tokens(value_ids, 0)  # add the tool output tokens without supervision
                            add_tokens(output_end, 0)  # add the output_end token without supervision
                        else:  # handle unexpected part types
                            raise ValueError(f"Unknown part type: {part['type']}")  # raise an error describing the unsupported part type
                else:  # handle unsupported assistant content structures
                    raise ValueError(f"Unknown content type: {type(content)}")  # raise an error when the content is not a string or list
                add_tokens(assistant_end, 1)  # append the assistant_end token and supervise it to teach proper termination
        ids = ids[:max_tokens]  # truncate the token id list to the maximum allowed length to avoid excessive sequences
        mask = mask[:max_tokens]  # truncate the mask list to stay aligned with the id list after truncation
        return ids, mask  # return the pair of token ids and supervision mask values

    def visualize_tokenization(self, ids, mask):  # display tokenization results with color-coded supervision markers
        RED = '\033[91m'  # define the ANSI escape sequence for red text representing unsupervised tokens
        GREEN = '\033[92m'  # define the ANSI escape sequence for green text representing supervised tokens
        RESET = '\033[0m'  # define the ANSI code that resets terminal coloring
        tokens = []  # prepare a list to accumulate rendered token snippets
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):  # iterate over tokens alongside their mask values and indices
            token_str = self.decode([token_id])  # decode each token id back into a string fragment for display
            color = GREEN if mask_val == 1 else RED  # choose green for supervised tokens and red for unsupervised ones
            tokens.append(f"{color}{token_str}{RESET}")  # append the colored token representation to the list
        return '|'.join(tokens)  # join the colored tokens with separators so debugging output is readable

    def render_for_completion(self, conversation):  # prepare token ids to prompt the assistant for a completion
        conversation = copy.deepcopy(conversation)  # clone the conversation to avoid mutating the caller's data
        messages = conversation["messages"]  # access the mutable list of messages after copying
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"  # ensure the final message originates from the assistant
        messages.pop()  # remove the final assistant message so the model can produce it during completion
        ids, mask = self.render_conversation(conversation)  # render the remaining conversation structure into ids and mask
        assistant_start = self.encode_special("<|assistant_start|>")  # resolve the assistant_start token to prime the completion
        ids.append(assistant_start)  # append the assistant_start token to signal the model to generate the next message
        return ids  # return only the token ids because the mask is not needed for completion prompting

def get_tokenizer():  # helper that loads the production tokenizer for the application
    from nanochat.common import get_base_dir  # import get_base_dir lazily to avoid circular dependencies
    base_dir = get_base_dir()  # determine the base directory that stores tokenizer assets
    tokenizer_dir = os.path.join(base_dir, "tokenizer")  # compute the location of the tokenizer files within the base directory
    return RustBPETokenizer.from_directory(tokenizer_dir)  # load the tokenizer assets from disk using the rustbpe wrapper

def get_token_bytes(device="cpu"):  # helper that loads the serialized token bytes for embedding construction
    import torch  # import torch locally so callers that do not need token bytes avoid the dependency
    from nanochat.common import get_base_dir  # import get_base_dir inside the function to keep module imports lightweight
    base_dir = get_base_dir()  # resolve the repository base directory to locate tokenizer assets
    tokenizer_dir = os.path.join(base_dir, "tokenizer")  # build the path to the tokenizer asset directory
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")  # compute the file path holding the serialized token bytes
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"  # ensure the token bytes file exists before loading
    with open(token_bytes_path, "rb") as f:  # open the token bytes file in binary mode
        token_bytes = torch.load(f, map_location=device)  # load the tensor of token byte representations using the requested device
    return token_bytes  # return the loaded tensor to the caller
