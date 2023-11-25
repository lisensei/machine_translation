import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm


def vocab_test(source_vocab, target_vocab):
    print(f"{source_vocab.lookup_indices(['I','need','another','story'])}")
    print(
        f"{' '.join(source_vocab.lookup_tokens(source_vocab.lookup_indices(['I','need','another','story'])))}")
    print(f"{target_vocab.lookup_indices(['我','需要','另','一个','故事'])}")
    print(
        f"{''.join(target_vocab.lookup_tokens(target_vocab.lookup_indices(['我','需要','另','一个','故事'])))}")


def collate_fn_test(source_vocab, tagart_vocab, dataloader):
    for i, (x, y) in enumerate(dataloader):
        print(
            f"\n===================\nsource:\n{' '.join(source_vocab.lookup_tokens(x[0].cpu().tolist()))}")
        print(
            f"\ntarget:\n{''.join(tagart_vocab.lookup_tokens(y[0].cpu().tolist()))}")


def fix_json_file(filename, new_filename):
    file = open(filename, "r", encoding="utf-8")
    new_file = open(new_filename, "w", encoding="utf-8")
    new_file.write("[")
    previous_line = file.readline()
    current_line = previous_line
    while current_line != "":
        current_line = file.readline()
        if current_line == "":
            previous_line = previous_line.strip()
        else:
            previous_line = previous_line.strip()+",\n"
        new_file.write(previous_line)
        previous_line = current_line
    new_file.write("]")
    file.close()
    new_file.close()


def token_yielder_json(filename, tokenizer, language, expected_size=None):
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if expected_size is not None:
                if i > expected_size-1:
                    break
            pair = json.loads(line)
            sentence = pair[language]
            yield tokenizer(sentence)


def generate_indices_set(filename: str, source_tokenizer, target_tokenizer, output_filename, source_vocab_name=None, target_vocab_name=None, expected_size=1000000):
    specials = ["<UNK>", "<BOS>", "<EOS>", "<PAD>"]
    eng_vocab = build_vocab_from_iterator(
        token_yielder_json(filename, source_tokenizer, "english", expected_size), specials=specials, special_first=True)
    eng_vocab.set_default_index(0)
    if source_vocab_name is None:
        source_vocab_name = "source_vocab_small.json"
    with open(source_vocab_name, "w", encoding="utf-8") as f:
        print(f"saving source vocab file")
        json.dump(OrderedDict([(token, 1)
                  for token in eng_vocab.vocab.get_itos()]), f)
    chi_vocab = build_vocab_from_iterator(
        token_yielder_json(filename, target_tokenizer, "chinese", expected_size), specials=specials, special_first=True)
    chi_vocab.set_default_index(0)
    if target_vocab_name is None:
        target_vocab_name = "target_vocab_small.json"
    with open(target_vocab_name, "w", encoding="utf-8") as f:
        print(f"saving target vocab file")
        json.dump(OrderedDict([(token, 1)
                  for token in chi_vocab.vocab.get_itos()]), f)
    new_file = open(output_filename, "w", encoding="utf-8")
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > expected_size-1:
                break
            pair = json.loads(line)
            eng_indices = [specials.index("<BOS>")]
            chi_indices = [specials.index("<BOS>")]
            eng_indices.extend(eng_vocab.lookup_indices(
                source_tokenizer(pair["english"])))
            chi_indices.extend(chi_vocab.lookup_indices(
                target_tokenizer(pair["chinese"])))
            eng_indices.append(specials.index("<EOS>"))
            chi_indices.append(specials.index("<EOS>"))
            json.dump({"english": eng_indices,
                       "chinese": chi_indices}, new_file)
            new_file.write("\n")
    new_file.close()


class IndexSet(Dataset):
    def __init__(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        pair = self.data[index]
        return pair["english"], pair["chinese"]

    def __len__(self):
        return len(self.data)


def collate_fn_index(x):
    source_batch = []
    target_batch = []
    for source, target in x:
        source_batch.append(torch.tensor(source, dtype=torch.int64))
        target_batch.append(torch.tensor(target, dtype=torch.int64))
    return pad_sequence(source_batch, True, 3), pad_sequence(target_batch, True, 3)


if __name__ == "__main__":
    source_tokenizer = get_tokenizer("spacy", "en_core_web_sm")
    target_tokenizer = get_tokenizer("spacy", "zh_core_web_sm")
    input_file = "./data/translation2019zh_train.json"
    expected_size = 250000
    source_vocab_name = f"./data/english_vocab_small_{str(expected_size)}.json"
    target_vocab_name = f"./data/chinese_vocab_small_{str(expected_size)}.json"
    temp_filename = "./data/temp.json"
    output_file = f"./data/train_index_set_small_{expected_size}.json"
    generate_indices_set(input_file,
                         source_tokenizer, target_tokenizer, temp_filename, source_vocab_name=source_vocab_name, target_vocab_name=target_vocab_name, expected_size=expected_size)
    fix_json_file(temp_filename,
                  output_file)
    print("Done")