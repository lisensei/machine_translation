import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, Subset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import vocab
from collections import OrderedDict
from nltk.translate.bleu_score import corpus_bleu
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
parser = ArgumentParser()
parser.add_argument("-parallel_set_path", type=str,
                    default="./data/validation.json")
parser.add_argument("-index_set_path", type=str,
                    default="./data/train_index_set_small_100000.json")
parser.add_argument("-source_tokenizer", type=str, default="en_core_web_sm")
parser.add_argument("-target_tokenizer", type=str, default="zh_core_web_sm")
parser.add_argument("-source_vocab_file", type=str,
                    default="data/english_vocab_small_100000.json")
parser.add_argument("-target_vocab_file", type=str,
                    default="data/chinese_vocab_small_100000.json")
parser.add_argument("-learning_rate", type=float, default=3e-4)
parser.add_argument("-dim_model", type=int, default=256)
parser.add_argument("-num_heads", type=int, default=8)
parser.add_argument("-encoder_layers", type=int, default=4)
parser.add_argument("-decoder_layers", type=int, default=4)
parser.add_argument("-max_sequence_length", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-epochs", type=int, default=200)
parser.add_argument("-training", type=int, default=1)
parser.add_argument("-training_samples", type=float, default=0.99999)
parser.add_argument("-training_evaluation_frequency", type=int, default=2)
parser.add_argument("-test_evaluation_frequency", type=int, default=2)
parser.add_argument("-logroot", type=str, default="./runs")
args = parser.parse_args()


class TranslationSet(Dataset):
    def __init__(self, source_path, target_path):
        self.source = []
        self.target = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                self.source.append(line.strip())
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                self.target.append(line.strip())

    def __getitem__(self, index):
        return (self.source[index], self.target[index])

    def __len__(self, ):
        return len(self.source)


class ParallelDataset(Dataset):
    def __init__(self, corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def __getitem__(self, index):
        pair = self.dataset[index]
        source, target = pair["english"], pair["chinese"]
        return source, target

    def __len__(self):
        return len(self.dataset)


class IndexSet(Dataset):
    def __init__(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        pair = self.data[index]
        return pair["english"], pair["chinese"]

    def __len__(self):
        return len(self.data)

    def get_dataset_stats(self):
        source_max_length = 0
        target_max_length = 0
        number_of_source_tokens = 0
        number_of_target_tokens = 0
        for i, pair in enumerate(self.data):
            source, target = pair["english"], pair["chinese"]
            source_length = len(source)
            target_length = len(target)
            number_of_source_tokens += source_length
            number_of_target_tokens += target_length
            if source_length > source_max_length:
                source_max_length = source_length
            if target_length > target_max_length:
                target_max_length = target_length
        stat_dict = {"total_source_tokens": number_of_source_tokens,
                     "total_target_tokens": number_of_target_tokens,
                     "max_source_sentence_length": source_max_length,
                     "max_target_sentence_length": target_max_length,
                     "number_of_pairs": i+1}
        return stat_dict


def token_yielder(datastream, tokenizer, language):
    for i, (source, target) in enumerate(datastream):
        if language:
            yield tokenizer(target)
        else:
            yield tokenizer(source)


np.random.seed(424242)
torch.random.manual_seed(424242)
# logging
run_date = datetime.now().isoformat(timespec='seconds')
logdir = f"{args.logroot}/{run_date}"
if not os.access(logdir, os.F_OK):
    os.makedirs(logdir)
writer = SummaryWriter(logdir)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logfile = f"{logdir}/log.txt"
file_handler = logging.FileHandler(logfile, encoding="utf-8")
logger.addHandler(file_handler)
logger.removeHandler(logger.handlers[0])
# create vocabulary
english_tokenizer = get_tokenizer("spacy", args.source_tokenizer)
chinese_tokenizer = get_tokenizer("spacy", args.target_tokenizer)
dataset = ParallelDataset(args.parallel_set_path)
specials = ["<UNK>", "<BOS>", "<EOS>", "<PAD>"]
UNK_INDEX, BOS_INDEX, EOS_INDEX, PAD_INDEX = specials.index(
    "<UNK>"), specials.index("<BOS>"), specials.index("<EOS>"), specials.index("<PAD>")
english_vocab_exist = False
chinese_vocab_exist = False
if os.access(args.source_vocab_file, os.F_OK):
    english_vocab_exist = True
    logger.info(f"found existing English vocabulary, loading")
    with open(args.source_vocab_file, "r") as f:
        english_vocabulary = vocab(json.load(f))
else:
    english_vocabulary = build_vocab_from_iterator(token_yielder(
        dataset, english_tokenizer, 0), specials=specials, special_first=True)
english_vocabulary.set_default_index(specials.index("<UNK>"))

if os.access(args.target_vocab_file, os.F_OK):
    chinese_vocab_exist = True
    logger.info(f"found existing Chinese vocabulary, loading")
    with open(args.target_vocab_file, "r") as f:
        chinese_vocabulary = vocab(json.load(f))
else:
    chinese_vocabulary = build_vocab_from_iterator(token_yielder(
        dataset, chinese_tokenizer, 1), specials=specials, special_first=True)
chinese_vocabulary.set_default_index(specials.index("<UNK>"))
if not english_vocab_exist:
    logger.info(f"saving english vocabulary")
    with open(args.source_vocab_file, "w", encoding="utf-8") as f:
        json.dump(OrderedDict([(token, 1)
                               for token in english_vocabulary.vocab.get_itos()]), f)
if not chinese_vocab_exist:
    logger.info(f"saving chinese vocabulary")
    with open(args.target_vocab_file, "w", encoding="utf-8") as f:
        json.dump(OrderedDict([(token, 1)
                               for token in chinese_vocabulary.get_itos()]), f)


def collate_fn(x):
    source_batch = []
    target_batch = []
    for source, target in x:
        source_indices = [BOS_INDEX]
        source_indices.extend(english_vocabulary(english_tokenizer(source)))
        source_indices.append(EOS_INDEX)
        source_batch.append(torch.tensor(source_indices, dtype=torch.int64))

        target_indices = [BOS_INDEX]
        target_indices.extend(chinese_vocabulary(chinese_tokenizer(target)))
        target_indices.append(EOS_INDEX)
        target_batch.append(torch.tensor(target_indices, dtype=torch.int64))
    return pad_sequence(source_batch, True, PAD_INDEX), pad_sequence(target_batch, True, PAD_INDEX)


def collate_fn_index(x):
    source_batch = []
    target_batch = []
    for source, target in x:
        source_batch.append(torch.tensor(source, dtype=torch.int64))
        target_batch.append(torch.tensor(target, dtype=torch.int64))
    return pad_sequence(source_batch, True, 3), pad_sequence(target_batch, True, 3)


class TransformerNet(nn.Module):
    def __init__(self, dim_model, heads, encoder_layers, decoder_layers, soruce_vocab_size, target_vocab_size,
                 max_sequence_length=1000, batch_first=True):
        super().__init__()
        self.heads = heads
        self.source_embeddings = nn.Embedding(soruce_vocab_size, dim_model)
        self.source_positional_embeddings = nn.Embedding(
            max_sequence_length, dim_model)
        self.target_embeddings = nn.Embedding(target_vocab_size, dim_model)
        self.target_positional_embeddings = nn.Embedding(
            max_sequence_length, dim_model)
        self.net = nn.Transformer(
            dim_model, heads, encoder_layers, decoder_layers, batch_first=batch_first)
        self.fc = nn.Linear(dim_model, target_vocab_size)

    def encode(self, source, source_key_padding_mask):
        source_sequence_length = source.size(1)
        source_embeddings = self.source_embeddings(source)
        source_positional_embeddings = self.source_positional_embeddings(
            torch.arange(source_sequence_length, device=source.device).unsqueeze(0).expand(source.size()))
        final_source_embeddings = source_embeddings + source_positional_embeddings
        out = self.net.encoder.forward(
            final_source_embeddings, src_key_padding_mask=source_key_padding_mask)
        return out

    def decode(self, target, memory, memory_key_padding_mask):
        target_sequence_length = target.size(1)
        target_embeddings = self.target_embeddings(target)
        target_positional_embeddings = self.target_positional_embeddings(
            torch.arange(target_sequence_length, device=target.device).unsqueeze(0).expand(target.size()))
        final_target_embeddings = target_embeddings + target_positional_embeddings
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_sequence_length).unsqueeze(
            0).expand((target.size(0) * self.heads, target_sequence_length, target_sequence_length)).to(target.device)
        target_key_padding_mask = (target == PAD_INDEX).to(target_mask.dtype)
        output = self.net.decoder.forward(tgt=final_target_embeddings, memory=memory, tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, source, target):
        source_key_padding_mask = source == PAD_INDEX
        memory = self.encode(source, source_key_padding_mask)
        output = self.decode(target=target, memory=memory,
                             memory_key_padding_mask=source_key_padding_mask)
        output = self.fc(output)
        return output

    @torch.no_grad()
    def infer(self, source_indices, device, max_length=args.max_sequence_length):
        # batch size must be 1
        if source_indices.size(0) != 1:
            raise Exception(f"batch size must equal one during inference.")
        self.eval()
        source_indices = source_indices.to(device)
        skpm = source_indices == PAD_INDEX
        output_memory = self.encode(source_indices, skpm)
        target = torch.tensor([BOS_INDEX], device=device).reshape(1, -1)
        while target[0, -1] != EOS_INDEX and target.size(1) < max_length:
            out = self.decode(target,
                              output_memory, None)
            output = self.fc(out)
            predicted_token = torch.argmax(
                output[0, -1]).detach().reshape(1, -1)
            target = torch.concat([target, predicted_token], dim=1)
        self.train()
        return target


def get_performance_on_dataset(model, dataloader, device, dataset_type="test"):
    references = []
    translations = []
    for i, (source, target) in enumerate(dataloader):
        source = source.to(device)
        target = target.reshape(-1).tolist()
        out = model.infer(source, device)
        translation = chinese_vocabulary.lookup_tokens(
            out.reshape(-1).tolist())
        reference = chinese_vocabulary.lookup_tokens(target)
        translations.append(translation)
        references.append([reference])
        if i < 10:
            source_sentence = english_indices_to_sentence(
                source.detach().cpu())
            translated_sentence = ''.join(translation)
            reference_sentence = ''.join(reference)
            logger.info(
                f"\n=================\n\n{dataset_type} evaluation: \n{dataset_type} example No.{i}\n\nsource:      {source_sentence} \n\ntranslation: {translated_sentence}\n\nreference:   {reference_sentence}\n\n=================\n")
    return corpus_bleu(references, translations)


def chinese_indices_to_sentence(source):
    if type(source) == torch.Tensor:
        source = source.reshape(-1).tolist()
    return ''.join(chinese_vocabulary.lookup_tokens(source))


def english_indices_to_sentence(source):
    if type(source) == torch.Tensor:
        source = source.reshape(-1).tolist()
    return " ".join(english_vocabulary.lookup_tokens(source))


def translate(sentence, model):
    source = english_vocabulary.lookup_indices(
        english_tokenizer(sentence))
    source.insert(0, specials.index("<BOS>"))
    source.append(specials.index("<EOS>"))
    tgt = model.infer(torch.tensor(source).reshape(1, -1), DEVICE)
    translation = ''.join(chinese_vocabulary.lookup_tokens(
        tgt.reshape(-1).cpu().tolist()))
    return translation


dataset = IndexSet(args.index_set_path)
SOURCE_VOCAB_SIZE = len(english_vocabulary)
TARGET_VOCAB_SIZE = len(chinese_vocabulary)
net = TransformerNet(args.dim_model, args.num_heads, args.encoder_layers, args.decoder_layers,
                     SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, args.max_sequence_length)

LOSS_FN = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
OPTIMIZER = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    net.cuda()
DATASET_SIZE = len(dataset)
if args.training:
    training_set_size = int(args.training_samples*DATASET_SIZE)
    training_set, testset = random_split(
        dataset, [training_set_size, DATASET_SIZE - training_set_size])
else:
    if DATASET_SIZE < args.batch_size*2:
        raise Exception("Insufficient samples")
    training_set_size = args.batch_size*2
    training_set, testset = Subset(dataset, list(range(training_set_size))), Subset(
        dataset, list(range(training_set_size, training_set_size+args.batch_size)))

training_evaluation_set = Subset(training_set, list(range(
    args.batch_size if training_set_size > args.batch_size else training_set_size)))
dataloader = DataLoader(training_set, args.batch_size,
                        shuffle=True, collate_fn=collate_fn_index)

testloader = DataLoader(testset, batch_size=1, collate_fn=collate_fn_index)

evaluation_dataloader = DataLoader(
    training_evaluation_set, batch_size=1, collate_fn=collate_fn_index)

losses = []
# _, axe = plt.subplots(1, 1, constrained_layout=True)
num_of_parameters = sum([param.numel() for param in net.parameters()])
for k, v in (args.__dict__ | dataset.get_dataset_stats()).items():
    logger.info(f"{k} : {v}")
logger.info(
    f"English vocabulary size: {len(english_vocabulary)}; Chinese vocabulary size: {len(chinese_vocabulary)}")
logger.info(f"total number of parameters: {num_of_parameters}")
logger.info(
    f"training set size: {training_set_size}, test set size: {len(testset)}")
logger.info(f"running on : {DEVICE}")
logger.info(f"training starts:\n")
for e in tqdm(range(args.epochs)):
    epoch_loss = 0
    running_total = 0
    epoch_training_bleu_simple_average = 0
    epoch_predictions = []
    epoch_references = []
    for i, (source, target) in enumerate(dataloader):
        if source.size(1) > args.max_sequence_length or target.size(1) > args.max_sequence_length:
            logger.info(
                f"Batch of long sequence skipped. source shape: {source.shape}; target shape: {target.shape}")
            continue
        training_predictions = []
        training_references = []
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        out = net(source, target[:, :-1])
        loss = LOSS_FN(out.permute(0, 2, 1), target[:, 1:])
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        epoch_loss += loss.detach().cpu().numpy()
        predictions = torch.argmax(out, dim=2).tolist()
        references = target.tolist()
        for ref, pred in zip(references, predictions):
            ref = chinese_vocabulary.lookup_tokens(ref)
            pred = chinese_vocabulary.lookup_tokens(pred)
            pred.insert(0, "<BOS>")
            ref_index = ref.index("<PAD>") if "<PAD>" in ref else None
            pred_index = pred.index("<EOS>")+1 if "<EOS>" in pred else None
            reference = ref[:ref_index]
            prediction = pred[:pred_index]
            training_references.append([reference])
            training_predictions.append(prediction)
        epoch_predictions.extend(training_predictions)
        epoch_references.extend(training_references)
        epoch_training_bleu_simple_average += corpus_bleu(training_references,
                                                          training_predictions)*(source.size(0)/training_set_size)
        running_total += source.size(0)
        mod = training_set_size//args.batch_size//args.training_evaluation_frequency
        if i % (mod if mod else 1) == 0:
            training_bleu = get_performance_on_dataset(
                net, evaluation_dataloader, DEVICE, "training")
            logger.info(
                f"progress: {running_total}/{training_set_size}; training bleu score: {epoch_training_bleu_simple_average}")
    epoch_corpus_bleu = corpus_bleu(epoch_references, epoch_predictions)
    logger.info(
        f"\n epoch: {e}; epoch loss: {epoch_loss}; simple average corpus bleu:{epoch_training_bleu_simple_average}; corpus bleu: {epoch_corpus_bleu}")
    if e % args.test_evaluation_frequency == 0 or e == args.epochs-1:
        bleu = get_performance_on_dataset(net, testloader, DEVICE)
        logger.info(f"test bleu score: {bleu}")
    writer.add_scalar("loss", epoch_loss, e)
    writer.add_scalar("train bleu", epoch_training_bleu_simple_average, e)
    writer.add_scalar("test bleu", bleu, e)
    # losses.append(epoch_loss)
    # plt.clf()
    # plt.plot(np.arange(e + 1), losses)
    # plt.show()
    torch.save(net.state_dict(), f"{logdir}/model.pth")

source_sentence = "I need another story"
translation = translate(source_sentence, net)
logger.info(
    f"\nTraining stopped\nEnglish: <BOS> {source_sentence} <EOS> \nChinese: {translation}")
