from functools import partial
from typing import Optional, Dict
from itertools import product

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import BertTokenizerFast
from rouge_score import rouge_scorer


class RACEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            model_name_or_path: str = 'bert-large-uncased',
            datasets_loader: str = 'race',
            task_name: str = 'all',
            max_seq_length: int = 640,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            num_workers: int = 8,
            num_preprocess_processes: int = 8,
            use_sentence_selection: bool = False,
            best_k_sentences: int = 5,
            **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_loader = datasets_loader
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_preprocess_processes = num_preprocess_processes
        self.use_sentence_selection = use_sentence_selection
        self.best_k_sentences = best_k_sentences

        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True, do_lower_case=True)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.load_dataset(self.dataset_loader, self.task_name)

        preprocessor = partial(self.preprocess, self.tokenizer, self.scorer, self.max_seq_length,
                               self.use_sentence_selection, self.best_k_sentences)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                preprocessor,
                # batched=True,
                remove_columns=['example_id'],
                num_proc=self.num_preprocess_processes,
                keep_in_memory=True,
            )
            self.dataset[split].set_format(type='torch',
                                           columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    def prepare_data(self):
        datasets.load_dataset(self.dataset_loader, self.task_name)
        BertTokenizerFast.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          sampler=DistributedSampler(self.dataset['train']),
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset['validation'],
                          sampler=SequentialSampler(self.dataset['validation']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          sampler=SequentialSampler(self.dataset['test']),
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers)

    # auto cache tokens
    @staticmethod
    def preprocess(tokenizer: BertTokenizerFast, scorer: rouge_scorer, max_seq_length: int, use_sentence_selection: bool,
                   best_k_sentences: int, x: Dict) -> Dict:
        choices_features = []
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        question = x["question"]
        article = x['article']
        if use_sentence_selection:
            qa = [question + option for option in x["options"]]

            # question_tokens = np.array(tokenizer(qa, add_special_tokens=False, truncation=True, max_length=25,
            #                                      padding='max_length')['input_ids'])
            # question_tokens = np.array(tokenizer(qa, add_special_tokens=False)['input_ids'])
            sentences = article.split('.')
            sentences = [s for s in sentences if s != '']
            # sentences_tokens = np.array(tokenizer(sentences, add_special_tokens=False, truncation=True, max_length=25,
            #                                       padding='max_length')['input_ids'])
            # sentences_tokens = np.array(tokenizer(sentences, add_special_tokens=False)['input_ids'])
            question_len = len(qa)
            sentences_len = len(sentences)
            sentence_scores = np.empty((sentences_len, question_len))
            for (i, j) in product(range(sentences_len), range(question_len)):
                scores = scorer.score(sentences[i], qa[j])
                sentence_scores[i, j] = scores['rouge1'].precision + scores['rouge2'].precision
            # sentence_scores = np.dot(sentences_tokens, question_tokens.T) / (np.linalg.norm(
            #     sentences_tokens, axis=1).reshape(-1, 1) @ np.linalg.norm(
            #     question_tokens, axis=1).reshape(1, -1))
            max_sentence_score = np.max(sentence_scores, axis=1)
            best_sentence_indices = max_sentence_score.argsort()[-best_k_sentences:][::-1]
            final_indices = set()
            for index in best_sentence_indices:
                final_indices.add(index - 1)
                final_indices.add(index)
                final_indices.add(index + 1)
            final_indices.discard(-1)
            final_indices.discard(sentences_len)

            article = '.'.join([sentences[i] for i in sorted(final_indices)])

        option: str
        for option in x["options"]:
            # question_option = question + option
            if question.find("_") != -1:
                # fill in the banks questions
                question_option = question.replace("_", option)
            else:
                question_option = question + " [SEP] " + option

            inputs = tokenizer(
                article,
                question_option,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            choices_features.append(inputs)

        labels = label_map.get(x["answer"], -1)
        label = torch.tensor(labels).long()

        return {
            "label": label,
            "input_ids": torch.cat([cf["input_ids"] for cf in choices_features]).reshape(-1),
            "attention_mask": torch.cat([cf["attention_mask"] for cf in choices_features]).reshape(-1),
            "token_type_ids": torch.cat([cf["token_type_ids"] for cf in choices_features]).reshape(-1),
        }


if __name__ == '__main__':
    dm = RACEDataModule(train_batch_size=32)
    dm.setup('train')
    d = (next(iter(dm.test_dataloader())))
    print(d)
