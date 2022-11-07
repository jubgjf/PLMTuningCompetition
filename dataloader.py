import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from modeling_tokenizer import T5Tokenizer

# 从huggingface datasets脚本中读取数据
def load_hf_dataset(task_name: str = 'SST-2', seed: int = 8, split: str = 'train') -> datasets.Dataset:
    """
    Please choose from:
    :param task_name: 'AGNews', 'MRPC', 'SNLI', 'SST-2', 'TREC', 'Yelp'
    :param seed: 8, 13, 42, 50, 60
    :param split: 'train', 'dev'
    """
    dataset = datasets.load_dataset(
        path=f'./datasets/{task_name}/{task_name}.py',
        split=f'{split}_{seed}'
    )
    return dataset


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=8)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "negative",
            1: "positive",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = 'text: %s %s . It was <extra_id_0> </s>' % (prompt, example['text'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        else:
            example['input_text'] = 'text: %s . It was <extra_id_0> </s>' % example['text']
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='SST-2', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle



class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Entailment",
            1: "Neutral",
            2: "Contradiction",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]

        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='SNLI', split=split, seed=seed)
        dataset = dataset.filter(lambda example: example['labels'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class DBPediaLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Company",
            1: "EducationalInstitution",
            2: "Artist",
            3: "Athlete",
            4: "OfficeHolder",
            5: "MeanOfTransportation",
            6: "Building",
            7: "NaturalPlace",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "WrittenWork",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s [ Category: <extra_id_0> ] %s' % (prompt, example['text'].strip())
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        else:
            example['input_text'] = '[ Category: <extra_id_0> ] %s' % (example['text'].strip())
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]

        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='DBPedia', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Entailment",
            1: "NotEntailment",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='QNLI', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QQPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? <extra_id_0> , %s' % (prompt, example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? <extra_id_0> , %s' % (
            example['text1'], example['text2'])
            example['target_text'] = '<pad> <extra_id_0> %s </s>' % self.label2text[example['labels']]

        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='QQP', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "decoder_input_ids": ins["decoder_input_ids"],
                    "decoder_attention_mask": ins["decoder_attention_mask"],
                    "labels": ins["decoder_input_ids"][2],
                }
                ds.append(Instance(**example))
            ds.set_input("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
