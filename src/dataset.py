import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm

from src.utils import read_features, get_class_names


class VGGSoundDataset(data.Dataset):
    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"
    @property
    def training_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_base{self.zero_shot_split}.pkl"
    @property
    def training_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_novel{self.zero_shot_split}.pkl"
    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"
    @property
    def train_val_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_base{self.zero_shot_split}.pkl"
    @property
    def train_val_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_1{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_2{self.zero_shot_split}.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    @property
    def test_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_1{self.zero_shot_split}.pkl"

    @property
    def test_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_2{self.zero_shot_split}.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": np.array(self.data["audio"]["data"])[classes_mask],
            "video": np.array(self.data["video"]["data"])[classes_mask],
            "text": np.array(self.data["text"]["data"])[sorted(self.classes.astype(int))],
            "target": np.array(self.data["audio"]["target"])[classes_mask],
            "url": np.array(self.data["audio"]["url"])[classes_mask],
            "fps": np.array(self.data['audio']['fps'])[classes_mask]
        }

    @property
    def map_embeddings_target(self):
        w2v_embedding = torch.Tensor(self.data["text"]["data"])[sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def features_processed_folder(self):
        return Path().cwd() / str("avgzsl_benchmark_non_averaged_datasets/"+self.embedding_type+"/VGGSound/_features_processed")

    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_1_base_train_class_name])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_1_val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_1_val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_2_base_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_2_test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.stage_2_test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/vggsound_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train_base":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "train_novel":
            return np.array([])
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_seen_class_names
        elif self.dataset_split == "train_val_base":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split=="train_val_novel_2":
            return np.array([])
        elif self.dataset_split == "train_stage_2":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split=="test_2":
            return self.stage_2_test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train_base":
            return np.array([])
        elif self.dataset_split == "train_novel":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_unseen_class_names
        elif self.dataset_split == "train_val_base":
            return np.array([])
        elif self.dataset_split == "train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split=="train_val_novel_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "train_stage_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split=="test_2":
            return self.stage_2_test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name] for name in self.unseen_class_names])

    @property
    def stage_1_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_base_train.txt")

    @property
    def stage_1_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_novel_train.txt")

    @property
    def stage_1_val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def stage_1_val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def stage_2_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_base_train.txt")

    @property
    def stage_2_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_novel_train.txt")

    @property
    def stage_2_test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def stage_2_test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(VGGSoundDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split
        self.embedding_type=args.embedding_type
        self.transform = transform
        self.text_embeddings_file=args.text_embeddings_file

        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_novel_file.exists() and self.training_base_file.exists() and self.val_file.exists() and \
            self.test_file.exists() and self.test_file_1.exists() and self.test_file_2.exists() and self.train_val_base_file.exists() \
            and self.train_val_novel_file.exists() and self.train_val_novel_file_1.exists() and self.train_val_novel_file_2.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        train_val_base_set = self.read_dataset(dataset_type="train_val_base")
        with self.train_val_base_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_base_file}")
            pickle.dump(train_val_base_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_base_set

        training_set_base = self.read_dataset(dataset_type="train_base")
        with self.training_base_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_base_file}")
            pickle.dump(training_set_base, f, pickle.HIGHEST_PROTOCOL)
        del training_set_base
        training_set_novel = self.read_dataset(dataset_type="train_novel")
        with self.training_novel_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_novel_file}")
            pickle.dump(training_set_novel, f, pickle.HIGHEST_PROTOCOL)
        del training_set_novel
        val_set = self.read_dataset(dataset_type="val")
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        del val_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel")
        with self.train_val_novel_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_1")
        with self.train_val_novel_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_1}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_2")
        with self.train_val_novel_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_2}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        test_set = self.read_dataset(dataset_type="test")
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_1")
        with self.test_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_1}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_2")
        with self.test_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_2}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train_novel":
            data_file = self.training_novel_file
        elif self.dataset_split == "train_base":
            data_file = self.training_base_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val_base":
            data_file = self.train_val_base_file
        elif self.dataset_split == "train_val_novel":
            data_file = self.train_val_novel_file
        elif self.dataset_split == "train_val_novel_1":
            data_file = self.train_val_novel_file_1
        elif self.dataset_split == "train_val_novel_2":
            data_file = self.train_val_novel_file_2
        elif self.dataset_split == "test":
            data_file = self.test_file
        elif self.dataset_split == "test_1":
            data_file = self.test_file_1
        elif self.dataset_split == "test_2":
            data_file = self.test_file_2
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        result = {"data": [], "target": [], "url": [], "fps": []}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text" / self.text_embeddings_file).resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key]] for key in list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train_base":
                split_names.append("stage_1_train_base")
            elif dataset_type == "train_novel":
                split_names.append("stage_1_train_novel")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type == "train_val_novel":
                split_names.append("stage_2_train_novel")
            elif dataset_type == "train_val_novel_1":
                split_names.append("stage_2_train_novel_1")
            elif dataset_type == "train_val_novel_2":
                split_names.append("stage_2_train_novel_2")
            elif dataset_type == "train_val_base":
                split_names.append("stage_1_train_base")
                split_names.append("stage_1_train_novel")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel")
            elif dataset_type == "test_1":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_1")
            elif dataset_type == "test_2":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_2")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url, fps = read_features(file)
                    #assert len(data[
                    #               0]) == self.args.input_size, f"Feature size {len(data[0])} is not compatible with specified --input_size {self.args.input_size}"
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem])
                        result["url"].append(url[i])
                        result["fps"].append(fps[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = result["data"]
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result


class ContrastiveDataset(data.Dataset):
    def __init__(self, zsl_dataset):
        super(ContrastiveDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Based on Dataset: {zsl_dataset.__class__.__name__}\t"
            f"with split: {zsl_dataset.dataset_split}")
        self.zsl_dataset = zsl_dataset
        self.dataset_split = self.zsl_dataset.dataset_split
        self.classes = self.zsl_dataset.classes
        if self.dataset_split == "train_base" or self.dataset_split == "train_novel" \
                or self.dataset_split=="train_val_base" or self.dataset_split=="train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split=="train_val_novel_2":
            self.targets = self.zsl_dataset.targets
            self.data = self.zsl_dataset.all_data
            self.targets_set = set(self.targets.tolist())
            self.target_to_indices = {target: np.where(self.zsl_dataset.targets == target)[0]
                                      for target in self.targets_set}

        elif self.dataset_split == "val" or self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split=="test_2":
            self.targets = self.zsl_dataset.targets
            self.data = self.zsl_dataset.all_data
            # generate fixed pairs for testing
            self.targets_set = set(self.targets.tolist())
            self.target_to_indices = {target: np.where(self.zsl_dataset.targets == target)[0]
                                      for target in self.targets_set}

            random_state = np.random.RandomState(29)

            # pos_neg_pairs = [i,j] -> list of all targets i with random respective negative index j
            pos_neg_pairs = [[i,
                              random_state.choice(self.target_to_indices[
                                                      np.random.choice(
                                                          list(self.targets_set - set([self.targets[i].item()]))
                                                      )
                                                  ])
                              ]
                             for i in range(len(self.targets))]
            self.val_pairs = pos_neg_pairs
        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

    def __len__(self):
        classes_mask = np.where(np.isin(self.zsl_dataset.targets, self.classes))[0]
        return len(self.zsl_dataset.targets[classes_mask])

    def __getitem__(self, index):
        if self.dataset_split == "train_base" or self.dataset_split == "train_novel"\
                or self.dataset_split=="train_val_base" or self.dataset_split=="train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split=="train_val_novel_2":
            positive_target = self.targets[index].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][index]
            x_v1 = self.data["video"][index]
            x_t1 = self.data["text"][pos_target_index]
            x_url1 = self.data["url"][index]
            fps_1=self.data["fps"][index]
            negative_target = np.random.choice(list(self.targets_set - set([positive_target])))
            negative_index = np.random.choice(self.target_to_indices[negative_target])
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][negative_index]
            x_v2 = self.data["video"][negative_index]
            x_t2 = self.data["text"][neg_target_index]
            x_url2 = self.data["url"][negative_index]
            fps_2=self.data['fps'][negative_index]
        elif self.dataset_split == "val" or self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split=="test_2":
            positive_target = self.targets[self.val_pairs[index][0]].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][self.val_pairs[index][0]]
            x_v1 = self.data["video"][self.val_pairs[index][0]]
            x_t1 = self.data["text"][pos_target_index]
            fps_1=self.data['fps'][self.val_pairs[index][0]]
            x_url1 = self.data["url"][self.val_pairs[index][0]]
            negative_target = self.targets[self.val_pairs[index][1]].item()
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][self.val_pairs[index][1]]
            x_v2 = self.data["video"][self.val_pairs[index][1]]
            x_t2 = self.data["text"][neg_target_index]
            fps_2=self.data['fps'][self.val_pairs[index][1]]
            x_url2 = self.data["url"][self.val_pairs[index][1]]
        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

        data = {
            "positive": {"audio": x_a1, "video": x_v1, "text": x_t1, "url": x_url1, "fps": fps_1},
            "negative": {"audio": x_a2, "video": x_v2, "text": x_t2, "url": x_url2, "fps": fps_2}
        }
        target = {
            "positive": positive_target,
            "negative": negative_target
        }
        return data, target


class DefaultCollator(object):
    def __init__(self, mode='fixed', max_len=60, trim='random', rate_video=16/25, rate_audio=0.96):
        self.mode = mode
        self.max_len = max_len
        self.rate_video = rate_video
        self.rate_audio = rate_audio
        self.trim = trim

    def get_max_seq_len(self, data):
        maxlen_audio_pos=0
        maxlen_audio_neg=0
        maxlen_video_pos=0
        maxlen_video_neg=0

        # get max sequence length in batch
        for element in data:
            positive=element['positive']
            negative=element['negative']
            positive_audio_size=positive['audio'].shape[0]
            positive_video_size=positive['video'].shape[0]
            negative_audio_size=negative['audio'].shape[0]
            negative_video_size=negative['video'].shape[0]
            if positive_audio_size > maxlen_audio_pos:
                maxlen_audio_pos=positive_audio_size
            if negative_audio_size > maxlen_audio_neg:
                maxlen_audio_neg=negative_audio_size
            if positive_video_size > maxlen_video_pos:
                maxlen_video_pos=positive_video_size
            if negative_video_size > maxlen_video_neg:
                maxlen_video_neg=negative_video_size

        return maxlen_audio_pos, maxlen_video_pos, maxlen_audio_neg, maxlen_video_neg

    def __call__(self, batch):
        data=[el[0] for el in batch]
        target=[el[1] for el in batch]
        batch_size=len(batch)

        if self.mode == 'max':
            len_audio_pos, len_video_pos, len_audio_neg, len_video_neg = self.get_max_seq_len(data)
        elif self.mode == 'fixed':
            # takes video features as anchor as more features per second
            len_video_pos, len_video_neg = self.max_len, self.max_len
            len_audio_pos, len_audio_neg = round(self.max_len *  self.rate_video * self.rate_audio), round(self.max_len *  self.rate_video * self.rate_audio)

        # init padding mask and timestep arrays
        mask_audio_pos = torch.ones((batch_size, len_audio_pos))
        mask_audio_neg=torch.ones((batch_size, len_audio_neg))
        mask_video_pos=torch.ones((batch_size, len_video_pos))
        mask_video_neg=torch.ones((batch_size, len_video_neg))

        timestep_audio_pos = np.repeat(np.expand_dims(np.arange(0, len_audio_pos, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_audio # audio has 0.96 features per second due to vggish
        timestep_audio_neg = np.repeat(np.expand_dims(np.arange(0, len_audio_neg, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_audio
        timestep_video_pos = np.repeat(np.expand_dims(np.arange(0, len_video_pos, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_video # video has 0.64 features per second
        timestep_video_neg = np.repeat(np.expand_dims(np.arange(0, len_video_neg, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_video

        for idx in range(len(data)):
            positive=data[idx]['positive']
            negative=data[idx]['negative']

            positive_audio_size = positive['audio'].shape[0]
            positive_video_size = positive['video'].shape[0]
            negative_audio_size = negative['audio'].shape[0]
            negative_video_size = negative['video'].shape[0]

            diff_pos_audio = positive_audio_size - len_audio_pos
            diff_neg_audio = negative_audio_size - len_audio_neg
            diff_pos_video = positive_video_size - len_video_pos
            diff_neg_video = negative_video_size - len_video_neg

            trim_start_video_pos = None
            trim_start_video_neg = None
            trim_start_audio_pos = None
            trim_start_audio_neg = None

            if diff_pos_video < 0:
                # padding
                mask_video_pos[idx, diff_pos_video:] = 0
                positive['video'] = np.pad(positive['video'], [(0, -diff_pos_video), (0, 0)])
            elif diff_pos_video > 0:
                if self.trim == 'random':
                    trim_start_video_pos = torch.randint(diff_pos_video, (1,))
                elif self.trim == 'center':
                    trim_start_video_pos = torch.tensor(diff_pos_video // 2)
                trim_start_audio_pos = min(diff_pos_audio, (trim_start_video_pos * self.rate_video / self.rate_audio).round().int()) if trim_start_video_pos > 0 else 0
                # trimming
                positive['video'] = positive['video'][trim_start_video_pos:trim_start_video_pos+len_video_pos]
            else:
                pass

            if diff_pos_audio < 0:
                # padding
                mask_audio_pos[idx, diff_pos_audio:] = 0
                positive['audio'] = np.pad(positive['audio'], [(0, -diff_pos_audio), (0,0)])
            elif diff_pos_audio > 0:
                if trim_start_audio_pos == None:
                    trim_start_audio_pos = 0
                # trimming
                positive['audio'] = positive['audio'][trim_start_audio_pos:trim_start_audio_pos+len_audio_pos]
            else:
                pass


            if diff_neg_video < 0:
                # padding
                mask_video_neg[idx, diff_neg_video:] = 0
                negative['video'] = np.pad(negative['video'], [(0, -diff_neg_video), (0, 0)])
            elif diff_neg_video > 0:
                if self.trim == 'random':
                    trim_start_video_neg = torch.randint(diff_neg_video, (1,))
                elif self.trim == 'center':
                    trim_start_video_neg = torch.tensor(diff_neg_video // 2)
                trim_start_audio_neg = min(diff_neg_audio, (trim_start_video_neg * self.rate_video / self.rate_audio).round().int()) if trim_start_video_neg > 0 else 0
                negative['video'] = negative['video'][trim_start_video_neg:trim_start_video_neg+len_video_neg]
                # trimming
            else:
                pass


            if diff_neg_audio < 0:
                # padding
                mask_audio_neg[idx, diff_neg_audio:] = 0
                negative['audio'] = np.pad(negative['audio'], [(0, -diff_neg_audio), (0, 0)])
            elif diff_neg_audio > 0:
                if trim_start_audio_neg == None:
                    trim_start_audio_neg = 0
                negative['audio'] =  negative['audio'][trim_start_audio_neg:trim_start_audio_neg+len_audio_neg]
                # trimming
            else:
                pass

            data[idx]['positive']=positive
            data[idx]['negative']=negative

        data_final={}
        target_final={}

        data_final['positive']={}
        data_final['positive']['audio']=torch.tensor([element['positive']['audio'] for element in data], dtype=torch.float32)
        data_final['positive']['video']=torch.tensor([element['positive']['video'] for element in data])
        data_final['positive']['video_mask']=mask_video_pos
        data_final['positive']['audio_mask']=mask_audio_pos
        data_final['positive']['text']=torch.tensor([element['positive']['text'] for element in data])
        data_final['positive']['url'] =[element['positive']['url'] for element in data]
        data_final['positive']['timestep']={'audio': timestep_audio_pos, 'video':timestep_video_pos}
        data_final['negative']={}
        data_final['negative']['audio'] = torch.tensor([element['negative']['audio'] for element in data], dtype=torch.float32)
        data_final['negative']['video'] = torch.tensor([element['negative']['video'] for element in data])
        data_final['negative']['audio_mask']=mask_audio_neg
        data_final['negative']['video_mask']=mask_video_neg
        data_final['negative']['text']=torch.tensor([element['negative']['text'] for element in data])
        data_final['negative']['url']=[element['positive']['url'] for element in data]
        data_final['negative']['timestep']={'audio':timestep_audio_neg, 'video':timestep_video_neg}

        target_final['positive']=torch.tensor([element['positive'] for element in target])
        target_final['negative']=torch.tensor([element['negative'] for element in target])

        return data_final, target_final


class UCFDataset(data.Dataset):

    @property
    def map_embeddings_target(self):
        w2v_embedding = torch.Tensor(self.data["text"]["data"])[sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def training_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_base{self.zero_shot_split}.pkl"

    @property
    def training_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_novel{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def train_val_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_base{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_1{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_2{self.zero_shot_split}.pkl"
    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    @property
    def test_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_1{self.zero_shot_split}.pkl"

    @property
    def test_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_2{self.zero_shot_split}.pkl"

    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": np.array(self.data["audio"]["data"])[classes_mask],
            "video": np.array(self.data["video"]["data"])[classes_mask],
            "text": np.array(self.data["text"]["data"])[sorted(self.classes.astype(int))],
            "target": np.array(self.data["audio"]["target"])[classes_mask],
            "url": np.array(self.data["audio"]["url"])[classes_mask],
            "fps": np.array(self.data['audio']['fps'])[classes_mask]
        }

    @property
    def features_processed_folder(self):
        return Path().cwd() / str("avgzsl_benchmark_non_averaged_datasets/"+self.embedding_type+"/UCF/_features_processed")

    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_base_train_class_names])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_base_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/ucf_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class.lower(): i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train_base":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "train_novel":
            return np.array([])
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_seen_class_names
        elif self.dataset_split == "train_val_base":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "train_val_novel" or self.dataset_split == "train_val_novel_1" or self.dataset_split == "train_val_novel_2":
            return np.array([])
        elif self.dataset_split == "train_stage_2":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "test" or self.dataset_split == "test_1" or self.dataset_split == "test_2":
            return self.stage_2_test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train_base":
            return np.array([])
        elif self.dataset_split == "train_novel":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_unseen_class_names
        elif self.dataset_split == "train_val_base":
            return np.array([])
        elif self.dataset_split == "train_val_novel" or self.dataset_split == "train_val_novel_1" or self.dataset_split == "train_val_novel_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "train_stage_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "test" or self.dataset_split == "test_1" or self.dataset_split =="test_2":
            return self.stage_2_test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.seen_class_names])

    @property
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.unseen_class_names])

    @property
    def stage_1_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_base_train.txt")

    @property
    def stage_1_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_novel_train.txt")

    @property
    def stage_1_val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def stage_1_val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def stage_2_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_base_train.txt")

    @property
    def stage_2_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_novel_train.txt")

    @property
    def stage_2_test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def stage_2_test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(UCFDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split
        self.embedding_type=args.embedding_type
        self.transform = transform
        self.text_embeddings_file=args.text_embeddings_file
        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return  self.training_novel_file.exists() and self.training_base_file.exists() and self.val_file.exists() and self.test_file.exists() and self.test_file_1.exists() and self.test_file_2.exists() and self.train_val_base_file.exists() and self.train_val_novel_file.exists() and self.train_val_novel_file_1.exists() and self.train_val_novel_file_2.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        train_val_base_set = self.read_dataset(dataset_type="train_val_base")
        with self.train_val_base_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_base_file}")
            pickle.dump(train_val_base_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_base_set


        training_set_base = self.read_dataset(dataset_type="train_base")
        with self.training_base_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_base_file}")
            pickle.dump(training_set_base, f, pickle.HIGHEST_PROTOCOL)
        del training_set_base
        training_set_novel = self.read_dataset(dataset_type="train_novel")
        with self.training_novel_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_novel_file}")
            pickle.dump(training_set_novel, f, pickle.HIGHEST_PROTOCOL)
        del training_set_novel
        val_set = self.read_dataset(dataset_type="val")
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        del val_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel")
        with self.train_val_novel_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_1")
        with self.train_val_novel_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_1}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_2")
        with self.train_val_novel_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_2}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set


        test_set = self.read_dataset(dataset_type="test")
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_1")
        with self.test_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_1}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_2")
        with self.test_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_2}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set



        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train_novel":
            data_file = self.training_novel_file
        elif self.dataset_split == "train_base":
            data_file = self.training_base_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val_base":
            data_file = self.train_val_base_file
        elif self.dataset_split == "train_val_novel":
            data_file = self.train_val_novel_file
        elif self.dataset_split == "train_val_novel_1":
            data_file=self.train_val_novel_file_1
        elif self.dataset_split == "train_val_novel_2":
            data_file=self.train_val_novel_file_2
        elif self.dataset_split == "test":
            data_file = self.test_file
        elif self.dataset_split=="test_1":
            data_file=self.test_file_1
        elif self.dataset_split=="test_2":
            data_file=self.test_file_2
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        result = {"data": [], "target": [], "url": [], 'fps':[]}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text" / self.text_embeddings_file).resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key].lower()] for key in
                                list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []

            if dataset_type=="train_base":
                split_names.append("stage_1_train_base")
            elif dataset_type=="train_novel":
                split_names.append("stage_1_train_novel")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type=="train_val_novel":
                split_names.append("stage_2_train_novel")
            elif dataset_type=="train_val_novel_1":
                split_names.append("stage_2_train_novel_1")
            elif dataset_type=="train_val_novel_2":
                split_names.append("stage_2_train_novel_2")
            elif dataset_type=="train_val_base":
                split_names.append("stage_1_train_base")
                split_names.append("stage_1_train_novel")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel")
            elif dataset_type=="test_1":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_1")
            elif dataset_type=="test_2":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_2")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url, fps = read_features(file)
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem.lower()])
                        result["url"].append(url[i])
                        result['fps'].append(fps[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = result["data"]
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result


class ActivityNetDataset(data.Dataset):

    @property
    def map_embeddings_target(self):
        w2v_embedding = torch.Tensor(self.data["text"]["data"])[sorted(self.classes)].cuda()
        sorted_classes = sorted(self.classes)
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def training_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_base{self.zero_shot_split}.pkl"

    @property
    def training_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training_novel{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def train_val_base_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_base{self.zero_shot_split}.pkl"

    @property
    def train_val_novel_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel{self.zero_shot_split}.pkl"
    @property
    def train_val_novel_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_1{self.zero_shot_split}.pkl"
    @property
    def train_val_novel_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val_novel_2{self.zero_shot_split}.pkl"
    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"
    @property
    def test_file_1(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_1{self.zero_shot_split}.pkl"
    @property
    def test_file_2(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test_2{self.zero_shot_split}.pkl"
    @property
    def targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.classes))[0]
        return {
            "audio": np.array(self.data["audio"]["data"])[classes_mask],
            "video": np.array(self.data["video"]["data"])[classes_mask],
            "text": np.array(self.data["text"]["data"])[sorted(self.classes.astype(int))],
            "target": np.array(self.data["audio"]["target"])[classes_mask],
            "url": np.array(self.data["audio"]["url"])[classes_mask],
            "fps": np.array(self.data['audio']['fps'])[classes_mask]
        }

    @property
    def features_processed_folder(self):
        return Path().cwd() / str("avgzsl_benchmark_non_averaged_datasets/"+self.embedding_type+"/ActivityNet/_features_processed")
    @property
    def all_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.all_class_names])

    @property
    def train_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_base_train_class_names])

    @property
    def val_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_val_seen_class_names])

    @property
    def val_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_1_val_unseen_class_names])

    @property
    def test_train_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_base_train_class_names])

    @property
    def test_seen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_test_seen_class_names])

    @property
    def test_unseen_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.stage_2_test_unseen_class_names])

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / "class-split/activitynet_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def classes(self):
        if self.zero_shot_split:
            return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

        else:
            if self.zero_shot_mode == "all":
                return self.all_class_ids
            elif self.zero_shot_mode == "seen":
                return self.seen_class_ids
            elif self.zero_shot_mode == "unseen":
                return self.unseen_class_ids
            else:
                raise AttributeError(f"Zero shot mode has to be either all, seen or unseen. Is {self.zero_shot_mode}")

    @property
    def class_to_idx(self):
        return {_class.lower(): i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def all_class_names(self):
        return get_class_names(self.root / "class-split/all_class.txt")

    @property
    def seen_class_names(self):
        if self.dataset_split == "train_base":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "train_novel":
            return np.array([])
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_base_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_seen_class_names
        elif self.dataset_split == "train_val_base":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split =="train_val_novel_2":
            return np.array([])
        elif self.dataset_split == "train_stage_2":
            return np.concatenate((self.stage_1_base_train_class_names, self.stage_1_val_unseen_class_names))
        elif self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split=="test_2":
            return self.stage_2_test_seen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train_base":
            return np.array([])
        elif self.dataset_split == "train_novel":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "train_stage_1":
            return self.stage_1_novel_train_class_names
        elif self.dataset_split == "val":
            return self.stage_1_val_unseen_class_names
        elif self.dataset_split == "train_val_base":
            return np.array([])
        elif self.dataset_split == "train_val_novel" or self.dataset_split=="train_val_novel_1" or self.dataset_split =="train_val_novel_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "train_stage_2":
            return self.stage_2_novel_train_class_names
        elif self.dataset_split == "test" or self.dataset_split=="test_1" or self.dataset_split =="test_2":
            return self.stage_2_test_unseen_class_names
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    # @lru_cache(maxsize=128)
    def seen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.seen_class_names])

    @property
    # @lru_cache(maxsize=128)
    def unseen_class_ids(self):
        return np.asarray([self.class_to_idx[name.lower()] for name in self.unseen_class_names])

    @property
    def stage_1_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_base_train.txt")

    @property
    def stage_1_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_novel_train.txt")

    @property
    def stage_1_val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def stage_1_val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def stage_2_base_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_base_train.txt")

    @property
    def stage_2_novel_train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_novel_train.txt")

    @property
    def stage_2_test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def stage_2_test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    def __init__(self, args, dataset_split, zero_shot_mode, download=False, transform=None):
        super(ActivityNetDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"Dataset split: {dataset_split}\t"
            f"Zero shot mode: {zero_shot_mode}")
        self.args = args
        self.root = args.root_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_mode = zero_shot_mode
        self.zero_shot_split = args.zero_shot_split
        self.embedding_type=args.embedding_type
        self.transform = transform
        self.text_embeddings_file=args.text_embeddings_file
        self.preprocess()
        self.data = self.get_data()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self.training_novel_file.exists() and self.training_base_file.exists() and self.val_file.exists() \
            and self.test_file.exists() and self.test_file_1.exists() and self.test_file_2.exists() \
            and self.train_val_base_file.exists() and self.train_val_novel_file.exists() and self.train_val_novel_file_1.exists() and self.train_val_novel_file_2.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        train_val_base_set = self.read_dataset(dataset_type="train_val_base")
        with self.train_val_base_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_base_file}")
            pickle.dump(train_val_base_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_base_set

        training_set_base = self.read_dataset(dataset_type="train_base")
        with self.training_base_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_base_file}")
            pickle.dump(training_set_base, f, pickle.HIGHEST_PROTOCOL)
        del training_set_base
        training_set_novel = self.read_dataset(dataset_type="train_novel")
        with self.training_novel_file.open("wb") as f:
            self.logger.info(f"Dumping to {self.training_novel_file}")
            pickle.dump(training_set_novel, f, pickle.HIGHEST_PROTOCOL)
        del training_set_novel
        val_set = self.read_dataset(dataset_type="val")
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        del val_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel")
        with self.train_val_novel_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_1")
        with self.train_val_novel_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_1}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        train_val_novel_set = self.read_dataset(dataset_type="train_val_novel_2")
        with self.train_val_novel_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_novel_file_2}")
            pickle.dump(train_val_novel_set, f, pickle.HIGHEST_PROTOCOL)
        del train_val_novel_set

        test_set = self.read_dataset(dataset_type="test")
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_1")
        with self.test_file_1.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_1}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        test_set = self.read_dataset(dataset_type="test_2")
        with self.test_file_2.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file_2}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        del test_set

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train_novel":
            data_file = self.training_novel_file
        elif self.dataset_split == "train_base":
            data_file = self.training_base_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val_base":
            data_file = self.train_val_base_file
        elif self.dataset_split == "train_val_novel":
            data_file = self.train_val_novel_file
        elif self.dataset_split == "train_val_novel_1":
            data_file = self.train_val_novel_file_1
        elif self.dataset_split == "train_val_novel_2":
            data_file = self.train_val_novel_file_2
        elif self.dataset_split == "test":
            data_file = self.test_file
        elif self.dataset_split == "test_1":
            data_file = self.test_file_1
        elif self.dataset_split == "test_2":
            data_file = self.test_file_2
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        result = {"data": [], "target": [], "url": [], "fps":[]}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / "text" / self.text_embeddings_file).resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key].lower()] for key in
                                list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train_base":
                split_names.append("stage_1_train_base")
            elif dataset_type == "train_novel":
                split_names.append("stage_1_train_novel")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type == "train_val_novel":
                split_names.append("stage_2_train_novel")
            elif dataset_type == "train_val_novel_1":
                split_names.append("stage_2_train_novel_1")
            elif dataset_type == "train_val_novel_2":
                split_names.append("stage_2_train_novel_2")
            elif dataset_type == "train_val_base":
                split_names.append("stage_1_train_base")
                split_names.append("stage_1_train_novel")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_novel")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel")
            elif dataset_type == "test_1":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_1")
            elif dataset_type == "test_2":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_novel_2")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url, fps = read_features(file)
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem.lower()])
                        result["url"].append(url[i])
                        result["fps"].append(fps[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = result["data"]
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result
