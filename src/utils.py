from copy import deepcopy
import pathlib
import yaml
import logging
import pickle
import socket
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import subprocess
import os
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter

from src.logger import PD_Stats, create_logger


def read_features(path):

    with open(path, "rb") as f:
        x = pickle.load(f)

    data = x["features"]
    fps = x["fps"]
    url = [str(u) for u in list(x["video_names"])]

    return data, url, fps


def fix_seeds(seed=42):

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_model_size(model, logger):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    logger.info(
        "Created network [%s] with total number of parameters: %.1f million."
        % (type(model).__name__, num_params / 1000000)
    )


def get_git_revision_hash():
    try:
        hash_string = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        hash_string = ""
    return hash_string


def get_active_branch_name():

    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def dump_config_yaml(args, exp_dir):
    args_dict = deepcopy(vars(args))
    for k, v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()

    with open((exp_dir / "args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f)


def log_hparams(writer, args, metrics):
    args_dict = vars(args)
    for k, v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()
    if "recall" in metrics:
        del metrics["recall"]
    metrics = {"Eval/" + k: v for k, v in metrics.items()}
    writer.add_hparams(args_dict, metrics)


def setup_experiment(args, *stats):
    if args.exp_name == "":
        exp_name = (
            f"{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"
        )
    else:
        exp_name = (
            str(args.exp_name)
            + f"_{datetime.now().strftime('%b%d_%H-%M-%S_%f')}_{socket.gethostname()}"
        )

    exp_dir = args.log_dir / exp_name
    exp_dir.mkdir(parents=True)

    (exp_dir / "checkpoints").mkdir()
    (exp_dir / "checkpoints_1").mkdir()
    (exp_dir / "checkpoints_2").mkdir()
    pickle.dump(args, (exp_dir / "args.pkl").open("wb"))

    dump_config_yaml(args, exp_dir)

    train_stats = PD_Stats(exp_dir / "train_stats.pkl", stats)
    val_stats = PD_Stats(exp_dir / "val_stats.pkl", stats)

    logger = create_logger(exp_dir / "train.log")

    logger.info(f"Start experiment {exp_name}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"The experiment will be stored in {exp_dir.resolve()}\n")
    logger.info("")

    writer = SummaryWriter(log_dir=exp_dir)
    return logger, exp_dir, writer, train_stats, val_stats


def setup_evaluation(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(
            f"{k}: {str(v)}"
            for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items())
        )
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer


def save_best_model(
    epoch, best_metric, model, optimizer, log_dir, metric="", checkpoint=False
):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    if optimizer is None:
        optimizer = model.optimizer_gen
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric,
    }
    if checkpoint:
        torch.save(
            save_dict, log_dir / f"{model.__class__.__name__}_{metric}_ckpt_{epoch}.pt"
        )
    else:
        torch.save(save_dict, log_dir / f"{model.__class__.__name__}_{metric}.pt")


def check_best_loss(epoch, best_loss, val_loss, model, optimizer, log_dir):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, metric="loss")
        return val_loss
    if val_loss < best_loss:
        best_loss = val_loss
        save_best_model(epoch, best_loss, model, optimizer, log_dir, metric="loss")
    return best_loss


def check_best_score(
    epoch, best_score, best_epoch, hm_score, model, optimizer, log_dir
):
    if not best_score:
        save_best_model(epoch, hm_score, model, optimizer, log_dir, metric="score")
        return hm_score, epoch, copy.deepcopy(model)
    if hm_score > best_score:
        best_score = hm_score
        best_epoch = epoch
        save_best_model(epoch, best_score, model, optimizer, log_dir, metric="score")
        best_model = copy.deepcopy(model)
    else:
        best_model = None
    return best_score, best_epoch, best_model


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if "module." in name:
            name = name.replace("module.", "")
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def evaluate_dataset_baseline(
    dataset_tuple,
    model,
    device,
    best_beta=None,
    dictionary=None,
    args=None,
    save_performances=False,
    which=None,
):

    dataset = dataset_tuple[0]
    data_loader = dataset_tuple[1]
    data_t = torch.tensor(dataset.all_data["text"]).to(device)
    accumulated_calibration_values = []
    accumulated_video_emb = []
    accumulated_data_num = []

    mapping_dict = dictionary

    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
        data_a = data["positive"]["audio"].to(device)
        data_v = data["positive"]["video"].to(device)
        data_num = target["positive"].to(device)
        masks = {}
        masks["positive"] = {
            "audio": data["positive"]["audio_mask"],
            "video": data["positive"]["video_mask"],
        }
        timesteps = {}
        timesteps["positive"] = {
            "audio": data["positive"]["timestep"]["audio"],
            "video": data["positive"]["timestep"]["video"],
        }

        all_data = (
            data_a,
            data_v,
            data_num,
            data_t,
            masks["positive"],
            timesteps["positive"],
        )
        try:
            if args.z_score_inputs:
                all_data = tuple(
                    [(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in all_data]
                )
        except AttributeError:
            print("Namespace has no fitting attribute. Continuing")

        model.eval()
        with torch.no_grad():
            for i in range(all_data[2].shape[0]):
                all_data[2][i] = mapping_dict[(all_data[2][[i]]).item()]
            calibration_value, video_emb, emb_cls = model.get_embeddings(
                all_data[0], all_data[1], all_data[3], all_data[4], all_data[5]
            )
            accumulated_calibration_values.append(calibration_value)
            accumulated_video_emb.append(video_emb)
            outputs_all = (calibration_value, video_emb, emb_cls)

        accumulated_data_num.append(all_data[2])

    stacked_calibration_values = torch.cat(accumulated_calibration_values)
    stacked_video_emb = torch.cat(accumulated_video_emb)
    data_num = torch.cat(accumulated_data_num)
    emb_cls = outputs_all[2]
    outputs_all = (stacked_calibration_values, stacked_video_emb, emb_cls)

    calibration_value, v_p, t_p = outputs_all

    video_evaluation = get_best_evaluation(
        dataset,
        data_num,
        v_p,
        t_p,
        device=device,
        best_beta=best_beta,
        save_performances=save_performances,
        which=which,
    )

    return {
        "audio": video_evaluation,
        "video": video_evaluation,
        "both": video_evaluation,
    }


def get_best_evaluation(
    dataset,
    targets,
    v_p,
    t_p,
    device,
    best_beta=None,
    save_performances=False,
    which=None,
):
    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []
    start = 0
    if which == "base":
        end = 0
    else:
        end = 1
    steps = (end - start) * 10 + 1
    betas = (
        torch.tensor([best_beta], dtype=torch.float, device=device)
        if best_beta
        else torch.linspace(start, end, steps, device=device)
    )

    embeddings_ordered, mapping_dict = dataset.map_embeddings_target

    seen_label_array = torch.tensor(
        dataset.seen_class_ids, dtype=torch.long, device=device
    )

    for i in range(seen_label_array.shape[0]):
        seen_label_array[i] = mapping_dict[(seen_label_array[[i]]).item()]

    unseen_label_array = torch.tensor(
        dataset.unseen_class_ids, dtype=torch.long, device=device
    )

    for i in range(unseen_label_array.shape[0]):
        unseen_label_array[i] = mapping_dict[(unseen_label_array[[i]]).item()]

    seen_unseen_array = torch.tensor(
        np.sort(np.concatenate((dataset.seen_class_ids, dataset.unseen_class_ids))),
        dtype=torch.long,
        device=device,
    )

    for i in range(seen_unseen_array.shape[0]):
        seen_unseen_array[i] = mapping_dict[(seen_unseen_array[[i]]).item()]

    classes_embeddings = t_p
    with torch.no_grad():
        for beta in betas:

            distance_mat = (
                torch.zeros(
                    (v_p.shape[0], len(dataset.all_class_ids)),
                    dtype=torch.float,
                    device=device,
                )
                - 99999999999999
            )
            distance_mat_zsl = (
                torch.zeros(
                    (v_p.shape[0], len(dataset.all_class_ids)),
                    dtype=torch.float,
                    device=device,
                )
                - 99999999999999
            )
            # L2
            distance_mat[:, seen_unseen_array] = v_p  # .pow(2)
            mask = torch.zeros(
                len(dataset.all_class_ids), dtype=torch.long, device=device
            )
            mask[seen_label_array] = -99999999999999
            distance_mat_zsl = distance_mat + mask

            mask = (
                torch.zeros(len(dataset.all_class_ids), dtype=torch.long, device=device)
                - beta
            )
            mask[unseen_label_array] = 0

            neighbor_batch = torch.argmax(distance_mat + mask, dim=1)

            match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(
                neighbor_batch[match_idx], minlength=len(dataset.all_class_ids)
            )[seen_unseen_array]
            target_counts = torch.bincount(
                targets, minlength=len(dataset.all_class_ids)
            )[seen_unseen_array]
            per_class_recall = torch.zeros(
                len(dataset.all_class_ids), dtype=torch.float, device=device
            )
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            seen_recall_dict = per_class_recall[seen_label_array]
            unseen_recall_dict = per_class_recall[unseen_label_array]
            s = seen_recall_dict.mean()
            u = unseen_recall_dict.mean()

            if save_performances:
                seen_dict = {
                    k: v
                    for k, v in zip(
                        np.array(dataset.all_class_names)[
                            seen_label_array.cpu().numpy()
                        ],
                        seen_recall_dict.cpu().numpy(),
                    )
                }
                unseen_dict = {
                    k: v
                    for k, v in zip(
                        np.array(dataset.all_class_names)[
                            unseen_label_array.cpu().numpy()
                        ],
                        unseen_recall_dict.cpu().numpy(),
                    )
                }
                save_class_performances(seen_dict, unseen_dict, dataset.dataset_name)

            hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)

            neighbor_batch_zsl = torch.argmax(distance_mat_zsl, dim=1)
            match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(
                neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_ids)
            )[seen_unseen_array]
            target_counts = torch.bincount(
                targets, minlength=len(dataset.all_class_ids)
            )[seen_unseen_array]
            per_class_recall = torch.zeros(
                len(dataset.all_class_ids), dtype=torch.float, device=device
            )
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            zsl = per_class_recall[unseen_label_array].mean()

            zsl_scores.append(zsl.item())
            seen_scores.append(s.item())
            unseen_scores.append(u.item())
            hm_scores.append(hm.item())
            per_class_recalls.append(per_class_recall.tolist())
        argmax_hm = np.argmax(hm_scores)
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]
        best_beta = betas[argmax_hm].item()
        correct_discriminator_seen = 0
        correct_discriminator_unseen = 0

    return {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta,
        "correct_seen": correct_discriminator_seen,
        "correct_unseen": correct_discriminator_unseen,
    }


def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch


def save_class_performances(seen_dict, unseen_dict, dataset_name):
    seen_path = Path(
        f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_seen.pkl"
    )
    unseen_path = Path(
        f"doc/cvpr2022/fig/final/class_performance_{dataset_name}_unseen.pkl"
    )
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")
