import logging
from tqdm import tqdm
import torch
import copy
from src.metrics import MeanClassAccuracy
from src.utils import check_best_loss, check_best_score, save_best_model


def train(
    train_loader,
    val_loader,
    model,
    optimizer,
    epochs,
    device,
    writer,
    metrics,
    train_stats,
    val_stats,
    log_dir,
    which,
    which_stage,
    dictionary,
    lr_reduction_epoch,
    iteration_test,
    args,
):
    best_loss = None
    best_score = None
    best_epoch = None
    best_model = None
    beta = None
    lr_epochs_reduction = []

    for epoch in range(epochs):

        if train_loader[1] == None:
            train_loss = train_step(
                train_loader[0],
                model,
                epoch,
                epochs,
                writer,
                device,
                metrics,
                train_stats,
                dictionary,
                args,
            )
        else:
            train_loss = train_step_two_dataloaders(
                train_loader[0],
                train_loader[1],
                model,
                epoch,
                epochs,
                writer,
                device,
                metrics,
                train_stats,
                dictionary,
                args,
            )

        if which_stage == "stage_1":
            val_loss, val_score, beta = val_step(
                val_loader,
                model,
                epoch,
                epochs,
                writer,
                device,
                metrics,
                val_stats,
                which,
                dictionary,
                args,
            )
            best_loss = check_best_loss(
                epoch, best_loss, val_loss, model, optimizer, log_dir
            )
            best_score, best_epoch, best_model_returned = check_best_score(
                epoch, best_score, best_epoch, val_score, model, optimizer, log_dir
            )
            if best_model_returned != None:
                best_model = best_model_returned
        elif which_stage == "stage_2":
            val_loss = best_loss = 0
            val_score = best_score = 0
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if args.save_checkpoints:
            save_best_model(
                epoch,
                val_score,
                model,
                optimizer,
                log_dir / ("checkpoints" + str(iteration_test)),
                metric="score",
                checkpoint=True,
            )

        logger = logging.getLogger()
        if which_stage == "stage_1":
            decreased_lr = model.optimize_scheduler(val_score)
            if decreased_lr == True:
                lr_epochs_reduction.append(epoch)
            logger.info(f"decreased lr : {decreased_lr}\t")
        else:
            if epoch in lr_reduction_epoch:
                logger.info("reduce lr manually")
                model.reduce_lr_manually(epoch)

    return best_loss, best_score, best_epoch, best_model, lr_epochs_reduction, beta


def add_loss_details(current_loss_details, batch_loss_details):
    for key, value in current_loss_details.items():
        if key not in batch_loss_details:
            batch_loss_details[key] = value
        else:
            batch_loss_details[key] += value
    return batch_loss_details


def add_logs_tensorboard(batch_loss_details, writer, batch_idx, step, which_stage):

    writer.add_scalar(
        f"Loss/total_loss_" + which_stage,
        batch_loss_details["Loss/total_loss"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/loss_generator_" + which_stage,
        batch_loss_details["Loss/loss_generator"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/loss_discriminator_" + which_stage,
        batch_loss_details["Loss/loss_discriminator"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/cross_entropy_" + which_stage,
        batch_loss_details["Loss/cross_entropy"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/loss_dicriminator_seen_novel_" + which_stage,
        batch_loss_details["Loss/loss_discriminator_seen_novel"] / (batch_idx),
        step,
    )


def add_logs_tensorboard_baselines(
    batch_loss_details, writer, batch_idx, step, which_stage
):

    writer.add_scalar(
        f"Loss/total_loss_" + which_stage,
        batch_loss_details["Loss/total_loss"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/loss_reg_" + which_stage,
        batch_loss_details["Loss/loss_reg"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/loss_cmd_rec_" + which_stage,
        batch_loss_details["Loss/loss_cmd_rec"] / (batch_idx),
        step,
    )
    writer.add_scalar(
        f"Loss/cross_entropy_" + which_stage,
        batch_loss_details["Loss/cross_entropy"] / (batch_idx),
        step,
    )


def train_step(
    data_loader,
    model,
    epoch,
    epochs,
    writer,
    device,
    metrics,
    stats,
    dictionary,
    args,
):
    logger = logging.getLogger()
    model.train()

    for metric in metrics:
        metric.reset()
    mapping_dict = dictionary[1]
    batch_loss = 0
    batch_loss_details = {}
    for batch_idx, (data, target) in tqdm(enumerate(data_loader)):

        p = data["positive"]
        q = data["negative"]

        x_p_a = p["audio"].to(device)
        x_p_v = p["video"].to(device)
        x_p_t = p["text"].to(device)
        x_p_num = target["positive"].to(device)

        masks = {}
        masks["positive"] = {"audio": p["audio_mask"], "video": p["video_mask"]}

        timesteps = {}
        timesteps["positive"] = {
            "audio": p["timestep"]["audio"],
            "video": p["timestep"]["video"],
        }

        inputs = (
            x_p_a,
            x_p_v,
            x_p_num,
            x_p_t,
            masks["positive"],
            timesteps["positive"],
        )

        if args.z_score_inputs:
            inputs = tuple(
                [(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs]
            )

        if args.cross_entropy_loss == True:
            for i in range(inputs[2].shape[0]):
                inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]
        loss, loss_details = model.optimize_params(
            *inputs,
            optimize=True,
            split="base_st_1",
        )
        batch_loss_details = add_loss_details(loss_details, batch_loss_details)
        audio_emb, video_emb, emb_cls = model.get_embeddings(
            inputs[0], inputs[1], inputs[3], inputs[4], inputs[5]
        )
        outputs = video_emb

        batch_loss += loss.item()

        p_target = target["positive"].to(device)
        q_target = target["negative"].to(device)

        # stats
        iteration = len(data_loader) * epoch + batch_idx
        if iteration % len(data_loader) == 0:
            for metric in metrics:
                if isinstance(metric, MeanClassAccuracy):
                    continue
                metric(
                    outputs, (p_target, q_target), (loss, loss_details), mapping_dict
                )
                for key, value in metric.value().items():
                    if "recall" in key:
                        continue
                    writer.add_scalar(f"train_{key}", value, iteration)

    batch_loss /= batch_idx + 1
    stats.update((epoch, batch_loss, None))

    add_logs_tensorboard(
        batch_loss_details,
        writer,
        (batch_idx + 1),
        len(data_loader) * (epoch + 1),
        "train",
    )

    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss


def train_step_two_dataloaders(
    data_loader,
    data_loader1,
    model,
    epoch,
    epochs,
    writer,
    device,
    metrics,
    stats,
    dictionary,
    args,
):
    logger = logging.getLogger()
    model.train()

    for metric in metrics:
        metric.reset()
    mapping_dict1 = dictionary[1]
    batch_loss = 0
    batch_loss_details = {}
    for batch_idx, ((data1, target1), (data2, target2)) in enumerate(
        zip(data_loader, data_loader1)
    ):

        p_data1 = data1["positive"]
        p_data2 = data2["positive"]

        x_p_a_data1 = p_data1["audio"].to(device)
        x_p_v_data1 = p_data1["video"].to(device)
        x_p_t_data1 = p_data1["text"].to(device)
        x_p_num_data1 = target1["positive"].to(device)

        x_p_a_data2 = p_data2["audio"].to(device)
        x_p_v_data2 = p_data2["video"].to(device)
        x_p_t_data2 = p_data2["text"].to(device)
        x_p_num_data2 = target2["positive"].to(device)

        masks_data1 = {}
        masks_data1["positive"] = {
            "audio": p_data1["audio_mask"],
            "video": p_data1["video_mask"],
        }

        masks_data2 = {}
        masks_data2["positive"] = {
            "audio": p_data2["audio_mask"],
            "video": p_data2["video_mask"],
        }

        timesteps_data1 = {}
        timesteps_data1["positive"] = {
            "audio": p_data1["timestep"]["audio"],
            "video": p_data1["timestep"]["video"],
        }

        timesteps_data2 = {}
        timesteps_data2["positive"] = {
            "audio": p_data2["timestep"]["audio"],
            "video": p_data2["timestep"]["video"],
        }

        inputs_data1 = (
            x_p_a_data1,
            x_p_v_data1,
            x_p_num_data1,
            x_p_t_data1,
            masks_data1["positive"],
            timesteps_data1["positive"],
        )

        inputs_data2 = (
            x_p_a_data2,
            x_p_v_data2,
            x_p_num_data2,
            x_p_t_data2,
            masks_data2["positive"],
            timesteps_data2["positive"],
        )

        if args.z_score_inputs:
            inputs_data1 = tuple(
                [(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs_data1]
            )
            inputs_data2 = tuple(
                [(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs_data2]
            )

        if args.cross_entropy_loss == True:
            for i in range(inputs_data1[2].shape[0]):
                inputs_data1[2][i] = mapping_dict1[(inputs_data1[2][[i]]).item()]

            for i in range(inputs_data2[2].shape[0]):
                inputs_data2[2][i] = mapping_dict1[(inputs_data2[2][[i]]).item()]

        loss1, loss_details1 = model.optimize_params(
            *inputs_data1,
            optimize=True,
            split="base",
        )
        loss2, loss_details2 = model.optimize_params(
            *inputs_data2,
            optimize=True,
            split="novel",
        )

        for key, value in loss_details1.items():
            loss_details1[key] = loss_details1[key] + loss_details2[key]

        batch_loss_details = add_loss_details(loss_details1, batch_loss_details)
        audio_emb, video_emb, emb_cls = model.get_embeddings(
            inputs_data1[0],
            inputs_data1[1],
            inputs_data1[3],
            inputs_data1[4],
            inputs_data1[5],
        )
        audio_emb2, video_emb2, emb_cls2 = model.get_embeddings(
            inputs_data2[0],
            inputs_data2[1],
            inputs_data2[3],
            inputs_data2[4],
            inputs_data2[5],
        )

        outputs = video_emb

        outputs2 = video_emb2

        outputs = torch.cat((outputs, outputs2))

        batch_loss += loss1.item() + loss2.item()

        p_target1 = target1["positive"].to(device)
        q_target1 = target1["negative"].to(device)

        p_target2 = target2["positive"].to(device)
        q_target2 = target2["negative"].to(device)

        p_target1 = torch.cat((p_target1, p_target2))
        q_target1 = torch.cat((q_target1, q_target2))

        # stats
        iteration = len(data_loader) * epoch + batch_idx
        if iteration % len(data_loader) == 0:
            for metric in metrics:
                if isinstance(metric, MeanClassAccuracy):
                    continue
                metric(
                    outputs,
                    (p_target1, q_target1),
                    (loss1, loss_details1),
                    mapping_dict1,
                )
                for key, value in metric.value().items():
                    if "recall" in key:
                        continue
                    writer.add_scalar(f"train_{key}", value, iteration)

    batch_loss /= batch_idx + 1
    stats.update((epoch, batch_loss, None))

    add_logs_tensorboard(
        batch_loss_details,
        writer,
        (batch_idx + 1),
        len(data_loader) * (epoch + 1),
        "train",
    )

    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"Loss: {batch_loss:.4f}\t"
    )
    return batch_loss


def val_step(
    data_loader,
    model,
    epoch,
    epochs,
    writer,
    device,
    metrics,
    stats,
    which,
    dictionary,
    args=None,
):

    logger = logging.getLogger()
    model.eval()

    for metric in metrics:
        metric.reset()
    mapping_dict = dictionary[1]

    novel_classes_list = []
    seen_classes_list = []
    for unseen_class in data_loader.dataset.zsl_dataset.unseen_class_ids:
        unseen_classes = mapping_dict[unseen_class]
        novel_classes_list.append(unseen_classes)
    for seen_class in data_loader.dataset.zsl_dataset.seen_class_ids:
        seen_classes = mapping_dict[seen_class]
        seen_classes_list.append(seen_classes)

    with torch.no_grad():
        batch_loss = 0
        hm_score = 0
        seen_score = 0
        unseen_score = 0
        batch_loss_details = {}
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            p = data["positive"]
            q = data["negative"]

            x_p_a = p["audio"].to(device)
            x_p_v = p["video"].to(device)
            x_p_t = p["text"].to(device)
            x_p_num = target["positive"].to(device)

            masks = {}
            masks["positive"] = {"audio": p["audio_mask"], "video": p["video_mask"]}

            timesteps = {}
            timesteps["positive"] = {
                "audio": p["timestep"]["audio"],
                "video": p["timestep"]["video"],
            }

            inputs = (
                x_p_a,
                x_p_v,
                x_p_num,
                x_p_t,
                masks["positive"],
                timesteps["positive"],
            )

            if args.z_score_inputs:
                inputs = tuple(
                    [(x - torch.mean(x)) / torch.sqrt(torch.var(x)) for x in inputs]
                )

            if args.cross_entropy_loss == True:
                for i in range(inputs[2].shape[0]):
                    inputs[2][i] = mapping_dict[(inputs[2][[i]]).item()]

            loss, loss_details = model.optimize_params(
                *inputs,
                optimize=False,
                split="val",
            )
            batch_loss_details = add_loss_details(loss_details, batch_loss_details)
            audio_emb, video_emb, emb_cls = model.get_embeddings(
                inputs[0], inputs[1], inputs[3], inputs[4], inputs[5]
            )
            outputs = (video_emb, emb_cls)

            batch_loss += loss.item()

            p_target = target["positive"].to(device)
            q_target = target["negative"].to(device)

            # stats
            iteration = len(data_loader) * epoch + batch_idx
            if iteration % len(data_loader) == 0:
                for metric in metrics:
                    metric(
                        outputs,
                        (p_target, q_target),
                        (loss, loss_details),
                        mapping_dict,
                    )
                    """
                    logger.info(
                        f"{metric.name()}: {metric.value()}"
                    )
                    """
                    for key, value in metric.value().items():
                        if "recall" in key:
                            continue
                        if "both_hm" in key:
                            hm_score = value
                            writer.add_scalar(f"metric_val/{key}", value, iteration)
                        if "both_zsl" in key:
                            zsl_score = value
                            writer.add_scalar(f"metric_val/{key}", value, iteration)
                        if "both_beta" in key:
                            beta = value
                        if "both_seen" in key:
                            seen_score = value
                            writer.add_scalar(f"metric_val/{key}", value, iteration)
                        if "both_unseen" in key:
                            unseen_score = value
                            writer.add_scalar(f"metric_val/{key}", value, iteration)
                        if "both_discriminator_seen" in key:
                            writer.add_scalar(f"metric_val/{key}", value, iteration)
                        if "both_discriminator_unseen" in key:
                            writer.add_scalar(f"metric_val/{key}", value, iteration)

        batch_loss /= batch_idx + 1
        if which == "base":
            stats.update((epoch, batch_loss, seen_score))
        else:
            stats.update((epoch, batch_loss, hm_score))

        add_logs_tensorboard(
            batch_loss_details,
            writer,
            (batch_idx + 1),
            len(data_loader) * (epoch + 1),
            "val",
        )

        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"Loss: {batch_loss:.4f}\t"
            f"FSL score: {zsl_score:.4f}\t"
            f"Base score: {seen_score:.4f}\t"
            f"Novel score:{unseen_score:.4f}\t"
            f"HM: {hm_score:.4f}\t"
            f"Beta: {beta:.4f}\t"
        )
    if which == "base":
        return batch_loss, seen_score, None
    else:
        return batch_loss, hm_score, beta
