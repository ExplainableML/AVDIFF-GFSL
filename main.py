import sys
import copy
import gc
import numpy as np
from torch.utils import data
from src.loss import L2Loss
from src.args import args_main
from get_evaluation import get_evaluation

from src.dataset import (
    ActivityNetDataset,
    ContrastiveDataset,
    VGGSoundDataset,
    UCFDataset,
)
from src.dataset import DefaultCollator
from src.metrics import (
    MeanClassAccuracy,
)
from src.AVDiff import AVDiff
from src.sampler import SamplerFactory
from src.train import train
from src.utils import (
    fix_seeds,
    setup_experiment,
    get_git_revision_hash,
    print_model_size,
    log_hparams,
    get_active_branch_name,
)
from src.utils_improvements import get_model_params


def get_dictionary(which_stage, which_dataset, args):
    if which_stage == "stage_1":
        if which_dataset == "VGGSound":
            val_all_dataset = VGGSoundDataset(
                args=args,
                dataset_split="val",
                zero_shot_mode=None,
            )
        elif which_dataset == "UCF":
            val_all_dataset = UCFDataset(
                args=args,
                dataset_split="val",
                zero_shot_mode=None,
            )
        elif which_dataset == "ActivityNet":
            val_all_dataset = ActivityNetDataset(
                args=args,
                dataset_split="val",
                zero_shot_mode=None,
            )
        else:
            raise NotImplementedError()
        embeddings, dictionary = val_all_dataset.map_embeddings_target

    elif which_stage == "stage_2":
        if which_dataset == "VGGSound":
            test = VGGSoundDataset(
                args=args,
                dataset_split="test",
                zero_shot_mode=None,
            )
        elif which_dataset == "UCF":
            test = UCFDataset(
                args=args,
                dataset_split="test",
                zero_shot_mode=None,
            )
        elif which_dataset == "ActivityNet":
            test = ActivityNetDataset(
                args=args,
                dataset_split="test",
                zero_shot_mode=None,
            )
        else:
            raise NotImplementedError()
        embeddings, dictionary = test.map_embeddings_target
    return [embeddings, dictionary]


def run():
    args, eval_args = args_main()

    run_mode = args.run
    best_epoch = None
    best_epoch_finetuned = None
    if run_mode == "stage-1" or run_mode == "all":
        args.retrain_all = False
        args.save_checkpoints = False

        dictionary_stage_1 = get_dictionary("stage_1", args.dataset_name, args)
        args.logits_length = len(dictionary_stage_1[1])

        (
            path_stage_1,
            best_epoch,
            model,
            setuped_experiment,
            lr_reduction_epoch_base,
            _,
        ) = main(args, "base", None, None, "stage_1", dictionary_stage_1, None, None)
        assert model != None
        (
            path_stage_1,
            best_epoch_finetuned,
            model,
            setuped_experiment,
            lr_reduction_epoch_full,
            beta,
        ) = main(
            args,
            "full",
            model,
            setuped_experiment,
            "stage_1",
            dictionary_stage_1,
            None,
            None,
        )
        eval_args.load_path_stage_A = path_stage_1

    if run_mode == "stage-2" or run_mode == "all":

        list_results = []
        args.retrain_all = True
        dictionary_stage_2 = get_dictionary("stage_2", args.dataset_name, args)
        args.logits_length = len(dictionary_stage_2[1])

        args.epochs = best_epoch + 1
        args.epochs_fine_tune = best_epoch_finetuned + 1
        path_stage_2, _, model, setuped_experiment, _, _ = main(
            args,
            "base",
            None,
            None,
            "stage_2",
            dictionary_stage_2,
            lr_reduction_epoch_base,
            None,
        )
        assert model != None
        args.save_checkpoints = True
        old_model = copy.deepcopy(model)
        for counter in range(3):

            if counter == 0:
                iteration_test = ""
            elif counter == 1:
                iteration_test = "_1"
            elif counter == 2:
                iteration_test = "_2"

            path_stage_2, _, _, setuped_experiment, _, _ = main(
                args,
                "full",
                copy.deepcopy(old_model),
                setuped_experiment,
                "stage_2",
                dictionary_stage_2,
                lr_reduction_epoch_full,
                iteration_test,
            )
            eval_args.load_path_stage_B = path_stage_2

            if run_mode == "eval" or run_mode == "all":
                assert eval_args.load_path_stage_A != None
                assert eval_args.load_path_stage_B != None
                results_evaluation, logger, tb_vizualization = get_evaluation(
                    eval_args,
                    dictionary_stage_1,
                    dictionary_stage_2,
                    iteration_test,
                    beta,
                )
                list_results.append(results_evaluation)

    seen_average = 0
    unseen_average = 0
    zsl_average = 0

    for result in list_results:
        seen_average += result["both"]["seen"]
        unseen_average += result["both"]["unseen"]
        zsl_average += result["both"]["zsl"]

    seen_average = seen_average / 3
    unseen_average = unseen_average / 3
    zsl_average = zsl_average / 3
    hm_average = (2 * seen_average * unseen_average) / (
        seen_average + unseen_average + np.finfo(float).eps
    )

    logger.info("Finished training")
    logger.info(
        rf"""Base performance={100 * seen_average:.2f}, Novel performance={100 * unseen_average:.2f}, GFSL performance={100 * hm_average:.2f}, FSL performance={100 * zsl_average:.2f} """
    )

    dict_average = {}
    dict_average["seen"] = seen_average
    dict_average["unseen"] = unseen_average
    dict_average["zsl"] = zsl_average
    dict_average["hm"] = hm_average

    log_hparams(tb_vizualization[0], tb_vizualization[1], dict_average)


def main(
    args,
    which,
    model,
    setuped_experiment,
    which_stage,
    dictionary,
    lr_reduction_epoch,
    iteration_test,
):

    if args.input_size is not None:
        args.input_size_audio = args.input_size
        args.input_size_video = args.input_size
    fix_seeds(args.seed)

    if model == None:
        logger, log_dir, writer, train_stats, val_stats = setup_experiment(
            args, "epoch", "loss", "hm"
        )
    else:
        logger, log_dir, writer, train_stats, val_stats = setuped_experiment

    logger.info("Git commit hash: " + get_git_revision_hash())
    logger.info("Current branch is:" + get_active_branch_name())

    if args.dataset_name == "VGGSound":
        if args.retrain_all == False:
            train_base = VGGSoundDataset(
                args=args,
                dataset_split="train_base",
                zero_shot_mode="train",
            )
            if which == "full":
                train_novel = VGGSoundDataset(
                    args=args,
                    dataset_split="train_novel",
                    zero_shot_mode="train",
                )
        if args.retrain_all == True:
            train_base = VGGSoundDataset(
                args=args,
                dataset_split="train_val_base",
                zero_shot_mode=None,
            )
            if which == "full":
                train_novel = VGGSoundDataset(
                    args=args,
                    dataset_split="train_val_novel" + iteration_test,
                    zero_shot_mode=None,
                )
        val_all_dataset = VGGSoundDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        if args.retrain_all == False:
            train_base = UCFDataset(
                args=args,
                dataset_split="train_base",
                zero_shot_mode="train",
            )
            if which == "full":
                train_novel = UCFDataset(
                    args=args,
                    dataset_split="train_novel",
                    zero_shot_mode="train",
                )
        if args.retrain_all == True:
            train_base = UCFDataset(
                args=args,
                dataset_split="train_val_base",
                zero_shot_mode=None,
            )
            if which == "full":
                train_novel = UCFDataset(
                    args=args,
                    dataset_split="train_val_novel" + iteration_test,
                    zero_shot_mode=None,
                )
        val_all_dataset = UCFDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        if args.retrain_all == False:
            train_base = ActivityNetDataset(
                args=args,
                dataset_split="train_base",
                zero_shot_mode="train",
            )
            if which == "full":
                train_novel = ActivityNetDataset(
                    args=args,
                    dataset_split="train_novel",
                    zero_shot_mode="train",
                )
        if args.retrain_all == True:
            train_base = ActivityNetDataset(
                args=args,
                dataset_split="train_val_base",
                zero_shot_mode=None,
            )
            if which == "full":
                train_novel = ActivityNetDataset(
                    args=args,
                    dataset_split="train_val_novel" + iteration_test,
                    zero_shot_mode=None,
                )
        val_all_dataset = ActivityNetDataset(
            args=args,
            dataset_split="val",
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    if args.retrain_all == False:
        contrastive_train_base = ContrastiveDataset(train_base)
        if which == "full":
            contrastive_train_novel = ContrastiveDataset(train_novel)
    if args.retrain_all == True:
        contrastive_train_val_base = ContrastiveDataset(train_base)
        if which == "full":
            contrastive_train_val_novel = ContrastiveDataset(train_novel)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)

    if args.retrain_all == False:
        train_sampler_base = SamplerFactory(logger).get(
            class_idxs=list(contrastive_train_base.target_to_indices.values()),
            batch_size=args.bs,
            n_batches=args.n_batches,
            alpha=1,
            kind="random",
        )
        if which == "full":
            train_novel_sampler = SamplerFactory(logger).get(
                class_idxs=list(contrastive_train_novel.target_to_indices.values()),
                batch_size=args.bs,
                n_batches=args.n_batches,
                alpha=1,
                kind="random",
            )

    if args.retrain_all == True:
        train_val_sampler_base = SamplerFactory(logger).get(
            class_idxs=list(contrastive_train_val_base.target_to_indices.values()),
            batch_size=args.bs,
            n_batches=args.n_batches,
            alpha=1,
            kind="random",
        )
        if which == "full":
            train_val_novel_sampler = SamplerFactory(logger).get(
                class_idxs=list(contrastive_train_val_novel.target_to_indices.values()),
                batch_size=args.bs,
                n_batches=args.n_batches,
                alpha=1,
                kind="random",
            )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(contrastive_val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind="random",
    )
    if args.selavi == False:
        collator_train = DefaultCollator(
            mode=args.batch_seqlen_train,
            max_len=args.batch_seqlen_train_maxlen,
            trim=args.batch_seqlen_train_trim,
        )
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_val_maxlen,
            trim=args.batch_seqlen_test_trim,
        )
    elif args.selavi == True:
        collator_train = DefaultCollator(
            mode=args.batch_seqlen_train,
            max_len=args.batch_seqlen_train_maxlen,
            trim=args.batch_seqlen_train_trim,
            rate_video=1,
            rate_audio=1,
        )
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_val_maxlen,
            trim=args.batch_seqlen_test_trim,
            rate_video=1,
            rate_audio=1,
        )

    if args.retrain_all == False:
        loader_base = data.DataLoader(
            dataset=contrastive_train_base,
            batch_sampler=train_sampler_base,
            collate_fn=collator_train,
            num_workers=4,
        )
        if which == "full":
            loader_novel = data.DataLoader(
                dataset=contrastive_train_novel,
                batch_sampler=train_novel_sampler,
                collate_fn=collator_train,
                num_workers=4,
            )

    val_all_loader_no_sampler = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        collate_fn=collator_test,
        batch_size=args.bs,
        num_workers=4,
    )

    if args.retrain_all == True:
        loader_base = data.DataLoader(
            dataset=contrastive_train_val_base,
            batch_sampler=train_val_sampler_base,
            collate_fn=collator_train,
            num_workers=4,
        )
        if which == "full":
            loader_novel = data.DataLoader(
                dataset=contrastive_train_val_novel,
                batch_sampler=train_val_novel_sampler,
                collate_fn=collator_train,
                num_workers=4,
            )

    val_all_loader_sampler = data.DataLoader(
        dataset=contrastive_val_all_dataset,
        batch_sampler=val_all_sampler,
        collate_fn=collator_test,
        num_workers=4,
    )
    model_params = get_model_params(
        args.lr,
        args.reg_loss,
        args.embedding_dropout,
        args.decoder_dropout,
        args.additional_dropout,
        args.embeddings_hidden_size,
        args.decoder_hidden_size,
        args.embeddings_batch_norm,
        args.rec_loss,
        args.cross_entropy_loss,
        args.transformer_use_embedding_net,
        args.transformer_dim,
        args.transformer_depth,
        args.transformer_heads,
        args.transformer_dim_head,
        args.transformer_mlp_dim,
        args.transformer_dropout,
        args.transformer_embedding_dim,
        args.transformer_embedding_time_len,
        args.transformer_embedding_dropout,
        args.transformer_embedding_time_embed_type,
        args.transformer_embedding_fourier_scale,
        args.transformer_embedding_embed_augment_position,
        args.lr_scheduler,
        args.optimizer,
        args.use_self_attention,
        args.use_cross_attention,
        args.transformer_average_features,
        args.audio_only,
        args.video_only,
        args.transformer_use_class_token,
        args.transformer_embedding_modality,
        args.latent_generator,
        args.discriminator_hidden_size,
        args.generator_hidden_size,
        args.calibration_net_hidden_size,
        args.learn_calibration_fake,
        args.use_calibration,
        args.detach_output,
        args.attention_type,
        args.use_diffusion_model,
        args.diffusion_steps,
        args.layer_change_attention,
        args.use_mixup,
        args.mixup_parameter,
        args.output_dimension_transformer,
        args.use_diffusion_batch_norm,
        args.diffusion_dropout_value,
        args.embedding_type,
        args.number_layers_diffusion,
    )

    if model == None:
        loader_train = (loader_base, None)
    elif model != None:
        loader_train = (loader_base, loader_novel)

    if model == None:
        if args.final_model == True:
            model = AVDiff(
                model_params,
                input_size_audio=args.input_size_audio,
                input_size_video=args.input_size_video,
                length_logits=args.logits_length,
            )
        else:
            raise AttributeError("No correct model name.")
    elif model != None:
        model.reset_lr()
        if args.freeze_model == True:
            model.freeze_parameters()

    print_model_size(model, logger)
    model.to(args.device)

    distance_fn = getattr(sys.modules[__name__], args.distance_fn)()

    if which == "base":
        metrics = [
            MeanClassAccuracy(
                model=model,
                dataset=(val_all_dataset, val_all_loader_no_sampler),
                device=args.device,
                distance_fn=distance_fn,
                new_model_sequence=args.new_model_sequence,
                args=args,
                use_calibration=False,
                which=which,
            )
        ]
    elif which == "full":
        metrics = [
            MeanClassAccuracy(
                model=model,
                dataset=(val_all_dataset, val_all_loader_no_sampler),
                device=args.device,
                distance_fn=distance_fn,
                new_model_sequence=args.new_model_sequence,
                args=args,
                use_calibration=True,
                which=which,
            )
        ]

    logger.info(model)
    logger.info(None)
    logger.info(None)
    logger.info(None)
    logger.info([metric.__class__.__name__ for metric in metrics])

    optimizer = None

    best_loss, best_score, best_epoch, best_model, epoch_lr_reduction, beta = train(
        train_loader=loader_train if args.retrain_all else loader_train,
        val_loader=val_all_loader_sampler,
        model=model,
        optimizer=optimizer,
        epochs=args.epochs if which == "base" else args.epochs_fine_tune,
        device=args.device,
        writer=writer,
        metrics=metrics,
        train_stats=train_stats,
        val_stats=val_stats,
        log_dir=log_dir,
        which=which,
        which_stage=which_stage,
        lr_reduction_epoch=lr_reduction_epoch,
        dictionary=dictionary,
        iteration_test=iteration_test,
        args=args,
    )

    logger.info(f"FINISHED. Run is stored at {log_dir}")

    if args.retrain_all == False:
        del loader_base, train_base, contrastive_train_base
        if which == "full":
            del loader_novel, train_novel, contrastive_train_novel
    elif args.retrain_all == True:
        del loader_base, train_base, contrastive_train_val_base
        if which == "full":
            del loader_novel, train_novel, contrastive_train_val_novel

    gc.collect()

    return (
        log_dir,
        best_epoch,
        best_model,
        (logger, log_dir, writer, train_stats, val_stats),
        epoch_lr_reduction,
        beta,
    )


if __name__ == "__main__":
    run()
