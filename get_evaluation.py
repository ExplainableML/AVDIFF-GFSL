from src.dataset import DefaultCollator
from src.args import args_main
from torch.utils import data
from src.dataset import (
    ActivityNetDataset,
    VGGSoundDataset,
    UCFDataset,
    ContrastiveDataset,
)
from src.AVDiff import AVDiff
from src.test import test
from src.utils import (
    fix_seeds,
    load_args,
    setup_evaluation,
    load_model_weights,
)
from src.utils_improvements import get_model_params


def get_evaluation(args, dictionary_stage_1, dictionary_stage_2, iteration_test, beta):

    config = load_args(args.load_path_stage_B)
    config.root_dir = args.root_dir
    if config.input_size is not None:
        config.input_size_audio = config.input_size
        config.input_size_video = config.input_size

    assert (
        config.retrain_all
    ), f"--retrain_all flag is not set in load_path_stage_B. Are you sure this is the correct path?. {args.load_path_stage_B}"
    fix_seeds(config.seed)

    logger, eval_dir, test_stats, tb_writer = setup_evaluation(
        args, config.__dict__.keys()
    )

    if args.dataset_name == "VGGSound":
        val_all_dataset = VGGSoundDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = VGGSoundDataset(
            args=config,
            dataset_split="test" + iteration_test,
            zero_shot_mode=None,
        )
    elif args.dataset_name == "UCF":
        val_all_dataset = UCFDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = UCFDataset(
            args=config,
            dataset_split="test" + iteration_test,
            zero_shot_mode=None,
        )
    elif args.dataset_name == "ActivityNet":
        val_all_dataset = ActivityNetDataset(
            args=config,
            dataset_split="val",
            zero_shot_mode=None,
        )
        test_dataset = ActivityNetDataset(
            args=config,
            dataset_split="test" + iteration_test,
            zero_shot_mode=None,
        )
    else:
        raise NotImplementedError()

    contrastive_val_dataset = ContrastiveDataset(val_all_dataset)
    contrastive_test_dataset = ContrastiveDataset(test_dataset)

    if config.selavi == False:
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_test_maxlen,
            trim=args.batch_seqlen_test_trim,
        )
        collator_val = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_val_maxlen,
            trim=args.batch_seqlen_test_trim,
        )

    elif config.selavi == True:
        collator_test = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_test_maxlen,
            trim=args.batch_seqlen_test_trim,
            rate_video=1,
            rate_audio=1,
        )
        collator_val = DefaultCollator(
            mode=args.batch_seqlen_test,
            max_len=args.batch_seqlen_val_maxlen,
            trim=args.batch_seqlen_test_trim,
            rate_video=1,
            rate_audio=1,
        )

    final_val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        collate_fn=collator_val,
        batch_size=args.eval_bs,
        num_workers=args.eval_num_workers,
    )

    final_test_loader = data.DataLoader(
        dataset=contrastive_test_dataset,
        collate_fn=collator_test,
        batch_size=args.eval_bs,
        num_workers=args.eval_num_workers,
    )

    model_params = get_model_params(
        config.lr,
        config.reg_loss,
        config.embedding_dropout,
        config.decoder_dropout,
        config.additional_dropout,
        config.embeddings_hidden_size,
        config.decoder_hidden_size,
        config.embeddings_batch_norm,
        config.rec_loss,
        config.cross_entropy_loss,
        config.transformer_use_embedding_net,
        config.transformer_dim,
        config.transformer_depth,
        config.transformer_heads,
        config.transformer_dim_head,
        config.transformer_mlp_dim,
        config.transformer_dropout,
        config.transformer_embedding_dim,
        config.transformer_embedding_time_len,
        config.transformer_embedding_dropout,
        config.transformer_embedding_time_embed_type,
        config.transformer_embedding_fourier_scale,
        config.transformer_embedding_embed_augment_position,
        config.lr_scheduler,
        config.optimizer,
        config.use_self_attention,
        config.use_cross_attention,
        config.transformer_average_features,
        config.audio_only,
        config.video_only,
        config.transformer_use_class_token,
        config.transformer_embedding_modality,
        config.latent_generator,
        config.discriminator_hidden_size,
        config.generator_hidden_size,
        config.calibration_net_hidden_size,
        config.learn_calibration_fake,
        config.use_calibration,
        config.detach_output,
        config.attention_type,
        config.use_diffusion_model,
        config.diffusion_steps,
        config.layer_change_attention,
        config.use_mixup,
        config.mixup_parameter,
        config.output_dimension_transformer,
        config.use_diffusion_batch_norm,
        config.diffusion_dropout_value,
        config.embedding_type,
        config.number_layers_diffusion,
    )
    if config.final_model == True:
        model_A = AVDiff(
            params_model=model_params,
            input_size_audio=config.input_size_audio,
            input_size_video=config.input_size_video,
            length_logits=len(dictionary_stage_1[1]),
        )
        model_B = AVDiff(
            params_model=model_params,
            input_size_audio=config.input_size_audio,
            input_size_video=config.input_size_video,
            length_logits=len(dictionary_stage_2[1]),
        )
    else:
        raise AttributeError("No correct model_A name.")

    weights_path_stage_A = list(args.load_path_stage_A.glob("*_score.pt"))[0]
    epoch_A = load_model_weights(weights_path_stage_A, model_A)
    weights_path_stage_B = list(
        (args.load_path_stage_B / ("checkpoints" + str(iteration_test))).glob(
            f"*_ckpt_{epoch_A - 1}.pt"
        )
    )[0]
    _ = load_model_weights(weights_path_stage_B, model_B)

    model_A.to(config.device)
    model_B.to(config.device)

    results = test(
        test_dataset=(test_dataset, final_test_loader),
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        args=config,
        dictionary=dictionary_stage_2[1],
        save_performances=args.eval_save_performances,
        best_beta=beta,
    )

    # Tensorboard HParam logging

    logger.info("FINISHED evaluation")
    logger.info("Test results", results)

    return results, logger, [tb_writer, config]


if __name__ == "__main__":
    args, eval_args = args_main()
    get_evaluation(eval_args)
