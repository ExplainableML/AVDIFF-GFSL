import configargparse
import pathlib


def args_main(*args, **kwargs):
    parser = configargparse.ArgParser(
        description="Explainable Audio Visual Low Shot Learning",
        default_config_files=["config/default.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument('-c', '--cfg', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--run', default='all', type=str, choices=['all', 'stage-1', 'stage-2', 'eval'])

    ### Filesystem ###

    parser.add_argument(
        "--root_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        required=True,
        type=pathlib.Path
    )

    parser.add_argument(
        "--log_dir",
        help="Path where to create experiment log dirs",
        type=pathlib.Path
    )

    parser.add_argument(
        "--exp_name",
        help="Flag to set the name of the experiment",
        type=str
    )

    parser.add_argument(
        "--feature_extraction_method",
        help="Name of folder containing respective extracted features. Has to match {feature_extraction_method} in --root_dir argument.",
        required=False,
        type=pathlib.Path
    )

    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        type=str
    )

    parser.add_argument(
        "--selavi",
        help="Wether to use selavi features or cls features",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--zero_shot_split",
        help="Name of zero shot split to use.",
        choices=["", "1-shot", "5-shot", "10-shot", "20-shot"],
    )

    parser.add_argument(
        "--manual_text_word2vec",
        help="Flag to use the manual word2vec text embeddings. CARE: Need to create cache files again!",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--val_all_loss",
        help="Validate loss with seen + unseen",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--additional_triplets_loss",
        help="Flag for using more triplets loss",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--reg_loss",
        help="Flag for setting the regularization loss",
        type=str_to_bool, nargs='?', const=True

    )

    parser.add_argument(
        "--cycle_loss",
        help="Flag for using cycle loss",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--retrain_all",
        help="Retrain with all data from train and validation",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--save_checkpoints",
        help="Save checkpoints of the model every epoch",
        type=str_to_bool, nargs='?', const=True
    )

    ### Development options ###
    parser.add_argument(
        "--debug",
        help="Run the program in debug mode",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--verbose",
        help="Run verbosely",
        type=str_to_bool, nargs='?', const=True,
    )
    parser.add_argument(
        "--debug_comment",
        help="Custom comment string for the summary writer",
        type=str
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        type=int
    )

    parser.add_argument(
        "--use_calibration",
        help="wether to use the calibration or not",
        type=str_to_bool, nargs='?', const=True,
    )

    parser.add_argument(
        "--epochs_fine_tune",
        help="Number of epochs for fine tuning",
        type=int
    )

    parser.add_argument(
        "--norm_inputs",
        help="Normalize inputs before model",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--z_score_inputs",
        help="Z-Score standardize inputs before model",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--batch_seqlen_train",
        type=str,
        choices=["max", "fixed"]
    )
    parser.add_argument(
        "--batch_seqlen_train_maxlen",
        type=int
    )
    parser.add_argument(
        "--batch_seqlen_train_trim",
        type=str,
        choices=["random", "center"]
    )
    parser.add_argument(
        "--batch_seqlen_test",
        type=str,
        choices=["max", "fixed"]
    )
    parser.add_argument(
        "--batch_seqlen_test_maxlen",
        type=int
    )

    parser.add_argument(
        "--batch_seqlen_val_maxlen",
        type=int
    )

    parser.add_argument(
        "--batch_seqlen_test_trim",
        type=str,
        choices=["random", "center"]
    )


    ### Hyperparameters ###

    parser.add_argument(
        "--mixup_parameter",
        help="What value to use in the mixup",
        type=float
    )

    parser.add_argument(
        "--output_dimension_transformer",
        help="the dimension of the output of the transformer",
        type=int
    )

    parser.add_argument(
        "--use_mixup",
        help="Wether to use mixup or not",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--cross_entropy_loss",
        help="Use the crossentropy loss",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--optimizer",
        help="Select Optimizer used for training",
        type=str,
        choices=["adam", "adam-sam"]
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float
    )
    parser.add_argument(
        "--bs",
        help="Batch size",
        type=int
    )
    parser.add_argument(
        "--n_batches",
        help="Number of batches for the balanced batch sampler",
        type=int
    )
    parser.add_argument(
        "--input_size",
        help="Dimension of the extracted features",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--input_size_audio",
        help="Dimension of the extracted audio features",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--input_size_video",
        help="Dimension of the extracted video features",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--embeddings_hidden_size",
        help="Hidden layer size for the embedding networks",
        type=int
    )
    parser.add_argument(
        "--decoder_hidden_size",
        help="Hidden layer size for the decoder loss network",
        type=int
    )
    parser.add_argument(
        "--embedding_dropout",
        help="Dropout in the embedding networks",
        type=float
    )
    parser.add_argument(
        "--decoder_dropout",
        help="Dropout in the decoder loss network",
        type=float
    )
    parser.add_argument(
        "--embedding_use_bn",
        help="Use batchnorm in the embedding networks",
        type=str_to_bool, nargs='?', const=True,
    )
    parser.add_argument(
        "--decoder_use_bn",
        help="Use batchnorm in the decoder network",
        type=str_to_bool, nargs='?', const=True,
    )
    parser.add_argument(
        "--normalize_decoder_outputs",
        help="L2 normalize the outputs of the decoder",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--margin",
        help="Margin for the contrastive loss calculation",
        type=float
    )
    parser.add_argument(
        "--distance_fn",
        help="Distance function for the contrastive loss calculation",
        choices=["L2Loss", "SquaredL2Loss"],
        type=str
    )
    parser.add_argument(
        "--lr_scheduler",
        help="Use LR_scheduler",
        type=str_to_bool, nargs='?', const=True,
    )

    # defaults
    parser.add_argument(
        "--seed",
        help="Random seed",
        type=int
    )

    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu"],
    )

    model_group = parser.add_argument_group('model')

    model_group.add_argument(
        "--baseline",
        help="Flag to use the baseline where we have two ALEs, one for each modality and we just try to push the modalities to text embeddings",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--audio_baseline",
        help="Flag to use the audio baseline",
        type=str_to_bool, nargs='?', const=True
    )
    model_group.add_argument(
        "--video_baseline",
        help="Flag to use the video baseline",
        type=str_to_bool, nargs='?', const=True

    )
    model_group.add_argument(
        "--concatenated_baseline",
        help="Flag to use the concatenated baseline",
        type=str_to_bool, nargs='?', const=True

    )
    model_group.add_argument(
        "--cjme",
        help="Flag to use the CJME baseline",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model",
        help="Flag to use the new model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model_early_fusion",
        help="Flag to use the early fusion new model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model_middle_fusion",
        help="Flag to set the middle fusion new model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model_attention",
        help="Flag to set the attention to the new model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model_attention_both_heads",
        help="Flag to set if attention should provide output from both branches",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--new_model_sequence",
        help="Flag to use multimodal Transformer on sequences",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--TCAF",
        help="Flag to use TCAF",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--TSL",
        help="Flag to use the TSL baseline",
        type=str_to_bool, nargs="?", const=True
    )

    model_group.add_argument(
        "--final_model",
        help="Flag to use the final model",
        type=str_to_bool, nargs="?", const=True
    )

    model_group.add_argument(
        "--bmt",
        help="Flag to use bmt model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--zorro",
        help="Flag to use the zorro model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--sldg_tav",
        help="flag to use the sldg_tav",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--perceiver",
        help="Flag to use the Perceiver model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--hierarchical_perceiver",
        help="Flag to use the Hierarchical Perceiver model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--AVCA",
        help="Flag to use the AVCA model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--AVCA_cls",
        help="Flag to use AVCA for classification",
        type=str_to_bool, nargs="?", const=True
    )

    model_group.add_argument(
        "--AVCA_cls_original",
        help="Flag to use AVCA for classification similar to original",
        type=str_to_bool, nargs="?", const=True
    )

    model_group.add_argument(
        "--attention_fusion",
        help="Flag to use the Attention fusion model",
        type=str_to_bool, nargs="?", const=True
    )

    model_group.add_argument(
        "--ale",
        help="Flag to set the ale",
        type=str_to_bool, nargs='?', const=True
    )
    model_group.add_argument(
        "--devise",
        help="Flag to set the devise model",
        type=str_to_bool, nargs='?', const=True
    )
    model_group.add_argument(
        "--sje",
        help="Flag to set the sje model",
        type=str_to_bool, nargs='?', const=True
    )

    model_group.add_argument(
        "--apn",
        help="Flag to set the apn model",
        type=str_to_bool, nargs='?', const=True
    )


    parser.add_argument(
        "--depth_transformer",
        help="Flag to se the number of layers of the transformer",
        type=int
    )

    parser.add_argument(
        "--first_additional_triplet",
        help="flag to set the first pair of additional triplets",
        type=int
    )

    parser.add_argument(
        "--second_additional_triplet",
        help="flag to set the second pair of additional triplets",
        type=int
    )

    parser.add_argument(
        "--layer_change_attention",
        help="layer to which to change from one type of attention to another type",
        type=int
    )

    parser.add_argument(
        "--third_additional_triplet",
        help="flag to set the third pair of additional triplets",
        type=int
    )
    parser.add_argument(
        "--additional_dropout",
        help="flag to set the additional dropouts",
        type=float
    )

    parser.add_argument(
        "--discriminator_hidden_size",
        help="size of the hidden dimension discriminator",
        type=int
    )
    parser.add_argument(
        "--generator_hidden_size",
        help="size of the hidden dimension generator",
        type=int,
    )
    parser.add_argument(
        "--calibration_net_hidden_size",
        help="size of hidden dimension calibrationnet",
        type=int
    )

    parser.add_argument(
        "--attention_type",
        help="type of the attention to be used in the transformer",
        choices = ["full_attention", "alternate", "m_to_o_stream","m_to_o_stream_alternate","o_to_m_stream","cross_attention", "cross_concatenation", "self_attention"],
        type = str
    )

    parser.add_argument(
        "--text_embeddings_file",
        help="name of the text embedding_file",
        type=str
    )

    parser.add_argument(
        "--embedding_type",
        help="the type of the text embedding",
        choices=["gpt_embedding", "w2v_embedding", "selavi"],
        type=str
    )


    parser.add_argument(
        "--freeze_model",
        help="wether to freeze the model_or_not",
        type = str_to_bool, nargs = '?', const = True,
    )

    parser.add_argument(
        "--number_layers_diffusion",
        help="how many layers in the diffusion model",
        type=int
    )

    parser.add_argument(
        "--use_diffusion_model",
        help="wether to use diffusion model or gans",
        type= str_to_bool, nargs = '?', const = True,
    )

    parser.add_argument(
        "--use_diffusion_batch_norm",
        help="wether to use diffusion for batch norm",
        type=str_to_bool, nargs='?', const=True,
    )

    parser.add_argument(
        "--diffusion_dropout_value",
        help="value of the dropout to use in diffusion",
        type=float,
    )

    parser.add_argument(
        "--diffusion_steps",
        help="how many steps to use in diffusion",
        type=int,
    )

    parser.add_argument(
        "--detach_output",
        help="wether to detach the output",
        type=str_to_bool, nargs='?', const=True,
    )

    parser.add_argument(
        "--learn_calibration_fake",
        help="wether to train the calibration also on fake samples",
        type=str_to_bool, nargs='?', const=True
    )

    ###### Model Sequence parameters
    parser.add_argument(
        "--rec_loss",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--ct_loss",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--w_loss",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--embeddings_batch_norm",
        type=str_to_bool, nargs='?', const=True
    )

    ###### Transformer model_sequence
    parser.add_argument(
        "--transformer_average_features",
        type=str_to_bool, nargs='?', const=True,
        help="This option temporally averages the features (before embedding them)."
    )
    parser.add_argument(
        "--transformer_use_class_token",
        type=str_to_bool, nargs='?', const=True,
    )
    parser.add_argument(
        "--use_self_attention",
        help="In transformer use only self-attention on each modality",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--use_cross_attention",
        help="In transformer use only cross-attention between modalities",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--audio_only",
        help="In transformer, use only audio modality.",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--video_only",
        help="In transformer, use only visual modality.",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--transformer_use_embedding_net",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--transformer_dim",
        type=int
    )
    parser.add_argument(
        "--latent_generator",
        type=int,
    )
    parser.add_argument(
        "--transformer_depth",
        type=int
    )
    parser.add_argument(
        "--transformer_heads",
        type=int
    )
    parser.add_argument(
        "--transformer_dim_head",
        type=int
    )
    parser.add_argument(
        "--transformer_mlp_dim",
        type=int
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float
    )

    ###### Transformer Embedding model_sequence
    parser.add_argument(
        "--transformer_embedding_dim",
        type=int
    )
    parser.add_argument(
        "--transformer_embedding_time_len",
        type=int
    )
    parser.add_argument(
        "--transformer_embedding_dropout",
        type=float
    )
    parser.add_argument(
        "--transformer_embedding_time_embed_type",
        type=str,
        choices=['none', 'fixed', 'sinusoid']
    )
    parser.add_argument(
        "--transformer_embedding_fourier_scale",
        type=float
    )
    parser.add_argument(
        "--transformer_embedding_embed_augment_position",
        type=str_to_bool, nargs='?', const=True,
    )
    parser.add_argument(
        "--transformer_embedding_modality",
        type=str_to_bool, nargs='?', const=True,
    )

    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument(
        "--load_path_stage_A",
        help="Path to experiment log folder of stage A",
        # required=True,
        type=pathlib.Path
    )
    eval_group.add_argument(
        "--load_path_stage_B",
        help="Path to experiment log folder of stage B",
        # required=True,
        type=pathlib.Path
    )
    eval_group.add_argument(
        "--eval_name",
        help="Evaluation name to be displayed in the final output string",
        type=str,
        # required=True
    )
    eval_group.add_argument(
        "--eval_bs",
        help="Batch size",
        type=int
    )
    eval_group.add_argument(
        "--eval_num_workers",
        help="Number of dataloader workers",
        type=int
    )
    eval_group.add_argument(
        "--eval_save_performances",
        help="Save class performances to disk",
        type=str_to_bool, nargs='?', const=True
    )
    args = parser.parse_args(*args, **kwargs)


    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=configargparse.Namespace(**group_dict)

    model_args = arg_groups['model']
    eval_args = arg_groups['eval']
    shared_args_list = ['root_dir', 'dataset_name', 'device', 'batch_seqlen_test', 'batch_seqlen_test_maxlen', 'batch_seqlen_test_trim', "batch_seqlen_val_maxlen"]
    shared_args_dict = {a:getattr(args,a,None) for a in shared_args_list}

    eval_main_args = configargparse.Namespace(**shared_args_dict, **vars(eval_args), **vars(model_args))


    return args, eval_main_args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')