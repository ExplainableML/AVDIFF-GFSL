#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

np.random.seed(0)

def get_model_params(
    lr,
    reg_loss,
    dropout_encoder,
    dropout_decoder,
    additional_dropout,
    encoder_hidden_size,
    decoder_hidden_size,
    embeddings_batch_norm,
    rec_loss,
    cross_entropy_loss,
    transformer_use_embedding_net,
    transformer_dim,
    transformer_depth,
    transformer_heads,
    transformer_dim_head,
    transformer_mlp_dim,
    transformer_dropout,
    transformer_embedding_dim,
    transformer_embedding_time_len,
    transformer_embedding_dropout,
    transformer_embedding_time_embed_type,
    transformer_embedding_fourier_scale,
    transformer_embedding_embed_augment_position,
    lr_scheduler,
    optimizer,
    use_self_attention,
    use_cross_attention,
    transformer_average_features,
    audio_only,
    video_only,
    transformer_use_class_token,
    transformer_embedding_modality,
    latent_generator,
    discriminator_hidden_size,
    generator_hidden_size,
    calibration_net_hidden_size,
    learn_calibration_fake,
    use_calibration,
    detach_output,
    attention_type,
    use_diffusion_model,
    diffusion_steps,
    layer_change_attention,
    use_mixup,
    mixup_parameter,
    output_dimension_transformer,
    use_diffusion_batch_norm,
    diffusion_dropout_value,
    embedding_type,
    number_layers_diffusion,
    ):

    params_model = dict()
    # Dimensions
    params_model['dim_out'] = output_dimension_transformer
    params_model['cross_entropy_loss']=cross_entropy_loss

    # Optimizers' parameters
    params_model['lr'] = lr
    params_model['optimizer'] = optimizer
    if encoder_hidden_size==0:
        encoder_hidden_size=None
    if decoder_hidden_size==0:
        decoder_hidden_size=None


    params_model['use_diffusion_batch_norm']=use_diffusion_batch_norm
    params_model['diffusion_dropout_value']=diffusion_dropout_value
    params_model['additional_dropout']=additional_dropout
    params_model['reg_loss']=reg_loss
    params_model['dropout_encoder']=dropout_encoder
    params_model['dropout_decoder']=dropout_decoder
    params_model['encoder_hidden_size']=encoder_hidden_size
    params_model['decoder_hidden_size']=decoder_hidden_size
    params_model['latent_generator']=latent_generator
    params_model['discriminator_hidden_size']=discriminator_hidden_size
    params_model['generator_hidden_size']=generator_hidden_size
    params_model['calibration_net_hidden_size']=calibration_net_hidden_size
    params_model['learn_calibration_fake']=learn_calibration_fake
    params_model['use_calibration']=use_calibration
    params_model['detach_output']=detach_output
    params_model['attention_type']=attention_type
    params_model['use_diffusion_model']=use_diffusion_model
    params_model['diffusion_steps']=diffusion_steps
    params_model['layer_change_attention']=layer_change_attention
    params_model['use_mixup']=use_mixup
    params_model['mixup_parameter']=mixup_parameter
    params_model['embedding_type']=embedding_type
    params_model['number_layers_diffusion']=number_layers_diffusion

    # Model Sequence
    params_model['embeddings_batch_norm'] = embeddings_batch_norm
    params_model['rec_loss'] = rec_loss
    params_model['transformer_average_features'] = transformer_average_features
    params_model['transformer_use_embedding_net'] = transformer_use_embedding_net
    params_model['transformer_dim'] = transformer_dim
    params_model['transformer_depth'] = transformer_depth
    params_model['transformer_heads'] = transformer_heads
    params_model['transformer_dim_head'] = transformer_dim_head
    params_model['transformer_mlp_dim'] = transformer_mlp_dim
    params_model['transformer_dropout'] = transformer_dropout
    params_model['transformer_embedding_dim'] = transformer_embedding_dim
    params_model['transformer_embedding_time_len'] = transformer_embedding_time_len
    params_model['transformer_embedding_dropout'] = transformer_embedding_dropout
    params_model['transformer_embedding_time_embed_type'] = transformer_embedding_time_embed_type
    params_model['transformer_embedding_fourier_scale'] = transformer_embedding_fourier_scale
    params_model['transformer_embedding_embed_augment_position'] = transformer_embedding_embed_augment_position
    params_model['transformer_embedding_modality'] = transformer_embedding_modality
    params_model['transformer_attention_use_self_attention']=use_self_attention
    params_model['transformer_attention_use_cross_attention']=use_cross_attention
    params_model['audio_only'] = audio_only
    params_model['video_only'] = video_only
    params_model['transformer_use_class_token'] = transformer_use_class_token

    params_model['lr_scheduler'] = lr_scheduler
    return params_model
