# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    
    if D.output_shapes[1][1] > 0:
       with tf.name_scope('LabelPenalty'):
           labels = labels / tf.reduce_sum(labels,axis=1)[:,None]
           label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
       loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.
    
    # wgan_target = 750

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True)) #real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))#fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            labels = labels / tf.reduce_sum(labels,axis=1)[:,None]
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------

# StyleGAN论文提倡的损失函数
# "Which Training Methods for GANs do actually Converge?"————论文地址：https://arxiv.org/abs/1801.04406

# def G_logistic_saturating(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
#     loss = -tf.nn.softplus(fake_scores_out)
#     return loss  # Loss_G = -log(exp(D(G(z))) + 1)

# def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     labels = training_set.get_random_labels_tf(minibatch_size)
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
#     loss = tf.nn.softplus(-fake_scores_out)
#     return loss  # Loss_G = log(exp(-D(G(z))) + 1)

# def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
#     fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
#     real_scores_out = tfutil.autosummary('Loss/scores/real', real_scores_out)
#     fake_scores_out = tfutil.autosummary('Loss/scores/fake', fake_scores_out)
#     loss = tf.nn.softplus(fake_scores_out)
#     loss += tf.nn.softplus(-real_scores_out)
#     return loss  # Loss_D = log(exp(D(G(z))) + 1) + log(exp(-D(x)) + 1)

# def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, labels, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
#     latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
#     fake_images_out = G.get_output_for(latents, labels, is_training=True)
#     real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
#     fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
#     real_scores_out = tfutil.autosummary('Loss/scores/real', real_scores_out)
#     fake_scores_out = tfutil.autosummary('Loss/scores/fake', fake_scores_out)
#     loss = tf.nn.softplus(fake_scores_out)
#     loss += tf.nn.softplus(-real_scores_out)

#     if r1_gamma != 0.0:
#         with tf.name_scope('R1Penalty'):
#             real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))  # 惩罚区(来自真实样本)的判别值D(x_real)
#             real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))  # 惩罚区样本的梯度∇T_real
#             r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])  # 惩罚项为∑(∇T_real^2)
#             r1_penalty = tfutil.autosummary('Loss/r1_penalty', r1_penalty)
#         loss += r1_penalty * (r1_gamma * 0.5)

#     if r2_gamma != 0.0:
#         with tf.name_scope('R2Penalty'):
#             fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))  # 惩罚区(来自生成样本)的判别值D(x_fake)
#             fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))  # 惩罚区样本的梯度∇T_fake
#             r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])  # 惩罚项为∑(∇T_fake^2)
#             r2_penalty = tfutil.autosummary('Loss/r2_penalty', r2_penalty)
#         loss += r2_penalty * (r2_gamma * 0.5)
#     return loss  # Loss_D = log(exp(D(G(z))) + 1) + log(exp(-D(x)) + 1) + r1_gamma*0.5*∑(∇T_real^2) + r2_gamma*0.5*∑(∇T_fake^2)

#----------------------------------------------------------------------------