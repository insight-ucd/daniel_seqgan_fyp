import numpy as np
import random

def train_epoch(sess, trainable_model, num_iter,
                proportion_supervised, g_steps, d_steps,
                next_sequence, results, epoch, verify_sequence=None,
                words=None,
                proportion_generated=0.5):
    """Perform training for model.

    sess: tensorflow session
    trainable_model: the model
    num_iter: number of iterations
    proportion_supervised: what proportion of iterations should the generator
        be trained in a supervised manner (rather than trained via discriminator)
    g_steps: number of generator training steps per iteration
    d_steps: number of discriminator t raining steps per iteration
    next_sequence: function that returns a groundtruth sequence
    verify_sequence: function that checks a generated sequence, returning True/False
    words:  array of words (to map indices back to words)
    proportion_generated: what proportion of steps for the discriminator
        should be on artificially generated data

    """
    supervised_g_losses = [0]  # we put in 0 to avoid empty slices
    unsupervised_g_losses = [0]  # we put in 0 to avoid empty slices
    d_losses = [0]
    expected_rewards = [[0] * trainable_model.sequence_length]
    supervised_correct_generation = [0]
    unsupervised_correct_generation = [0]
    supervised_gen_x = None
    unsupervised_gen_x = None
    print('running %d iterations with %d g steps and %d d steps' % (num_iter, g_steps, d_steps))
    print('of the g steps, %.2f will be supervised' % proportion_supervised)
    for it in range(num_iter):
        for _ in range(g_steps):
            if random.random() < proportion_supervised:
                seq = next_sequence()
                _, g_loss, g_pred = trainable_model.pretrain_step(sess, seq)
                supervised_g_losses.append(g_loss)

                supervised_gen_x = np.argmax(g_pred, axis=1)
                if verify_sequence is not None:
                    supervised_correct_generation.append(
                        verify_sequence(supervised_gen_x))
            else:
                _, _, g_loss, expected_reward, unsupervised_gen_x = \
                    trainable_model.train_g_step(sess)
                expected_rewards.append(expected_reward)
                unsupervised_g_losses.append(g_loss)

                if verify_sequence is not None:
                    unsupervised_correct_generation.append(
                        verify_sequence(unsupervised_gen_x))

        for _ in range(d_steps):
            if random.random() < proportion_generated:
                seq = next_sequence()
                _, d_loss = trainable_model.train_d_real_step(sess, seq)
            else:
                _, d_loss = trainable_model.train_d_gen_step(sess)
            d_losses.append(d_loss)
    

    results['d_losses'][epoch] = np.mean(d_losses)
    results['supervised_g_losses'][epoch] = np.mean(supervised_g_losses)
    results['unsupervised_g_losses'][epoch] = np.mean(unsupervised_g_losses)
    results['supervised_generations'][epoch] = [words[x] if words else x for x in supervised_gen_x] if supervised_gen_x is not None else None
    results['unsupervised_generations'][epoch] = [words[x] if words else x for x in unsupervised_gen_x] if unsupervised_gen_x is not None else None
    results['rewards'][epoch] = np.mean(expected_rewards, axis=0)
    print('epoch statistics:')
    print('>>>> discriminator loss:', results['d_losses'][epoch])
    
    print('>>>> generator loss:', results['supervised_g_losses'][epoch], results['unsupervised_g_losses'][epoch])

    if verify_sequence is not None:
        print('>>>> correct generations (supervised, unsupervised):', np.mean(supervised_correct_generation), np.mean(unsupervised_correct_generation))
    print('>>>> sampled generations (supervised, unsupervised):',)
    print(results['supervised_generations'][epoch],)
    print(results['unsupervised_generations'][epoch])
    print('>>>> expected rewards:', results['rewards'][epoch])
