import gym
import tensorflow as tf
import reachability as rch
import utils as utl


def run(env, player, target_len=10):
    obs = env.reset()

    R = 0
    r = 0.0
    count = 1
    episodes_played = 0

    while True:
        # HACK soln to padding sequences. wrap them and continue
        # rather than using the done flag

        seq_break = True if (count % (target_len+1) == 0) and (count != 1) else False

        a = player(obs, r, seq_break)
        obs, r, done, info = env.step(a)
        R += r

        if done:
            player.learner.reset()  # reset the explorers memory
            obs = env.reset()
            R = 0
            episodes_played += 1

        count += 1

        if count % 20 == 0:
            M = len(player.learner.memory)
            B = len(player.buffer.buffer)

            print('\ri: {} R: {} M: {}, B: {}'.format(episodes_played, R, M, B), end='', flush=True)

    return R, M


if __name__ == "__main__":
    tf.enable_eager_execution()
    env = gym.make('MontezumaRevenge-v0')
    player = utl.Worker(rch.Explorer(env.action_space.n), batch_size=50)
    run(env, player, 100)
