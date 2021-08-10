import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.trainer.track_information import InfoTracker
from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(
            polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def get_lstm_state_ph(name='', n_batches=None, num_units=64):
    c = tf.placeholder(
        tf.float32, [n_batches, 1,  num_units], name=name+'c_ph')
    h = tf.placeholder(tf.float32, [n_batches, 1, num_units], name=name+'h_ph')
    return c, h


def create_init_state(num_batches, len_sequence):
    c_init = np.zeros((num_batches, 1, len_sequence), np.float32)
    h_init = np.zeros((num_batches, 1, len_sequence), np.float32)
    return c_init, h_init


def get_lstm_states(_type, trainers):
    if _type == 'p':
        return [(agent.p_c, agent.p_h) for agent in trainers]
    else:
        raise ValueError("unknown type")


def _p_train(make_obs_ph_n, act_space_n, p_index, p_func, p_lstm_on, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None, 1], name="action"+str(i)) for i in range(len(act_space_n))]

        p_res = int(act_pdtype_n[p_index].param_shape()[0])

        # for actor
        p_c_ph, p_h_ph = get_lstm_state_ph(
            name='p_', n_batches=None, num_units=num_units)
        p_c_ph_n, p_h_ph_n = [p_c_ph for i in range(len(obs_ph_n))], [
            p_h_ph for i in range(len(obs_ph_n))]

        if p_lstm_on:
            p_input = tf.concat([obs_ph_n[p_index], p_c_ph, p_h_ph], -1)
            p, p_state_out = p_func(
                p_input, p_res, scope="p_func", num_units=num_units)
        else:
            p_input = obs_ph_n[p_index]
            p = p_func(p_input, p_res, scope="p_func", num_units=num_units)

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        act_sample = act_pd.sample()

        perturber = tf.train.GradientDescentOptimizer(
            learning_rate=1.0)
        ploss = tf.reduce_mean(p)
        p_grad = perturber.compute_gradients(
            ploss, var_list=p_func_vars)

        noise_ph = [tf.placeholder(dtype=tf.float32, shape=grad.shape)
                    for i, (grad, var) in enumerate(p_grad)]
        p_grad_ph = []
        # Create policy gradient placeholder
        for noise, (grad, var) in zip(noise_ph, p_grad):
            if grad is not None:
                p_grad_ph.append((noise, var))
        p_update = perturber.apply_gradients(p_grad_ph)

        if p_lstm_on:
            act = U.function(inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=[
                             act_sample, p_state_out])
            p_values = U.function(
                inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=p)
            p_get_grad = U.function(
                inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph], outputs=p_grad)

            p_apply_update = U.function(
                inputs=[obs_ph_n[p_index], p_c_ph, p_h_ph] + noise_ph, outputs=ploss, updates=[p_update])

        else:
            act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
            p_values = U.function(inputs=[obs_ph_n[p_index]], outputs=p)

            p_get_grad = U.function(
                inputs=[obs_ph_n[p_index]], outputs=p_grad)

            p_apply_update = U.function(
                inputs=[obs_ph_n[p_index]]+noise_ph, outputs=ploss, updates=[p_update])

        return act, p_values, p_get_grad, p_apply_update


class ZOMADDPGAgentTrainer(AgentTrainer):
    def __init__(self, name, mlp_model, lstm_model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(
                obs_shape_n[i], name="observation"+str(i)).get())

        # LSTM placeholders
        p_res = 7

        # set up initial states
        self.p_c, self.p_h = create_init_state(
            num_batches=1, len_sequence=args.num_units)

        p_model = lstm_model if self.args.actor_lstm else mlp_model

        print("P model: {} because actor_lstm: {}".format(
            p_model, self.args.actor_lstm))

        self.act, self.p_values, self.p_get_grad, self.p_apply_update = _p_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=p_model,
            p_lstm_on=self.args.actor_lstm,
            num_units=args.num_units,
        )

        # Information tracking
        self.tracker = InfoTracker(self.name, self.args)

    def reset_lstm(self):
        self.p_c, self.p_h = create_init_state(
            num_batches=1, len_sequence=self.p_h.shape[-1])

    def action(self, obs):
        if self.args.actor_lstm:
            action, state = self.act(*[obs[None], self.p_c, self.p_h])
            self.p_c, self.p_h = state
        else:
            action = self.act(obs[None])

        action = action[0]
        if self.args.tracking:
            self.tracker.record_information(
                "communication", np.argmax(action[0][-2:]))
        return action

    def get_p_grad(self, obs):
        if self.args.actor_lstm:
            return self.p_get_grad(obs[None], self.p_c, self.p_h)
        else:
            return self.p_get_grad(obs[None])

    def perturb_policy(self, obs, noise):
        # print('obs type is {}'.format(type(obs[None])))
        # print('\n noise type is {}'.format(type(noise)))
        if self.args.actor_lstm:
            return self.p_apply_update(*([obs[None], self.p_c, self.p_h] + noise))
        else:
            return self.p_apply_update(*([obs[None]] + noise))
