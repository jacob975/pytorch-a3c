import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_atari_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

# Note by YLChiu
# rank is the index of agents
# args is arguements from main.py
# shared_model is the model declared in main.py
# counter is number of iterations of all agents
# optimizer is used to sync. params with shared_model
def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank) # Use different seed in agents

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach() # detach from current graph, i.e. no gradient
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((
                state.unsqueeze(0),
                (hx, cx)
            ))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True) # This entropy can only be used on discrete action space.
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach() # sample an action from policy
            log_prob = log_prob.gather(1, action) # The entropy of the sampled action

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        # Take the last q-value from model
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        # Estimate the advantage of each time steps with the last q-value and immediate rewards.
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            # Use entropy in Soft Actor-Critic style
            # =Estimated likelihood + entropy * alpha
            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i] 

        # Reset gradients
        optimizer.zero_grad()

        # Estimate the gradient
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Put paras. from model to shared_model
        ensure_shared_grads(model, shared_model)

        # Apply gradients on shared_model
        optimizer.step()
