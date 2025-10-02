import gymnasium as gym
import numpy as np
import pygame

############
# configs
############

ENV_ID = "FrozenLake-v1"
MAP_NAME = "8x8"
IS_SLIPPERY = True 
SEED = 42

'''
this is the number of episodes to train the agent
'''
EPISODES = 40000

'''
this is called the discount factor! it spans 0 to 1 and controls how much 
the agent cares about future actions vs immediate rewards. i set it at 0.99
because i want the agent to plan several moves ahead
'''
GAMMA = 0.99

'''
this is the learning rate! a 1.0 learning rate would mean that the old q-value
would be entirely replaced. a 0.0 learning rate would mean that the q-value would
never update. i set it to 0.2 to allow the agent to learn, but not too quickly
'''
ALPHA = 0.1

'''
this is the initial probability of picking a random action. 1.0 would mean 
completely random actions - which we want, so the agent can explore
'''
EPS_START = 1.0

'''
we want the agent to eventually stop exploring in the end
'''
EPS_END = 0.01

'''
this is the number of episodes over which the exploration prob decays from EPS_START to EPS_END.
we set it close to the total number of episodes so that the agent explores for most of training.
'''
EPS_DECAY_EPISODES = 30000

'''
after 2000 episodes, evaluate the performance
'''
EVAL_INTERVAL = 2000

'''
number of episodes to use for each evaluation
'''
EVAL_EPISODES = 50

'''
cap the number of steps (don't want agent to keep moving forever)
'''
MAX_STEPS = 50 

'''
number of attempts before we call it a day
'''
FIND_SUCCESS_MAX_TRIES = 50


PLAYBACK_FPS = 6
WINDOW_SCALE = 1

ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


#########
# utils
########

def make_env(render_mode=None, seed=SEED):
    env = gym.make(
        ENV_ID,
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode=render_mode
    )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


'''
calculates epsilon for the given episode
'''
def epsilon_by_episode(ep):
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    return EPS_START - (EPS_START - EPS_END) * (ep / EPS_DECAY_EPISODES)


'''
evaluates progress. create a fresh environment, see if episode ends and reward is 1.0.
return the success rate
'''
def evaluate_policy(Q, episodes=EVAL_EPISODES):
    env = make_env(seed=SEED+999)
    wins = 0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if done and r == 1.0:
                wins += 1
    env.close()
    return wins / episodes

'''
training! first create the environment, then initialize the Q-table to all zeros.
for each episode, reset to initial state, compute epsilon, then at each step, choose 
an action (may be random based on probability epsilon).
q-value is updated using the q-learning formula.
evaluate progress every EVAL_INTERVAL episodes.
'''
def train_q_learning():
    env = make_env()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    rng = np.random.default_rng(SEED)

    def select_action(state, epsilon):
        if rng.random() < epsilon:
            return env.action_space.sample()
        return int(np.argmax(Q[state]))

    best_eval = 0.0
    for ep in range(EPISODES):
        s, _ = env.reset()
        epsilon = epsilon_by_episode(ep)
        done = False
        while not done:
            a = select_action(s, epsilon)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            td_target = r + GAMMA * (0 if done else np.max(Q[s2]))
            Q[s, a] += ALPHA * (td_target - Q[s, a])
            s = s2

        if (ep + 1) % EVAL_INTERVAL == 0:
            score = evaluate_policy(Q, EVAL_EPISODES)
            best_eval = max(best_eval, score)
            print(f"Episode {ep+1:5d} | eps={epsilon:.3f} | eval success={score:.2%} | best={best_eval:.2%}")

    env.close()
    return Q

'''
after we have the learned q-table, we use a greedy method (pick the best action at each step) 
to choose our trajectory. we try this up to max_tries times to find a successful trajectory
'''
def find_successful_greedy_rollout(Q, max_tries=FIND_SUCCESS_MAX_TRIES):
    env = make_env(render_mode="rgb_array", seed=SEED+1234)
    trajectories = None

    for attempt in range(1, max_tries + 1):
        states, actions, rewards, frames = [], [], [], []
        s, _ = env.reset()
        done = False
        steps = 0

        frames.append(env.render())
        states.append(int(s))

        while not done and steps < MAX_STEPS:
            a = int(np.argmax(Q[s]))
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            steps += 1

            actions.append(a)
            rewards.append(float(r))
            states.append(int(s2))
            frames.append(env.render())

            s = s2

        # success if we terminated with reward 1.0
        if (len(rewards) > 0) and rewards[-1] == 1.0 and done:
            trajectories = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "frames": frames,
                "attempt": attempt,
            }
            break

    env.close()
    return trajectories


###########
# ui stuff
###########
def play_frames_once(frames, states, actions, rewards, fps=PLAYBACK_FPS):
    pygame.init()
    pygame.display.set_caption("FrozenLake: Successful Rollout")

    # size window from first frame
    f0 = frames[0]
    h, w, _ = f0.shape
    win_size = (int(w * WINDOW_SCALE), int(h * WINDOW_SCALE))
    screen = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    def draw_text_bar(step_idx):
        # translucent box
        bar_h = 56
        s = pygame.Surface((win_size[0], bar_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 120))
        screen.blit(s, (0, 0))

        # compose HUD lines
        if step_idx == 0:
            action_txt = "Start"
            r = 0.0
        else:
            action_txt = ACTION_NAMES[actions[step_idx - 1]]
            r = rewards[step_idx - 1]

        lines = [
            f"Step {step_idx}/{len(frames)-1} | State={states[step_idx]} | Action={action_txt} | Reward={r:.0f}",
            f"Slippery={IS_SLIPPERY} | Map={MAP_NAME}",
        ]
        y = 6
        for line in lines:
            img = font.render(line, True, (255, 255, 255))
            screen.blit(img, (6, y))
            y += 18

    # play frames once
    running = True
    step_idx = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if step_idx < len(frames):
            frame = frames[step_idx]
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.scale(surf, win_size)
            screen.blit(surf, (0, 0))
            draw_text_bar(step_idx)
            pygame.display.flip()
            clock.tick(fps)
            step_idx += 1
        else:
            # Hold the final frame for a moment, then quit.
            pygame.time.delay(1200)
            running = False

    pygame.quit()

def main():
    print("Training Q-learning agent...")
    Q = train_q_learning()

    np.set_printoptions(precision=3, suppress=True)
    print("\nLearned Q-table (rows=states, cols=actions [Left,Down,Right,Up]):\n", Q)

    print("\nSearching for a single successful greedy rollout...")
    traj = find_successful_greedy_rollout(Q)

    if traj is None:
        print("No successful rollout found within the attempt limit.")
        return

    print(f"Success found on attempt #{traj['attempt']} in {len(traj['actions'])} steps. Playing it once...")
    play_frames_once(traj["frames"], traj["states"], traj["actions"], traj["rewards"])

if __name__ == "__main__":
    main()
