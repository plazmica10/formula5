#%% Importing Libraries
from model import CarModel
from collections import deque
import gymnasium as gym
import cv2 as cv
import numpy as np

#%% util functions
def convert_to_grayscale(state):
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    return state.astype(float)/255.0
# This function transforms the deque of frames into a single frame stack
def get_state_frames(deque):
    frame_stack = np.array(deque)
    # Convert to format that the model expects as input because frame_stack is in the shape of (frame_stack_num, 96, 96)
    return np.transpose(frame_stack, (1, 2, 0))

#%% Parameters and Environment
render              = True
batch_size          = 64
start_ep            = 1     #Starting episode
final_ep            = 600   #Number of episodes
skip_frames         = 2     #Agent doesnt need to make decision for every frame so we skip some frames
save_freq           = 50    #Frequency of episodes on which we save the model
update_freq         = 5     #Frequency on which we update the target model
epsilon             = 1     #Initial epsilon value   
env = gym.make('CarRacing-v2',render_mode='human')
agent = CarModel(epsilon=epsilon)

#%% Training
for episode in range(start_ep, final_ep+1):
    '''
    Number of consecutive frames that are stacked together to create the input to the model.
    This is done to give the model some information about the dynamics of the environment, i.e.,
     how the state is changing over time.
    '''
    init_state = env.reset()
    init_state = convert_to_grayscale(init_state[0])
    still_frames = 0
    reward_sum = 0
    negative_rewards = 0
    state_frame_stack_queue = deque([init_state]*agent.num_frames, maxlen=agent.num_frames)
    time_frame_counter = 1
    done = False
    while True:
        if render:
            env.render()

        current_state_frame_stack = get_state_frames(state_frame_stack_queue)
        action = agent.move(current_state_frame_stack)

        reward = 0
        for _ in range(skip_frames+1):
            next_state, r, done,_,_ = env.step(action)
            reward += r
            if done:
                break
        # Terminate the episode if the car is stuck
        negative_rewards = negative_rewards + 1 if time_frame_counter > 100 and reward < 0 else 0
        # Reward the model for pressing gas
        if action[1] == 1 and action[2] == 0:
            reward *= 1.5
        if action[1] == 0:
            still_frames += 1
        else:
            still_frames = 0

        reward_sum += reward
        next_state = convert_to_grayscale(next_state)
        state_frame_stack_queue.append(next_state)
        next_state_frame_stack = get_state_frames(state_frame_stack_queue)

        agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)
        if done or negative_rewards >= 25 or reward_sum < 0 or still_frames > 200: break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        time_frame_counter += 1

    if episode % update_freq == 0:
        agent.update_target_model()
    if episode % save_freq == 0:
        agent.save('./trained_models/model{}.h5'.format(episode))
env.close()