from model import CarModel
from collections import deque
import gymnasium as gym
import cv2 as cv
import numpy as np
import time

#%% utils 
def convert_to_grayscale(state):
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    return state.astype(float)/255.0
# transforms the deque of frames into a single frame stack
def get_state_frames(deque):
    frame_stack = np.array(deque)
    # Convert to format that the model expects as input because frame_stack is in the shape of (frame_stack_num, 96, 96)
    return np.transpose(frame_stack, (1, 2, 0))

#%% Parameters and Environment
# np.random.seed(0)
env = gym.make('CarRacing-v2',render_mode='human')
env.reset(seed=10)
#epsilon is 0 so that all actions are made by the agent
agent = CarModel(epsilon=0)
agent.load('./trained_models/model1.h5')

#%% PLAY THE GAME WITH THE TRAINED MODEL
init_state = env.reset(seed=10)
#init state is a tuple, init_state[0] is the state in format of (96, 96, 3) -> (height, width, color_channels)
# print(init_state)
init_state = convert_to_grayscale(init_state[0])
# print(init_state)
# print(init_state.shape)

start_time = time.time()
# print(start_time)
still_frames = 0
total_reward = 0
state_frame_stack_queue = deque([init_state]*agent.num_frames, maxlen=agent.num_frames)
time_frame_counter = 1

while True:
    env.render()

    current_state_frame_stack = get_state_frames(state_frame_stack_queue)

    action = agent.move(current_state_frame_stack)
    #print(action)
    #check if the action is to not move
    if action[1] == 0:
        still_frames += 1
    else:
        still_frames = 0
    
    next_state, reward, done, _,_ = env.step(action)
    total_reward += reward

    next_state = convert_to_grayscale(next_state)
    # print(next_state)
    # end_time = time.time()
    # print(end_time - start_time)
    # if compare_images(next_state, init_state) and end_time - start_time > 10:
    #     print("A lap has been completed")
    state_frame_stack_queue.append(next_state)

    if done or total_reward < -100 or still_frames > 100: break
    time_frame_counter += 1
env.close()