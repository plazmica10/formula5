#%% Importing Libraries
from keras.models import Sequential
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten,Dropout, Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from collections import deque
import gymnasium as gym
import cv2 as cv

#%% util functions
def convert_to_grayscale(state):
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    return state.astype(float)/255.0
# This function transforms the deque of frames into a single frame stack
def get_state_frames(deque):
    frame_stack = np.array(deque)
    # Convert to format that the model expects as input because frame_stack is in the shape of (frame_stack_num, 96, 96)
    return np.transpose(frame_stack, (1, 2, 0))
'''
From gymnasium about the map:

Some indicators are shown at the bottom of the window along with the
state RGB buffer. From left to right: true speed, four ABS sensors,
steering wheel position, and gyroscope.
'''
render              = True
batch_size          = 64
start_ep            = 1     #Starting episode
final_ep            = 600   #Number of episodes
skip_frames         = 2     #Agent doesnt need to make decision for every frame so we skip some frames
save_freq           = 50    #Frequency of episodes on which we save the model
update_freq         = 5     #Frequency on which we update the target model
epsilon             = 1     #Initial epsilon value   
#%%  Agent Class
class CarModel:
    def __init__(self,epsilon = 1.0, gamma = 0.95, num_frames = 3,memory_size = 5000, epsilon_min = 0.1,epsilon_decay = 0.9999,learning_rate = 0.001,
        #combinations of (Steering, Gas, Break) -1 is left, 0 is straight, 1 is right
        action_space         = [ (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), (-1, 1,   0), (0, 1,   0), (1, 1,   0), (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]):
        self.memory          = deque(maxlen=memory_size)
        #exploration rate
        self.epsilon         = epsilon
        #discount rate
        self.gamma           = gamma
        self.num_frames      = num_frames
        self.action_space    = action_space
        self.learning_rate   = learning_rate
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.model           = self.create_DQN()
        self.target_model    = self.create_DQN()
        self.update_target_model()
    #%% Neural Network Model for Deep Q Network
    def create_DQN(self):
        model = Sequential()
        model.add(Input(shape=(96, 96, self.num_frames)))
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dropout(0.4))
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space)))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))
        
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.target_model.save_weights(name)
    #%% test
    def move(self, state):
            """
            Selects an action based on the given state.

            Parameters:
            state (numpy.ndarray): The current state of the environment.

            Returns:
            int: The index of the selected action.
            """
            if np.random.rand() > self.epsilon:
                q_values = self.model.predict(np.expand_dims(state, axis=0),verbose=0)
                # print(q_values[0])
                action_index = np.argmax(q_values[0])
            else:
                action_index = random.randrange(len(self.action_space))
            return self.action_space[action_index]
#%% train
    def train(self, batch_size):
        """
        Replays a batch of experiences and trains the model.
        Args:
            batch_size (int): The number of experiences to replay.
        Returns:
            None
        """
        experiences = random.sample(self.memory, batch_size)
        # train_state is the state that the model has encountered
        train_state = []
        # target Q-values for the actions taken in the state
        train_target = []
#%% bellman
        for state, action_index, reward, next_state, done in experiences:
            # print(action_index)
            target = self.model.predict(np.expand_dims(state, axis=0),verbose=0)[0]
            if done:
                target[action_index] = reward
            else:
                #bellman equation
                next_state_q_values = self.target_model.predict(np.expand_dims(next_state, axis=0),verbose=0)[0]
                # print(next_state_q_values)
                target[action_index] = reward + self.gamma * np.amax(next_state_q_values)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = gym.make('CarRacing-v2',render_mode='human')
agent = CarModel(epsilon=epsilon)

if __name__ == '__main__':
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