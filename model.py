#%% Importing Libraries
from keras.models import Sequential
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten,Dropout, Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random

'''
From gymnasium about the map:

Some indicators are shown at the bottom of the window along with the
state RGB buffer. From left to right: true speed, four ABS sensors,
steering wheel position, and gyroscope.
'''
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
