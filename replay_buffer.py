import random
from collections import namedtuple, deque

class ReplayBuffer(object):
    def __init__(self, size, batch_size):
        """
        Creates an instance of the Replay buffer which is used for storing tuple experiences.

        Args:
            size (int): Buffer maximum size.
            batch_size (int): Batch size.
        """
        self._buffer = []
        self._next_ind = 0
        self._maxsize = size
        self._experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._batch_size = batch_size
        #self.buffer = deque(maxlen=max_size)
        
    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self._buffer)
    
    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the buffer.
        """ 
        experience = self._experience(state, action, reward, next_state, done)
        
        if self._next_ind >= len(self._buffer):
            self._buffer.append(experience)
        else:
            self._buffer[self._next_ind] = experience
            
        self._next_ind = (self._next_ind + 1) % self._maxsize
        
    def sample(self):
        """
        Samples a random batch from the buffer.
        """
        indxes = [random.randint(0, len(self._buffer) - 1) for _ in range(self._batch_size)]
        batch = [self._buffer[i] for i in indxes]
        return batch
