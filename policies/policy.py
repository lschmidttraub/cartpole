from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def act(self, obs, train=False):
        pass

    @abstractmethod
    def train(self):
        pass

    def store(self, transition):
        pass