class Agent:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def act(self, observation):
        return self.brain.get_action(observation)
