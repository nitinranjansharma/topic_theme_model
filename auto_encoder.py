# adding auto encoder class to train and get a dense embedding for clustering
# and figure out topic words and do theme classification

class AutoEncoder(object):
    def __init__(self, path):
        self.path = path

    def construct_model(self):

