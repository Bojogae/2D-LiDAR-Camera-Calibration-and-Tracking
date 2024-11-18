class Postion:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]

    def tuple(self):
        return (self.x, self.y)

class Cluster:
    def __init__(self, label, pos):
        self.label = label
        position = Postion(pos)
        self.position = position