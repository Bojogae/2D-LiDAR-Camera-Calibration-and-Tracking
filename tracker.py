

class Tracking:
    def __init__(self, id, ekf, dist):
        self.id = id
        self.ekf = ekf
        self.last_nearst_distance = dist


    def getPosition(self):
        return (self.ekf.state[0], self.ekf.state[1])