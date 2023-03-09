from snake_ai.networks.actor_critic import ActorCritic
from snake_ai.networks.icm import ICM

if __name__ == '__main__':
    import numpy as np

    b = np.zeros((50, 50))
    b[0, 0] = 1
    b = b.reshape((1, 50, 50, 1))
    c = np.zeros((1, 4))
    i = ICM(4, 50)
    print(i([c, b, b]))
    i.summary()
    i.plot()
    ac = ActorCritic(4, 50)
    ans = ac([b, c])
    print(ans[0], ans[1])
    ac.plot()