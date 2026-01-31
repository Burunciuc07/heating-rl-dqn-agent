from environment import Building


env = Building(dynamic=True, eval=False)
s = env.reset()
print("init price:", s[3])

for _ in range(5):
    s, r, done = env.step((0,2))
    print("t", env.time, "T", s[1], "sun", s[2], "price", s[3], "done", done)
env = Building(dynamic=True, eval=False)
env.reset()
while True:
    _, _, done = env.step((0,2))
    if done:
        print("episode finished at time:", env.time)
        break
