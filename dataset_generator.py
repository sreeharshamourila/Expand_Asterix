import gym
import cv2
env = gym.make('Asterix-ram-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        b=env.render(mode="rgb_array")
        #print(b)
        cv2.imwrite('/home/local/ASUAD/smourila/Asterix_images/'+str(t)+'.jpg', b)
        #cv2.imshow("image", b)
        #cv2.waitKey(3000)
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
