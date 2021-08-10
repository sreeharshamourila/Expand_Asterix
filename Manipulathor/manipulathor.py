from ai2thor.controller import Controller
import gym
import time
import os
import json
from addict import Dict
from PIL import Image
import cv2

import numpy as np


def print_grid(grid):
    for i in range(0, len(grid)):
        print(grid[i])


class CustomGridEnv(gym.Env):
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = self._read_config(config_file)
        self.name = 'robot_gridworld'
        self.reward_range = [0, 1]
        # noinspection PyArgumentList
        self.ACTION = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }
        self.action_space = gym.spaces.Discrete(len(self.ACTION))
        self.spec = Dict({'name': self.name,
                          'action_space': self.action_space,
                          'observation_space': self.observation_space,
                          'reward_range': self.reward_range})

        super(CustomGridEnv, self).__init__()
        self.goal = [5, -3, 0]
        width = self.config['layout_width']
        height = self.config['layout_height']
        zoom = self.config['zoom']
        final_width = int(width) * 30
        final_height = int(height) * 30
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(final_width, final_height, 3),
                                                dtype=np.uint8)
        self.controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene="FloorPlan203",
            visibilityDistance=1.5,
            GridSize=0.25,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=300,
            height=300,
            fieldOfView=60
        )
        # c
        self.controller.reset(scene="FloorPlan15", fieldOfView=80)
        positions = self.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        self.controller.step(action="SetHandSphereRadius",radius=0.49)
        print(positions)
        position=dict(x=-2.5, y=0.914, z=0.5)
        print(position)
        event=self.controller.step(action="Teleport", position=position,rotation=dict(x=0,y=90,z=0))
        #self.controller.step(
        #    action="MoveArm",
        #    position=dict(x=0, y=0.2,z=0.8),
        #    coordinateSpace="armBase",
        #    restrictMovement=False,
        #    speed=1,
        #    fixedDeltaTime=0.02
        #)
        event=self.controller.step(action="LookDown")
        pickupable_objs = [
            obj['objectId'] for obj in self.controller.last_event.metadata['objects']
            if obj['pickupable']]

        # choose an object to pickup (and pick it up)
        print(pickupable_objs)
        obj_id_to_pickup = pickupable_objs[1]
        pickup_event = self.controller.step(
            'PickupObject', objectId=obj_id_to_pickup, forceAction=True)


        self.controller.step(
            action="PickupObject",
            objectId="Knife|-01.36|+00.88|+00.50",
            forceAction=True
        )
        #event = self.controller.step(action="LookUp")
        event = self.controller.step(action="LookDown")
        im2=Image.fromarray(event.frame)
        im2.show()
        time.sleep(2)
        #for i in range(0,len(positions)):
        #    print(positions[i])
        #    self.controller.step(action="Teleport", position=positions[i])
        #    time.sleep(1)

        mini_x = 10000
        maxi_x = -10000
        mini_z = 10000
        maxi_z = -10000
        for i in positions:
            if i['x'] > maxi_x:
                maxi_x = i['x']
            if mini_x > i['x']:
                mini_x = i['x']
            if i['z'] > maxi_z:
                maxi_z = i['z']
            if mini_z > i['z']:
                mini_z = i['z']
        m = (maxi_x - mini_x) / 0.25
        n = (maxi_z - mini_z) / 0.25

        print(mini_x,mini_z,maxi_x,maxi_z)

        self.grid = [[0] * int(n) for _ in range(0, int(m))]

        self.X = []
        self.Y = []
        for i in range(0, int(m)):
            self.X.append(mini_x + (i * 0.25))
        for i in range(0, int(n)):
            self.Y.append(mini_z + (i * 0.25))

        self.initial_index = [m//2, n//2]

        self.current_X = int(self.initial_index[0])
        self.current_Y = int(self.initial_index[1])
        #self.controller.reset(scene="FloorPlan15", fieldOfView=80)
        #self.controller.step(action="Teleport",position=dict(x=-2,y=0.9,z=0))
        #self.controller.step(action="MoveAhead")
        self.grid[self.current_X][self.current_Y] = 1

    def step(self, action=None):
        action = self.ACTION[int(action)]
        current_index_position = [self.current_X, self.current_Y]
        self.grid[self.current_X][self.current_Y] = 0
        self.reward = 0
        self.done = False
        info = {}
        self.obs = 0
        if action is not None:
            self.obs, final_index_position, final_target_position = self.take_action(action, self.X, self.Y,
                                                                                     current_index_position)
            self.current_X = final_index_position[0]
            self.current_Y = final_index_position[1]
            self.grid[self.current_X][self.current_Y] = 1
            self.reward, self.done = self.get_reward_done(self.goal,
                                                          [self.X[self.current_X], self.Y[self.current_Y], 0])
            return self.obs, self.reward, self.done, info
        else:
            event = self.controller.reset()
            self.obs = event.frame
            return self.obs, self.reward, self.done, info

    def reset(self):
        event = self.controller.reset()
        return event.frame

    def take_action(self, direction, X, Y, current_index_position):
        A = int(current_index_position[0])
        B = int(current_index_position[1])
        # print("A,B",A,B)
        if direction is 'UP':
            ##assigning the co-ordinates using index positions
            position = [X[A], Y[B], 0]
            if A < (len(X) - 1):
                final_index_position = [int(A + 1), int(B)]
                final_target_position = [X[A + 1], Y[B], 0]
                event = self.controller.step(action="MoveAhead")
                # print(final_index_position,final_target_position)
                return event.frame, final_index_position, final_target_position
            else:
                # print("grid limit in X+ direction reached")
                event = self.controller.reset()
                current_index_position[0] = int(self.initial_index[0])
                current_index_position[1] = int(self.initial_index[1])
                return event.frame, current_index_position, position
        elif direction is 'DOWN':
            position = [X[A], Y[B], 0]
            # print(position)
            if (A != 0):
                final_index_position = [A - 1, B]
                final_target_position = [X[A - 1], Y[B], 0]
                event = self.controller.step(action="MoveBack")
                # print(np.asarray(event.frame))

                # print(final_index_position,final_target_position)
                return event.frame, final_index_position, final_target_position
            else:
                print("grid limit in X- direction reached")
                event = self.controller.reset()
                current_index_position[0] = int(self.initial_index[0])
                current_index_position[1] = int(self.initial_index[1])
                return event.frame, current_index_position, position
        elif direction is 'RIGHT':
            position = [X[A], Y[B], 0]
            # print(position)
            if B is not 0:
                final_index_position = [A, B - 1]
                final_target_position = [X[A], Y[B - 1], 0]
                self.controller.step(action="RotateRight")
                self.controller.step(action="MoveAhead")
                event = self.controller.step(action="RotateLeft")
                return event.frame, final_index_position, final_target_position
            else:
                # print("grid limit in Y- direction reached")
                event = self.controller.reset()
                current_index_position[0] = int(self.initial_index[0])
                current_index_position[1] = int(self.initial_index[1])
                return event.frame, current_index_position, position
        elif direction is 'LEFT':
            position = [X[A], Y[B], 0]
            if B != (len(Y) - 1):
                final_index_position = [A, B + 1]
                final_target_position = [X[A], Y[B + 1], 0]
                self.controller.step(action="RotateRight", degrees=90)
                self.controller.step(action="MoveBack")
                event = self.controller.step(action="RotateLeft", degrees=90)
                return event.frame, final_index_position, final_target_position
            else:
                print("grid limit in Y+ direction reached")
                event = self.controller.reset()
                current_index_position[0] = int(self.initial_index[0])
                current_index_position[1] = int(self.initial_index[1])
                return event.frame, current_index_position, position

    def _read_config(self, config_file):
        config = {}
        if config_file is None:
            config_file = os.path.dirname(os.path.abspath(__file__)) + '/environments/default_config_robot.json'

        with open(config_file) as f:
            config_parser = json.loads(f.read())

        config['layout_width'] = config_parser['layoutWidth']
        config['layout_height'] = config_parser['layoutHeight']

        zoom = config_parser['zoom']
        config['zoom'] = zoom
        config['frame_time'] = config_parser['frameTime']

        return config

    def set_last_rgb_obs(self, obs):
        pass

    def get_reward_done(self, position, goal):
        if goal is position:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return reward, done
    def get_top_down_view(self):
        agent_view = self.controller.last_event.frame
        event = self.controller.step('ToggleMapView')
        map_view = event.frame
        im = Image.fromarray(map_view)
        return im
    def get_third_party_camera_frame(self):
        event=self.controller.step(action="AddThirdPartyCamera", rotation=dict(x=0, y=90, z=0), position=dict(x=5.0, z=-2.0, y=2.0))
        im=event.third_party_camera_frames[0]
        im2 = Image.fromarray(im)
        return im2

    def material_randomization(self,use_train=False,use_val=False,use_test=False,in_room=None):
        self.controller.step(
            action="RandomizeMaterials",
            useTrainMaterials=use_train,
            useValMaterials=use_val,
            useTestMaterials=use_test,
            inRoomTypes=in_room
        )

    def lightening_randomization(self,brightness=(0.5,1.5),randomize_colour=True,hue=(0,1),saturation=(0.5,1),synchronized=False):
        self.controller.step(
            action="RandomizeLighting",
            brightness=brightness,
            randomizeColor=randomize_colour,
            hue=hue,
            saturation=saturation,
            synchronized=synchronized
        )

    def initial_random_spawn(self,random_seed=0,force_visible=False,num_placement_attempts=5,place_stationary=True,numDuplicatesOfType=None,excluded_receptacles=None,excluded_obj_ids=None):
        numDuplicatesOfType = [
            {
                "objectType": "Statue",
                "count": 20
            },
            {
                "objectType": "Bowl",
                "count": 20
            }
        ]
        excluded_receptacles = {"CounterTop", "DiningTable"}
        excluded_obj_ids = ["Apple|1|1|2"]
        self.controller.step(action="InitialRandomSpawn",
                        randomSeed=random_seed,
                        forceVisible=force_visible,
                        numPlacementAttempts=num_placement_attempts,
                        placeStationary=place_stationary,
                        numDuplicatesOfType=numDuplicatesOfType,
        excludedReceptacles = excluded_receptacles,
        excludedObjectIds = excluded_obj_ids
        )

    def colour_randomization(self):
        self.controller.step(action="RandomizeColors")

    def get_obj_n_ids(self):
        event=self.controller.last_event
        obj=event.metadata["objects"]
        for i in range(0,len(obj)):
            print(obj[i]['name'],obj[i]['pickupable'])
        #print(event.metadata["objects"])








RobotEnv=CustomGridEnv()

#RobotEnv.colour_randomization()
im=RobotEnv.get_top_down_view()
im.show()
RobotEnv.get_obj_n_ids()
#exit()
#cv2.waitKey(1000)

#exit()
pressed_key = cv2.waitKey(0)
q=0
while(pressed_key != ord('a') and q !=0):
    if pressed_key==ord('m'):
        RobotEnv.material_randomization()
    elif pressed_key==ord('l'):
        RobotEnv.lightening_randomization()
    elif pressed_key==ord('c'):
        RobotEnv.colour_randomization()
    elif pressed_key==ord('s'):
        RobotEnv.initial_random_spawn()

#im=RobotEnv.get_top_down_view()
#im.show()



#im=RobotEnv.get_third_party_camera_frame()
#im.show()


# RobotEnv.step(action="UP")
# time.sleep(3)
# RobotEnv.step(action="DOWN")
# time.sleep(3)
# RobotEnv.step(action="LEFT")
# time.sleep(3)
# RobotEnv.step(action="RIGHT")
# time.sleep(3)
# RobotEnv.step(action="UP")
# time.sleep(3)
# for i in range(0,500):
# RobotEnv.step()