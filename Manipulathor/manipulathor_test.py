import ai2thor
import copy
import time
import random
import ai2thor.controller
from datetime import datetime
import cv2
import os
import matplotlib.pyplot as plt
import os
from jupyter_helper import initialize_arm, only_reset_scene, transport_wrapper, ADITIONAL_ARM_ARGS, execute_command

screen_size=900
controller = ai2thor.controller.Controller(gridSize=0.25,
                width=255, height=255, agentMode='arm', fieldOfView=100,
                agentControllerType='mid-level',
                server_class=ai2thor.fifo_server.FifoServer,
                useMassThreshold = True, massThreshold = 10, renderDepthImage=True)


############################functions######################################################################################
def translate(action):
    translation = {
        'MoveArmHeightM': 'j',
        'MoveArmHeightP': 'u',
        'MoveArmXM': 'a',
        'MoveArmXP': 's',
        'MoveArmZM': 'z',
        'MoveArmZP': 'w',
        'MoveArmYM': '4',
        'MoveArmYP': '3',
        'MoveAhead': 'm',
        'RotateRight': 'r',
        'RotateLeft': 'l',
        '':'',
        'Done':'q',
        'PickUp':'p', 
        'Finish': 'q',
        'Drop': 'd',
    }
    return translation[action]

def run_action_sequence(controller, action_sequence, object_id=None, target_location=None, logger_number=0, translated=False, seq_num_start = 0):
    picked_up = False
    if translated:
        translated_sequence = action_sequence
    else:
        translated_sequence = [translate(action) for action in action_sequence if action != '']
    for (seq_number, seq) in enumerate(translated_sequence):
        execute_command(controller, seq,ADITIONAL_ARM_ARGS)
                
def manual_task(scene_name, logger_number =0, final=False):
    only_reset_scene(controller, scene_name)
    
    
    all_actions = []
    actions_ran_so_far = 0
    while(True):
        action = input()
        action = translate(action)
        if action == 'q':
            break
        all_actions.append(action)
        
        run_action_sequence(controller, [action], 
                            logger_number=logger_number, translated=True, seq_num_start=actions_ran_so_far)
        actions_ran_so_far += 1
        plt.cla()
        plt.imshow(controller.last_event.frame)
        plt.show()
        
    print(scene_name)
    print(all_actions)
###############################################################################################################################

manual_task(scene_name='FloorPlan1_physics',final=True)

exit()



###############################################################################################################################
####Incorporated the working model from previous script to test if its working in this code###################################
##############################################################################################################################

pickupable_objs = [
    obj['objectId'] for obj in controller.last_event.metadata['objects']
    if obj['pickupable']]

# choose an object to pickup (and pick it up)
obj_id_to_pickup = pickupable_objs[0]
pickup_event = controller.step(
    'PickupObject', objectId=obj_id_to_pickup, forceAction=True)

#controller.step(action="LookDown")
if (inventory := pickup_event.metadata['inventoryObjects']):
    picked_up_id = inventory[0]['objectId']
controller.step(action="LookUp")
time.sleep(2)

time.sleep(5)


