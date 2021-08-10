from ai2thor.controller import Controller
import time

controller = Controller(scene='FloorPlan1')

# get all pickupable objects
pickupable_objs = [
    obj['objectId'] for obj in controller.last_event.metadata['objects']
    if obj['pickupable']]

# choose an object to pickup (and pick it up)
obj_id_to_pickup = pickupable_objs[0]
pickup_event = controller.step(
    'PickupObject', objectId=obj_id_to_pickup, forceAction=True)

# metadata to determine if objectId was actually picked up
if (inventory := pickup_event.metadata['inventoryObjects']):
    picked_up_id = inventory[0]['objectId']
controller.step(action="LookUp")
time.sleep(10)


    # hide and unhide the object
    #hide_event = controller.step('HideObject', objectId=picked_up_id)
    #unhide_event = controller.step('UnhideObject', objectId=picked_up_id)
