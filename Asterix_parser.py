import cv2
import numpy as np

def parse_asterix(RGB_img, templ_path):
    """
    Parse *RGB* (same format as from ALE) frame of Asterix into state dict
    using template matching over the image.
    "target"s are the rewarding objects and "demon"s are the punishing objects.
    Absent agent/objects are filled with [None x 4] to match shape.

    :param RGB_img: input game frame as from ALE. shape: 210x160x3.
    :param templ_path: path to all the templates for parsing.
                      Should include 2 for agents, 8 for targets, and 1 for demon.
    :return: frame_dict: {'agent': [y1, y2, x1, x2],                  (shape: 4,)
                          'target': [[y1, y2, x1, x2] for each lane], (shape: 8x4)
                          'demon': [[y1, y2, x1, x2] for each lane]}  (shape: 8x4)
             Note here the y-then-x convention follows matrix access convention.
             All coordinates are in lists.
    """

    # ================== helper functions =====================
    def match_templ(img, templ, threshold=0.6):
        """Helper function for parsing Asterix.
        Return bounding box of max matching object (assuming only one object present in the given img).
        For finding targets, which are never flipped.
        """
        # init return
        y1, y2, x1, x2 = None, None, None, None

        w, h = templ.shape[::-1]
        orig_res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
        orig_loc = np.where(orig_res > threshold)

        if len(orig_loc[0]) > 0:
            y1, x1 = np.unravel_index(np.argmax(orig_res, axis=None), orig_res.shape)
            y1, x1 = int(y1), int(x1)
            x2 = int(x1 + w - 1)
            y2 = int(y1 + h - 1)
        return y1, y2, x1, x2

    def match_templ_flip(img, templ, threshold=0.6):
        """Helper function for parsing Asterix.
        Template matching with both the original template & its mirror image (flipped across vertical axis).
        Return bounding box of max matching over both orig & mirror (assuming only one object present in the given img).
        If original matching exists, does not return flipped matching locations.
        For agent and demons, which have flipped instances.
        """
        # init return
        y1, y2, x1, x2 = None, None, None, None

        mirror_templ = cv2.flip(templ, 1)
        w, h = templ.shape[::-1]

        orig_res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
        orig_loc = np.where(orig_res > threshold)
        mirror_res = cv2.matchTemplate(img, mirror_templ, cv2.TM_CCOEFF_NORMED)
        mirror_loc = np.where(mirror_res > threshold)
        chosen_res = None

        # determine matching of which template to use
        if len(orig_loc[0]) > 0:
            # choose original matching with priority
            chosen_res = orig_res
            if len(mirror_loc[0]) > 0:
                # if both exist, choose one with max res
                if np.max(orig_res[orig_loc]) < np.max(mirror_res[mirror_loc]):
                    chosen_res = mirror_res
        # orig does not exist, but mirror exists
        elif len(mirror_loc[0]) > 0:
            chosen_res = mirror_res

        # get bounding box for return
        if chosen_res is not None:
            # returns index of max value in the matrix (y, x)
            y1, x1 = np.unravel_index(np.argmax(chosen_res, axis=None), chosen_res.shape)
            y1, x1 = int(y1), int(x1)
            x2 = int(x1 + w - 1)
            y2 = int(y1 + h - 1)
        return y1, y2, x1, x2

    # ==================== parsing asterix ========================
    # make frame BGR, so cv2 readable
    img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2GRAY)[:150, :]  # only take arena of 8 lanes

    # Initializing output for frame
    state = {'agent': [None for _ in range(4)],
             'target': [[None for _ in range(4)] for _ in range(8)],
             'demon': [[None for _ in range(4)] for _ in range(8)]}
    # for template matching in each lane
    lanes_y = {0: range(25, 40),
               1: range(40, 55),
               2: range(55, 72),
               3: range(72, 88),
               4: range(88, 104),
               5: range(104, 120),
               6: range(120, 135),
               7: range(135, 150)}

    # ===== agent =====
    # read templates
    agents = [cv2.imread(f'{templ_path}/agent{i}.png', 0) for i in range(2)]
    # check each agent template
    for agent in agents:
        y1, y2, x1, x2 = match_templ_flip(img, agent, 0.6)
        # if matching exists for this template
        if y1 is not None:
            state['agent'] = [y1, y2, x1, x2]
            # break out of template loop
            break

    # ===== demon: could be flipped =====
    # read template
    demon = cv2.imread(f'{templ_path}/demon.png', 0)
    # init list for all demons in frame
    coord_list = []
    detected = False
    # check each lane
    for lane_idx, lane_y in lanes_y.items():
        y1, y2, x1, x2 = match_templ_flip(img[lane_y, :], demon, 0.6)
        # no matching exists: add placeholder to maintain shape
        if y1 is None:
            coord_list.append([None, None, None, None])
        # matching exists for this lane
        else:
            coord_list.append([y1 + lanes_y[lane_idx][0], y2 + lanes_y[lane_idx][0], x1, x2])
            detected = True
    # put in frame
    if detected:
        state['demon'] = coord_list.copy()

    # ===== targets: never flipped =====
    # read templates
    targets = [cv2.imread(f'{templ_path}/target{i}.png', 0) for i in range(8)]  # 8 targets in total
    # check each target template
    for target in targets:
        # init list for all targets in frame
        coord_list = []
        detected = False
        # check each lane
        for lane_idx, lane_y in lanes_y.items():
            y1, y2, x1, x2 = match_templ(img[lane_y, :], target, 0.6)
            # no matching exists: add placeholder for shape
            if y1 is None:
                coord_list.append([None, None, None, None])
            # matching exists for this template
            else:
                coord_list.append([y1 + lanes_y[lane_idx][0], y2 + lanes_y[lane_idx][0], x1, x2])
                detected = True
        # only one kind of target can be present in a frame
        # --> if there exist matchings for this template, break out of template loop
        if detected:
            break
    # put in frame
    if detected:
        state['target'] = coord_list.copy()

    return state

new_image="/home/local/ASUAD/smourila/meta-sim/parse-atari/samples/asterix_sample.png"
templates_path="/home/local/ASUAD/smourila/meta-sim/parse-atari/asterix_templ"

im=cv2.imread(new_image)
result=parse_asterix(im,templates_path)
print(result)
