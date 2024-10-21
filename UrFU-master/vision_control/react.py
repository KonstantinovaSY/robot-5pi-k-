from time import sleep

from movement_control.catch import make_a_move_to_ball, come_at_distance, rotate_to_pix, make_a_move_to_cube
from movement_control.move_commands import up, catch_ball, catch_box, straight_servo, prepare_to_catch, right

catch_methods = {0: make_a_move_to_ball, 8: make_a_move_to_cube}
catching = {'box': catch_box, 'ball': catch_ball}


def react():
    f = open('cur_inf.txt', encoding='UTF-8')
    current_pos = list(map(float, f.readline().split()))
    try:
        current_pos = current_pos[:-1], current_pos[-1]
    except:
        return 0
    if current_pos[-1] != -1:
        dots, cls_id = current_pos
        if not (dots == -1 or dots == 12):
            res = catch_methods[cls_id](*dots)
            print(res)
            if res[0] == 'catch':
                come_at_distance(1)
                prepare_to_catch()
                straight_servo()
                come_at_distance(6)
                sleep(0.4)
                catching[res[1]]()
                up()
                return 0.5
            elif res[0] == 'rotate':
                rotate_to_pix(*res[1:])
                return 0.1
            elif res[0] == 'come':
                come_at_distance(res[1])
                return 1
    else:
        pass
        # rotate_to_pix(0, 600)


    return 0.5
while True:
    t = react()
    sleep(t + 0.05)
