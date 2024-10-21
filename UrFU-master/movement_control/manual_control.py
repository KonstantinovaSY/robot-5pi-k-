from time import sleep

import keyboard  # using module keyboard

from connection import send_command
from move_commands import prepare_to_catch, set_speed, fw, left, back, right, stop_moving, scenario1, put, normal, connect, host, port, catch_ball, up
from movement_control.move_commands import turn_right, turn_onoff_lights

set_speed(40)

active = 'l'
micro_tik = 0.1
iterr = 0
while True:  # making a loop
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('a'):  # if key 'q' is pressed
            if active != 'l':
                left(micro_tik)
                sleep(micro_tik)
            active = 'l'
        elif keyboard.is_pressed('d'):  # if key 'q' is pressed
            if active != 'r':
                right(micro_tik)
                sleep(micro_tik)
            active = 'r'
        elif keyboard.is_pressed('w'):  # if key 'q' is pressed
            if active != 'w':
                fw(micro_tik)
                sleep(micro_tik)
            active = 'w'
        elif keyboard.is_pressed('s'):  # if key 'q' is pressed
            if active == 's':
                back(micro_tik)
                sleep(micro_tik)
            active = 's'
        elif keyboard.is_pressed('h'):
            scenario1()
        elif keyboard.is_pressed('j'):
            put()
        elif keyboard.is_pressed('n'):
            normal()
        elif keyboard.is_pressed('c'):
            catch_ball()
            up()
        elif keyboard.is_pressed('p'):
            prepare_to_catch()
        elif keyboard.is_pressed('u'):
            start_autocatch()
        else:
            if active != ' ':
                stop_moving()
            active = ' '
    except:
        break  # if user pressed a key other than the given key the loop will break
