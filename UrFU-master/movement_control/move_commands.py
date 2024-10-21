import asyncio
import time
from sys import byteorder
def num_to_time_bytes(num):
    num = bin(num)[2:]
    marc = len(num) - 1
    res1 = ''
    res2 = ''
    for i in range(8):
        if marc >= 0:
            res1 += num[marc]
        else:
            res1 += '0'

        marc -= 1
    for i in range(8):
        if marc >= 0:
            res2 += num[marc]
        else:
            res2 += '0'
        marc -= 1
    res1, res2 = res1[::-1], res2[::-1]
    return int(res2, 2).to_bytes(1, byteorder='big')+ int(res1, 2).to_bytes(1, byteorder='big')

from connection import create_connection, send_command, host, port

forward = b'\xFF\x00\x01\x00\xFF'
stop = b'\xFF\x00\x00\x00\xFF'
turn_left = b'\xFF\x00\x03\x00\xFF'
turn_right = b'\xFF\x00\x04\x00\xFF'
reverse = b'\xFF\x00\x02\x00\xFF'

connect = create_connection(host, port)

def costyl_fw():
    send_command(connect, forward)


def set_speed(speed):
    speed = int(speed)
    setfullspeed = (b'\xFF\x02\x01' + speed.to_bytes(1, byteorder='big') + b'\xFF' +
                    b'\xFF\x02\x02' + speed.to_bytes(1, byteorder='big') + b'\xFF')
    send_command(connect, setfullspeed)

def fw(t):
    t *= 1000
    t = int(t)
    send_command(connect, b'\xff\xfd' + num_to_time_bytes(t) + b'\xff')


def stop_moving():
    send_command(connect, stop)

def back(t):
    t *= 1000
    t = int(t)
    send_command(connect, b'\xff\xf4' + num_to_time_bytes(t) + b'\xff')


def left(t):
    t *= 1000
    t = int(t)
    send_command(connect, b'\xff\xf5' + num_to_time_bytes(t) + b'\xff')


def right(t):
    t *= 1000
    t = int(t)
    send_command(connect, b'\xff\xf6' + num_to_time_bytes(t) + b'\xff')

def set_servo_angle(angle, num):
    send_command(connect, b'\xFF\x01' + int(num).to_bytes(1, byteorder='big') +
                 int(angle).to_bytes(1, byteorder='big')
                 + b'\xFF')
    time.sleep(0.5)

def normal():
    set_servo_angle(75, 7)
    set_servo_angle(20, 8)
    set_servo_angle(60, 4)
    set_servo_angle(80, 4)
    set_servo_angle(50, 3)
    set_servo_angle(180, 2)
    set_servo_angle(160, 1)


def straight_servo():
    set_servo_angle(93, 3)
    set_servo_angle(150 , 2)
    set_servo_angle(93, 1)
    set_servo_angle(180, 2)

    set_servo_angle(87, 1)
lights_on = 0
def turn_onoff_lights():
    global lights_on
    if lights_on:
        send_command(connect, b'\xff\x40\x09\x00\xff')
        lights_on = 0
    else:
        send_command(connect, b'\xff\x40\x09\x08\xff')
        lights_on = 1

def prepare_to_catch():
    set_servo_angle(93, 3)
    set_servo_angle(50, 4)

def catch_ball():
    set_servo_angle(87, 4)

def catch_box():
    set_servo_angle(72, 4)


def up():
    for j in range(120, 180, 20):
        set_servo_angle(j, 1)
    set_servo_angle(120, 2)

def put():
    set_servo_angle(55, 4)

def scenario1():
    prepare_to_catch()
    straight_servo()
if __name__ == '__main__':
    send_command(connect, b'\xff\x40\x09\x08\xff')

