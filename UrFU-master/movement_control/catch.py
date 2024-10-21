import asyncio
import time

from movement_control.move_commands import fw, stop_moving, set_speed, back, left, \
    right

movement_coeff = 0.05
success_pts = list(map(float, '242.524169921875 231.10682678222656 '
                              '361.70269775390625 343.0059814453125'.split()))


def get_mid(dist):
    return 250


def count_dist_ball(square):  # возвращает дистанцию до объекта по площади на камере
    return -0.0015701334676403928 * square + 54.02883494683959


def good_speed(dist):
    if dist <= 33:
        return 35
    if dist >= 38:
        return 80
    return 36


def come_at_distance(d):
    set_speed(good_speed(d))
    time_sleep = abs(d * movement_coeff)
    if d > 36:
        time_sleep *= 1
    if d > 0:
        fw(time_sleep)
    else:
        back(time_sleep)
    time.sleep(time_sleep)


def get_good_delta(dist):
    return dist * 2


def rotate_to_pix(x, mid):
    set_speed(50)
    slee_time = max(0.02, abs(mid - x) / 3400)
    if x < mid:
        left(slee_time)
    else:
        right(slee_time)
    time.sleep(slee_time)

def make_a_move_to_ball(x, y, X, Y):
    width = X - x
    height = Y - y
    x_mid = x + width // 2
    y_mid = y + height // 2
    right_bound = 35
    left_bound = 34
    dist = count_dist_ball(width * height)
    mid = get_mid(dist)

    print('dist', count_dist_ball(width * height), height * width)
    delta = get_good_delta(dist)
    print('delta', delta)
    if y_mid < 50 and width * height < 3000 and 300 < x_mid < 500:
        return 'come', 35
    if mid - delta > x_mid or x_mid > mid + delta:
        return 'rotate', x_mid, mid
    elif left_bound < dist < right_bound:
        return 'catch', 'ball'
    else:
        out = dist - (left_bound + right_bound) / 2
        return 'come', out

def make_a_move_to_cube(x, y, X, Y):
    width = X - x
    height = Y - y
    x_mid = x + width // 2
    y_mid = y + height // 2
    dist = count_dist_ball(width * height)
    mid = get_mid(dist)
    if y_mid > 260:
        return 'come', -3
    elif not (280 < x_mid < 290):
        return 'rotate', x_mid, 285
    elif y_mid < 200:
        return 'come', 10
    elif y_mid < 220:
        return 'come', 4
    else:
        return 'catch', 'box'
if __name__ == '__main__':
    print(get_good_delta(33))
    print(good_speed(40))
