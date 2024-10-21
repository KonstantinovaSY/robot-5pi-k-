from movement_control.catch import come_at_distance


def is_distance_good():
    return

def make_a_move_to_bring(dots):
    x, y, X, Y = dots
    x_mid = (x + X) // 2
    y_mid = (y + Y) // 2
    if not (380 < x_mid < 420):
        return 'rotate', x_mid, 400
    elif is_distance_good():
        return 'catch', 'put_in_basket'
    else:
        return 'come', 5

    come_at_distance(10)

