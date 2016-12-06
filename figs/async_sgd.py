#!/usr/bin/env python

from seb.plot import Drawing, Plot, MAUREENSTONE_COLORS
# from seb.plot import Animation

PLOT_WIDTH = 7200
PLOT_HEIGHT = 5400
PLOT_HEIGHT = 3900
TEXT_SIZE = 20

X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -10.0, 10.0

REPLICA_WIDTH = 5.5
REPLICA_HEIGHT = 6.0 / 1.9

SM_WIDTH = 16.0
SM_HEIGHT = REPLICA_HEIGHT

R1_X, R1_Y = 0.0, 5.0
R2_X, R2_Y = -4.5, -5.0
R3_X, R3_Y = 4.5, -5.0
R1_COLOR, R2_COLOR, R3_COLOR = MAUREENSTONE_COLORS[0:3]
ALPHA, ALPHA_FADED = 0.8, 0.2

SM_COLOR = MAUREENSTONE_COLORS[1]
SM_X, SM_Y = 0.0, Y_MAX - SM_HEIGHT

GLOBAL_COLOR = MAUREENSTONE_COLORS[4]
REPLICAS = [{
                'x': 0.0,
                'y': -5.0,
                'color': MAUREENSTONE_COLORS[0],
                'faded': False,
            }, {
                'x': -6.5,
                'y': -5.0,
                'color': MAUREENSTONE_COLORS[0],
                'faded': True,
            }, {
                'x': 6.5,
                'y': -5.0,
                'color': MAUREENSTONE_COLORS[0],
                'faded': True,
            }, ]


def shared_memory_draw(graph):
    graph.fancybox(SM_X, SM_Y, SM_WIDTH, SM_HEIGHT, fill=True, color=SM_COLOR, linewidth=2.0, alpha=0.8)
    graph.text('Shared Memory', (SM_X - 2.2, SM_Y-0.3), size=TEXT_SIZE+2)

def replica_draw(graph, r):
    alpha = ALPHA
    offset = -0.1
    if r['faded']:
        alpha = ALPHA_FADED
        # offset = 0.2
    graph.fancybox(r['x'], r['y'], REPLICA_WIDTH, REPLICA_HEIGHT, fill=True,
                   linewidth=2.0, color=r['color'], alpha=alpha)
    graph.text('Replica', (r['x'] - 0.92, r['y'] - offset), size=TEXT_SIZE, alpha=alpha)


def replica_text(graph, r, text='', length=1.0):
    replica_text_color(graph, r, text, length, color=r['color'])


def replica_text_color(graph, r, text='', length=1.0, color=GLOBAL_COLOR, size=None):
    alpha = ALPHA
    if size is None: 
        size = TEXT_SIZE - 2
    if r['faded']:
        alpha = ALPHA_FADED
    graph.text(text, (r['x'] - length, r['y'] - REPLICA_HEIGHT / 2.7), size=size, color=color, alpha=alpha)


def add_instruction(graph, text):
    graph.set_title(text)


if __name__ == '__main__':
    # anim = Animation(fps=0.35)

    graph = Drawing('Asynchronous SGD', PLOT_HEIGHT, PLOT_WIDTH)
    graph.set_scales('linear', 'linear')
    graph.set_lims(x=(X_MIN, X_MAX), y=(Y_MIN, Y_MAX))

    # Draw shared memory
    shared_memory_draw(graph)

    # Draw replicas
    for r in REPLICAS:
        replica_draw(graph, r)

    replica_text_color(graph, REPLICAS[1], 'Independent Work', length=2.0, color='black', size=TEXT_SIZE - 4)
    replica_text_color(graph, REPLICAS[2], 'Independent Work', length=2.0, color='black', size=TEXT_SIZE - 4)
    graph.annotate('', (-7, 5), (-7, -3.3), rad=0.0, shape='->', width=1.0)
    graph.annotate('', (-6, -3.3), (-6, 5), rad=0.0, shape='->', width=1.0)
    graph.annotate('', (7, 5), (7, -3.3), rad=0.0, shape='->', width=1.0)
    graph.annotate('', (6, -3.3), (6, 5), rad=0.0, shape='->', width=1.0)

    text_length = REPLICA_WIDTH / 2.0 - 0.1
    replica = REPLICAS[0]

    # anim.add_frame(graph.numpy())

    # Copy params
    add_instruction(graph, 'Copy $W^G$ into $W_t^L$')
    replica_text_color(graph, replica, '$W_t^L$,', length=text_length)
    graph.annotate('', (-2.9, 5), (-2.9, -4), rad=0.3, shape='->', width=1.3)
    # anim.add_frame(graph.numpy())
    text_length -= 0.85

    # Compute Gradients
    add_instruction(graph, 'Compute gradients $\\nabla_{W_t^L} \mathcal{L}$ locally')
    replica_text(graph, replica, '$\\nabla_{W_t^L} \mathcal{L}$,', length=text_length)
    # anim.add_frame(graph.numpy())
    text_length -= 1.7

    # Compute Update
    add_instruction(graph, 'Compute update $\\Delta W_{t+1}^L$ with favorite optimizer')
    replica_text(graph, replica, '$\\Delta W_{t+1}^L$,', length=text_length)
    # anim.add_frame(graph.numpy())
    text_length -= 1.8

    # Apply update globally
    add_instruction(graph, 'Apply local update $\\Delta W_{t+1}^L$ to global params $W^G$')
    replica_text_color(graph, replica, '$W^G$', length=text_length)
    graph.annotate('', (2.9, -4), (2.9, 5), rad=0.3, shape='->', width=1.3)
    # anim.add_frame(graph.numpy())
    text_length -= 0.8

    graph.save('./async.png')
    graph.save('./async.pdf')
    # anim.save('./async.gif')
    # anim.save('./async.mp4')
