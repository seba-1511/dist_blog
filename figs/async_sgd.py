#!/usr/bin/env python

from seb.plot import Drawing, Animation, Plot, MAUREENSTONE_COLORS

PLOT_WIDTH = 7200
PLOT_HEIGHT = 5400
PLOT_HEIGHT = 3900
TEXT_SIZE = 20

# X_MIN, X_MAX = -12.0 * PLOT_WIDTH / 7200.0, 12.0 * PLOT_WIDTH / 7200.0
# Y_MIN, Y_MAX = -12.0 * PLOT_HEIGHT / 5400.0, 12.0 * PLOT_HEIGHT / 5400.0
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -10.0, 10.0

REPLICA_WIDTH = 6.0
REPLICA_HEIGHT = REPLICA_WIDTH / 1.9

R1_X, R1_Y = 0.0, 5.0
R2_X, R2_Y = -4.5, -5.0
R3_X, R3_Y = 4.5, -5.0
R1_COLOR, R2_COLOR, R3_COLOR = MAUREENSTONE_COLORS[0:3]

GLOBAL_COLOR = MAUREENSTONE_COLORS[4]
REPLICAS = [{
                'x': 0.0,
                'y': 5.0,
                'color': MAUREENSTONE_COLORS[0],
            }, {
                'x': -4.5,
                'y': -5.0,
                'color': MAUREENSTONE_COLORS[1],
            }, {
                'x': 4.5,
                'y': -5.0,
                'color': MAUREENSTONE_COLORS[2],
            }, ]


def replica_draw(graph, r):
    # r = REPLICAS[r]
    graph.fancybox(r['x'], r['y'], REPLICA_WIDTH, REPLICA_HEIGHT, fill=True,
                   linewidth=2.0, color=r['color'], alpha=0.8)
    graph.text('Replica', (r['x'] - 0.92, r['y'] - 0.0), size=TEXT_SIZE)


def replica_text(graph, r, text='', length=1.0):
    replica_text_color(graph, r, text, length, color=r['color'])


def replica_text_color(graph, r, text='', length=1.0, color=GLOBAL_COLOR):
    graph.text(
        text, (r['x'] - length, r['y'] - REPLICA_HEIGHT / 2.5), size=TEXT_SIZE - 2, color=color)


def add_instruction(graph, text):
    graph.set_title(text)


if __name__ == '__main__':
    anim = Animation(fps=0.35)

    graph = Drawing('Synchronous SGD', PLOT_HEIGHT, PLOT_WIDTH)
    graph.set_scales('linear', 'linear')
    graph.set_lims(x=(X_MIN, X_MAX), y=(Y_MIN, Y_MAX))

    # Draw replicas
    for r in REPLICAS:
        replica_draw(graph, r)
    anim.add_frame(graph.numpy())

    text_length = REPLICA_WIDTH / 2.0 - 0.1

    # Write initial text
    add_instruction(graph, 'Start with $W_t$')
    for r in REPLICAS:
        replica_text_color(graph, r, '$W_t$,', length=text_length)
    anim.add_frame(graph.numpy())
    text_length -= 0.7

    add_instruction(
        graph, 'Compute $\\nabla \mathcal{L}$ with local minibatch')
    for r in REPLICAS:
        replica_text(graph, r, '$\\nabla \mathcal{L}$,', length=text_length)
    anim.add_frame(graph.numpy())
    text_length -= 0.95

    add_instruction(graph, 'Compute $\\Delta W_t^L$ with favorite optimizer')
    for r in REPLICAS:
        replica_text(graph, r, '$\\Delta W_t^L$,', length=text_length)
    anim.add_frame(graph.numpy())
    text_length -= 1.45

    add_instruction(graph, 'AllReduce the $\\Delta W_t^L$ across replicas')
    for r in REPLICAS:
        replica_text_color(graph, r, '$\\Delta W_t^G$,', length=text_length)
    graph.annotate('', (-4.0, -7), (4.0, -7), rad=0.3, shape='<->')
    graph.annotate('', (-3.2, 5), (-5.5, -3.2), rad=0.3, shape='<->')
    graph.annotate('', (5.5, -3.2), (3.2, 5), rad=0.3, shape='<->')
    anim.add_frame(graph.numpy())
    text_length -= 1.4

    add_instruction(
        graph, 'Update to $W_{t+1}$ with normalized  $\\Delta W_t$')
    for r in REPLICAS:
        replica_text_color(graph, r, '$W_{t+1}$', length=text_length)
    anim.add_frame(graph.numpy())
    text_length -= 1.3

    graph.save('./sync.png')
    graph.save('./sync.pdf')
    anim.save('./sync.gif')
    anim.save('./sync.mp4')
