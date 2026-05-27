import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_game(game, generations=150, guardar=None):

    figure, axis = plt.subplots()

    image = axis.imshow(

        game.get_state(),
        cmap="binary"

    )

    plt.title("Conway Simulation")


    def update(frame):

        game.step()

        image.set_array(

            game.get_state()

        )

        return [image]


    anim = animation.FuncAnimation(

        figure,
        update,
        frames=generations,
        interval=80,
        blit=True

    )

    if guardar is not None:

        anim.save(guardar, writer="pillow", fps=12)

        plt.close(figure)

        print("Animacion guardada en:", guardar)

    else:

        plt.show()

    return anim
