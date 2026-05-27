from game_of_life import GameOfLife

from patterns import create_glider
from patterns import create_blinker
from patterns import create_toad
from patterns import create_block

from visualization import animate_game

from benchmark import run_benchmark


print()
print("=================================")
print("       GAME OF LIFE")
print("=================================")

print()
print("1 - Random Board")
print("2 - Glider")
print("3 - Blinker")
print("4 - Toad")
print("5 - Block")
print("6 - Benchmark")
print()

option = input("Select an option: ")


if option == "1":

    rows = int(input("Rows: "))
    cols = int(input("Columns: "))

    backend = "numpy" if rows * cols > 4096 else "python"

    game = GameOfLife(rows, cols, backend=backend)

    animate_game(game)


elif option == "2":

    pattern = create_glider(32)

    game = GameOfLife(32, 32, pattern)

    animate_game(game)


elif option == "3":

    pattern = create_blinker(16)

    game = GameOfLife(16, 16, pattern)

    animate_game(game)


elif option == "4":

    pattern = create_toad(16)

    game = GameOfLife(16, 16, pattern)

    animate_game(game)


elif option == "5":

    pattern = create_block(8)

    game = GameOfLife(8, 8, pattern)

    animate_game(game)


elif option == "6":

    run_benchmark()


else:

    print()
    print("Invalid option")
