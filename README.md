# Juego de la Vida de Conway

## Descripción

Este es mi proyecto del Juego de la Vida de Conway hecho en Python con programación orientada a objetos.

El juego es una cuadrícula donde cada celda está viva o muerta, y va cambiando generación tras generación siguiendo cuatro reglas simples: una celda viva muere si tiene menos de 2 vecinos (soledad) o más de 3 (superpoblación), sobrevive si tiene 2 o 3, y una celda muerta nace si tiene exactamente 3 vecinos vivos.

Además de la simulación, el proyecto incluye la visualización de varios patrones clásicos y un análisis de rendimiento probando distintos tamaños de tablero.

## Archivos

- main.py: es el menú principal, desde aquí se ejecuta todo.
- game_of_life.py: tiene la clase GameOfLife con la lógica del juego.
- patterns.py: crea los patrones clásicos (Glider, Blinker, Toad, Block).
- visualization.py: hace las animaciones con matplotlib.
- benchmark.py: corre las pruebas de rendimiento y genera las gráficas.

## Requisitos

Hay que instalar estas librerías:

    pip install numpy matplotlib numba pillow

Numba es opcional. Si no se puede instalar (por ejemplo con versiones muy nuevas de Python), el programa igual funciona usando numpy y avisa por consola.

## La clase GameOfLife

La clase tiene los métodos que pedía el enunciado:

- __init__(self, rows, cols, initial_state=None): inicializa el tablero. Si no se le da un estado inicial, lo crea aleatorio.
- step(self): calcula la siguiente generación con las reglas de Conway.
- run(self, steps): corre varias generaciones de una vez y devuelve la lista de estados.
- get_state(self): devuelve el tablero actual.

## Sobre los tres backends

Al principio hice el step() con bucles normales recorriendo celda por celda, que es lo más fácil de entender. El problema fue que cuando probé tableros grandes (512 o 1024) se volvía lentísimo y el benchmark tardaba demasiado. Por eso terminé agregando dos formas más de calcular cada generación, y se puede elegir cuál usar con el parámetro backend:

- "python": los bucles normales, mi primera versión. Sirve bien hasta como 128x128.
- "numpy": en vez de bucles, suma el tablero corrido en las 8 direcciones para contar los vecinos de golpe. Es muchísimo más rápido y aguanta los tableros grandes.
- "numba": son los mismos bucles pero compilados y corriendo en paralelo con prange. Esta es la versión que uso para comparar paralelo contra secuencial.

Probé que las tres versiones dan exactamente el mismo resultado.

## Explicación del código

Como dejé el código sin comentarios para que se lea más limpio, acá explico cómo funciona cada parte.

**game_of_life.py**

Arriba importo numpy (para las versiones rápidas) e intento importar numba dentro de un try/except, así si no está instalado no se cae nada y simplemente uso numpy.

La función _step_numba calcula una generación recorriendo el tablero a mano pero compilada con numba y en paralelo. Para cada celda mira las 8 de alrededor, cuenta cuántas están vivas y aplica las reglas. Los bordes los trato como si afuera todo estuviera muerto.

En el __init__ guardo el tamaño y el backend. Si me piden numba y no está, cambio a numpy solo. El tablero lo guardo como lista de listas para el backend python o como array de numpy para los otros dos.

El step() llama a la versión que corresponda. _step_python es la de bucles: crea un tablero nuevo, recorre todo, cuenta vecinos con dos bucles y aplica las reglas. _step_numpy hace lo mismo pero sin bucles, poniéndole un marco de ceros al tablero con np.pad y sumando las 8 direcciones desplazadas, que fue la parte que más me costó entender pero es la que lo hace rápido.

El run() corre varias generaciones y va guardando una copia de cada estado para poder animarlas después.

**patterns.py**

Cada función arma un tablero del tamaño que le pida y le pone el patrón en el centro. Esto lo hice así porque el Glider se mueve, entonces si lo pongo en un tablero de 3x3 choca con el borde y se muere a las pocas generaciones (me pasó al principio). Poniéndolo en un tablero grande tiene espacio para desplazarse.

**visualization.py**

La función animate_game arma la animación con FuncAnimation. Acá había un detalle que me dio problemas: hay que guardar la animación en una variable (anim), porque si no matplotlib la borra y la ventana sale congelada. Si le paso un nombre de archivo guarda un GIF, si no abre la ventana.

**benchmark.py**

La función medir calcula el tiempo promedio por iteración de un tamaño dado. Para numba hago una iteración extra antes de medir, porque la primera vez se tarda compilando y no quería que eso ensuciara el tiempo.

run_benchmark prueba los tres backends de 32x32 hasta 1024x1024 (python puro solo hasta 128 porque después tarda demasiado), arma las curvas teóricas O(n) y O(n^2) para comparar, y genera las tres gráficas. Al final imprime cuánto más rápido es numba que numpy en cada tamaño.

**main.py**

Es el menú. Según la opción que elija, arma el juego con el patrón y lo anima, o corre el benchmark.

## Cómo ejecutar el programa

Desde la carpeta del proyecto:

    python main.py

Sale un menú con estas opciones:

1. Tablero aleatorio (pide cuántas filas y columnas).
2. Glider: la nave que se mueve en diagonal.
3. Blinker: oscilador de 3 celdas.
4. Toad: otro oscilador un poco más grande.
5. Block: un cuadrado que se queda quieto.
6. Benchmark: las pruebas de rendimiento.

## Cómo generar las visualizaciones

Las animaciones salen al elegir cualquiera de las opciones 1 a 5 del menú. Se abre una ventana con la animación. Para guardarla como GIF en vez de solo verla, se le pasa un nombre de archivo a la función:

    animate_game(game, generations=40, guardar="glider.gif")

## Cómo reproducir los experimentos

Para el rendimiento elijo la opción 6 del menú (o corro directamente python benchmark.py). El programa mide el tiempo de cada tamaño con los tres backends, lo imprime en la consola, y guarda tres gráficas: performance.png, performance_loglog.png y memory.png.

## Discusión de resultados

Estos son los tiempos que me dieron (segundos por iteración, varían según la máquina):

| Tablero   | Celdas    | Python    | NumPy     | Numba     |
|-----------|-----------|-----------|-----------|-----------|
| 32x32     | 1.024     | 0.000975  | 0.000052  | 0.000009  |
| 64x64     | 4.096     | 0.004160  | 0.000052  | 0.000010  |
| 128x128   | 16.384    | 0.015040  | 0.000072  | 0.000034  |
| 256x256   | 65.536    | —         | 0.000138  | 0.000112  |
| 512x512   | 262.144   | —         | 0.000421  | 0.000417  |
| 1024x1024 | 1.048.576 | —         | 0.002447  | 0.001672  |

Lo primero que noté es que el algoritmo es O(n) respecto al número de celdas, o sea que el tiempo crece igual de rápido que la cantidad de celdas. Lo confirmé con la versión de python: cuando paso de 64x64 a 128x128 las celdas se multiplican por 4 y el tiempo también por 4 más o menos. Si fuera O(n^2) se multiplicaría por 16. En la gráfica log-log se ve que las tres curvas quedan paralelas a la línea O(n) y por debajo de la O(n^2).

Lo que cambia entre las versiones no es cómo crecen, sino la velocidad de base. NumPy es como 200 a 400 veces más rápido que los bucles de python porque internamente trabaja en C en vez de ir celda por celda en el intérprete.

Sobre comparar paralelo (numba) contra secuencial (numpy): en tableros chicos numba gana bastante (como 6 veces en 32x32) porque no anda creando matrices temporales. Pero conforme el tablero crece la ventaja se va achicando hasta que en 512x512 quedan casi iguales. Esto pasa porque en tableros grandes el cuello de botella ya no es calcular sino mover los datos por la memoria, y ahí da igual cuántos núcleos uses.

Sobre la memoria, el tablero ocupa 1 byte por celda (uint8), así que crece de forma lineal: 1 MB para el de 1024x1024. La gráfica memory.png lo muestra como una recta. En cada step se crea un tablero nuevo del mismo tamaño, así que por un momento hay dos en memoria, pero igual sigue siendo lineal.

Los cuellos de botella que vi fueron: en python puro el intérprete es lo lento, por eso solo lo medí hasta 128. En numpy lo que pesa en tableros grandes es la memoria de las matrices temporales. Y en numba el único costo raro es la compilación de la primera vez, que por eso la descarté al medir.

## Conclusiones

Con este proyecto entendí mejor cómo de reglas tan simples salen comportamientos complejos como los osciladores, las naves que se mueven y las figuras que se quedan quietas.

También me sirvió para ver en la práctica la diferencia entre que un algoritmo sea O(n) y que sea rápido: las tres versiones son O(n), pero vectorizar con numpy o compilar con numba lo hacen muchísimo más veloz que los bucles normales. Empezar con la versión de bucles me ayudó a entender bien el algoritmo, y tuve que agregar numpy y numba para poder llegar a los tableros de 1024x1024 sin esperar una eternidad.
