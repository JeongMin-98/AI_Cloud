import queue

class State:
    def __init__(self, board, goal, moves=0):
        self.board = board
        self.moves = moves
        self.goal = goal

    def get_new_board(self, i1, i2, moves):
        new_board = self.board[:]
        new_board[i1], new_board[i2] = new_board[i2], new_board[i1]
        return State(new_board, self.goal, moves)

    def expand(self, moves):
        result = []

        i = self.board.index(0)

        if i not in [0, 1, 2]:  # up
            result.append(self.get_new_board(i, i - 3, moves))
        if i not in [0, 3, 6]:  # left
            result.append(self.get_new_board(i, i - 1, moves))
        if i not in [2, 5, 8]:  # right
            result.append(self.get_new_board(i, i + 1, moves))
        if i not in [6, 7, 8]:  # down
            result.append(self.get_new_board(i, i + 3, moves))
        return result

    def f(self):
        return self.h()+self.g()

    def h(self):
        return sum([1 if self.board[i] != self.goal[i] else 0 for i in range(8)])

    def g(self):
        return self.moves

    def __lt__(self, other):
        return self.f() < other.f()

    def __str__(self):
        return "---------------- f(N)= " + str(self.f()) + "\n" +\
               "---------------- h(N)= " + str(self.h()) + "\n" +\
               "---------------- g(N)= " + str(self.g()) + "\n" +\
               str(self.board[:3]) + "\n" \
               + str(self.board[3:6]) + "\n" \
               + str(self.board[6:]) + "\n" \
               + "---------------"

puzzle = [1,2,3,0,4,6,7,5,8]
goal = [1,2,3,4,5,6,7,8,0]

open_queue = queue.PriorityQueue()
open_queue.put(State(puzzle, goal))

closed_queue = []
moves = 0
while not open_queue.empty():
    # print("START of OPENQ")
    # for elem in open_queue:
    #     print(elem)
    # print("END of OPNEQ")

    current = open_queue.get()
    print(current)
    if current.board == goal:
        print("탐색성공")
        break

    moves = current.moves+1
    for state in current.expand(moves):
        if state not in closed_queue:
            open_queue.put(state)
    closed_queue.append(current)
else:
    print("탐색실패")