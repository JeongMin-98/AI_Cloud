# 맹목적 탐색방법

board = [0] * 16

# 1행 board[0:4]
# 2행 board[4:8]
# 3행 board[8:12]
# 4행 board[12:16]

# N*N 보드에 불가능한 구역 만들기
# 1. 같은 행 지우기
# 2. 오른쪽 대각선 N+1씩 지우기
# 3. 같은 열 지우기 N 씩 지우기
# 4. 왼쪽 대각선 N-1씩 지우기

# 안되는 조건
# 1. 같은 행에 말을 놓지 못하면 안됨.


# 끝나는 조건
# 1. 모든 행에 말이 놓아졌을 경우.

class State:

    def __init__(self, board, moves=0):
        self.board = board
        self.moves = moves

    def __str__(self):
        return str(self.board[:4])+"\n"+ \
               str(self.board[4:8]) + "\n"+ \
               str(self.board[8:12]) + "\n"+ \
               str(self.board[12:16]) + "\n"+\
               "------------------------"

    def put(self, index, moves):
        new_board = self.board[:]
        if moves == 0:
            new_board[index] = '*'
        else:
            new_board[index+(self.moves)*4] = '*'
        return State(new_board, self.moves+1)

    def check_put(self, moves):
        move = moves
        result = []
        if move == 0:
            for i in range(0,4):
                result.append(self.put(i, moves))
            return result
        else:
            i = self.board[4*(move-1):4*move].index('*')
            no = i + ((moves-1)*4)
            temp = no
            if i % 4 == 0:
                for toward in [4, 5]:
                    while no < 15:
                        no = no + toward
                        if no > 15:
                            break
                        self.board[no] = 1
                        if toward == 4:
                            if no in [12,13,14,15]:
                                break
                        if toward == 5:
                            if no in [3,7,11,15]:
                                break
                    no = temp
            elif i % 4 == 1:
                for toward in [3, 4, 5]:
                    while no < 15:
                        no = no + toward
                        if no > 15:
                            break
                        self.board[no]=1
                        if toward == 3:
                            if no in [0,4,8,12]:
                                break
                        if toward == 4:
                            if no in [12,13,14,15]:
                                break
                        if toward == 5:
                            if no in [3,7,11,15]:
                                break

                    no = temp
            elif i % 4 == 2:
                for toward in [3, 4, 5]:
                    while no < 15:
                        no = no + toward
                        if no >= 16:
                            break
                        self.board[no] = 1
                        if toward == 3:
                            if no in [0,4,8,12]:
                                break
                        if toward == 4:
                            if no in [12,13,14,15]:
                                break
                        if toward == 5:
                            if no in [3,7,11,15]:
                                break
                    no = temp
            else:
                for toward in [3, 4]:
                    while no < 15:
                        no = no + toward
                        if no > 16:
                            break
                        self.board[no] = 1
                        if toward == 3:
                            if no in [0,4,8,12]:
                                break
                        if toward == 4:
                            if no in [12,13,14,15]:
                                break
                    no = temp
        if self.moves == 4:
            print("종료했습니다.\n")
            return result
        temp_board = self.board[4*move:4*(move+1)]
        for i in range(0, 4):
            if temp_board[i] == 0:
                result.append(self.put(i, self.moves+1))
        print("확인했습니다. {}회".format(self.moves))
        return result


board = [0] * 16
open_queue = []
open_queue.append(State(board, 0))
closed_queue = []

moves = 1

while len(open_queue) != 0 or moves <= 4:
    
    current = open_queue.pop()
    print(current)
    closed_queue.append(current)
    moves = current.moves

    for state in current.check_put(moves):
        if state in closed_queue:
            continue
        else:
            open_queue.append(state)







