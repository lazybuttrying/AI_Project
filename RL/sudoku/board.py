import torch
from random import sample
from copy import deepcopy


class Sudoku:
    def __init__(self):
        Sudoku.sayHi()

        self.answer = torch.tensor([])
        self.fixed = torch.ones(9, 9)
        self.board = torch.tensor([])

        self.base = 3
        self.side = self.base*self.base

    def sayHi():
        print("Welcome Sudoku")

    def reset(self):
        self.answer = torch.tensor([])
        self.fixed = torch.ones(9, 9)
        self.board = torch.tensor([])

        self.base = 3
        self.side = self.base*self.base

        self.generateAns()
        self.generateQue()
        # self.printBoard()

        return self.fixed

    def generateAns(self):
        # https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python

        # pattern for a baseline valid solution
        def pattern(r, c): return (self.base*(r %
                                              self.base)+r//self.base+c) % self.side

        # randomize rows, columns and numbers (of valid base pattern)

        def shuffle(s): return sample(s, len(s))
        rBase = range(self.base)
        rows = [g*self.base +
                r for g in shuffle(rBase) for r in shuffle(rBase)]
        cols = [g*self.base +
                c for g in shuffle(rBase) for c in shuffle(rBase)]
        # nums = shuffle(range(1, self.base*self.base+1, 1.0))
        nums = shuffle([i*1.0 for i in range(1, self.base*self.base+1)])

        # produce board using randomized baseline pattern
        self.answer = torch.tensor(
            [[nums[pattern(r, c)] for c in cols] for r in rows],
            dtype=torch.long)
        self.answer = self.answer.unsqueeze(2)
        self.board = self.answer.clone()

    def generateQue(self):
        squares = self.side * self.side
        empties = squares * 3//4
        for p in sample(range(squares), empties):
            self.board[p//self.side][p % self.side] = 0
        self.fixed = self.board.clone()

    def printBoard(self):
        print("===[Board]===")
        numSize = len(str(self.side))
        for line in self.board:
            # print(*(f"{n or '.':{numSize}} " for n in line))
            print("..".join([str(n.data.data)[8] for n in line]))
        print("=== === ===")
        print()

    def updateBoard(self, value) -> bool:
        x, y, value = value["x"], value["y"], value["v"]+1

        if not (0 <= x < 9 and 0 <= y < 9 and 0 < value < 10):
            # print("Wrong input. Try Again")
            return False
        # print(self.fixed[x][y])
        elif self.fixed[x][y]:
            return False

        self.board[x][y] = value
        return True

    def calcScore(self):
        score = 0
        for i in range(9):
            if len(set(v[0].item() for v in self.board[i])) == 9:
                score += 1
            if len(set(self.board[j][i][0].item() for j in range(9))) == 9:
                score += 1

        for i in range(3):
            for j in range(3):
                if len(set(self.board[3*i+k][3*j+l][0].item() for k in range(3) for l in range(3))) == 9:
                    score += 1

        return score/27 if score else -1


if __name__ == "__main__":
    sudo = Sudoku()
    sudo.reset()
    while True:
        print("===[Input x, y and value]===")
        print("ex - 2 3 9")
        x, y, value = map(int, input().split())

        if not (0 <= x < 9 and 0 <= y < 9 and 0 < value < 10):
            print("Wrong input. Try Again")
            continue

        result = sudo.updateBoard(x, y, value)

        print()
        if result:
            sudo.printBoard()
        else:
            print("That position cannot be modified. Try Again\n")
            continue

        print("Score: ", sudo.calcScore(sudo.board))
