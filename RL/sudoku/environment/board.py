import torch
from random import sample
from copy import deepcopy
import ctypes
import numpy as np

c = ctypes.CDLL("./environment/c/sudoku.so")


class Sudoku:
    def __init__(self, device):
        Sudoku.sayHi()
        self.device = device

        self.answer = torch.tensor([])
        self.fixed = torch.ones(9, 9)
        self.board = torch.tensor([])

        self.base = 3
        self.side = self.base*self.base

    def sayHi():
        print("\nWelcome Sudoku\n")

    def reset(self, type=None):
        if type == "weak":
            self.board = self.fixed.clone()
        else:
            self.generateAns()
            self.generateQue()
            # self.printBoard()
        return self.fixed

    def generateAns(self):
        # https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python

        def pattern(r, c): return (self.base*(r %
                                              self.base)+r//self.base+c) % self.side

        def shuffle(s): return sample(s, len(s))

        rBase = range(self.base)
        rows = [g*self.base +
                r for g in shuffle(rBase) for r in shuffle(rBase)]
        cols = [g*self.base +
                c for g in shuffle(rBase) for c in shuffle(rBase)]
        nums = shuffle([i*1.0 for i in range(1, self.base*self.base+1)])

        # produce board using randomized baseline pattern
        self.answer = torch.tensor(
            [[nums[pattern(r, c)] for c in cols] for r in rows],
            dtype=torch.long, device=self.device)
        self.answer = self.answer.unsqueeze(2)
        self.board = self.answer.clone()

    def generateQue(self):
        squares = self.side * self.side
        empties = squares * 3//4
        for p in sample(range(squares), empties):
            self.board[p//self.side][p % self.side] = 0
        self.fixed = self.board.clone()

    def printBoard(self, printing=True):
        result = ",\n".join([",".join([str(n.data.data)[8] for n in line])
                             for line in self.board])
        if printing:
            print("===[Board]===")
            print(result)
            print("=== === ===")
            print()
        return result

    def updateBoard(self, value) -> bool:
        x, y, value = value["x"], value["y"], value["v"]+1

        # if not (0 <= x < 9 and 0 <= y < 9 and 0 < value < 10):
        #     # print("Wrong input. Try Again")
        #     return False
        # print(self.fixed[x][y])
        if self.fixed[x][y]:
            return False

        self.board[x][y] = value
        return True

    # def emptyCell(self):
    #     empty = 0
    #     for i in range(9):
    #         for j in range(9):
    #             if self.board[i][j] == 0:
    #                 empty += 1
    #     return empty

    def calcScore(self):
        return c.calc_score(
            self.board.cpu().numpy().astype(dtype=np.int32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_int))
        )
        # score = 0
        # for i in range(9):
        #     if len(set(v[0].item() for v in self.board[i])) == 9:
        #         score += 1
        #     if len(set(self.board[j][i][0].item() for j in range(9))) == 9:
        #         score += 1

        # for i in range(3):
        #     for j in range(3):
        #         if len(set(self.board[3*i+k][3*j+l][0].item() for k in range(3) for l in range(3))) == 9:
        #             score += 1

        # return score


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
