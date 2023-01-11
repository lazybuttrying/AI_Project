from random import sample
from copy import deepcopy

class Sudoku:
    def __init__(self):
        Sudoku.sayHi()

        self.answer = []
        self.fixed = [[1]*9 for _ in range(9)]
        self.quest = []
        self.empty_count = 0

        self.base  = 3
        self.side  = self.base*self.base
        
    
    def sayHi():
        print("Welcome Sudoku")

    def reset(self):
        self.answer = []
        self.fixed = [[1]*9 for _ in range(9)]
        self.quest = []
        self.empty_count = 0

        self.base  = 3
        self.side  = self.base*self.base

        self.generateAns()
        self.generateQue()
        self.printBoard()

        return self.fixed


    def generateAns(self):
        # https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python


        # pattern for a baseline valid solution
        def pattern(r,c): return (self.base*(r%self.base)+r//self.base+c)%self.side

        # randomize rows, columns and numbers (of valid base pattern)
        from random import sample
        def shuffle(s): return sample(s,len(s)) 
        rBase = range(self.base) 
        rows  = [ g*self.base + r for g in shuffle(rBase) for r in shuffle(rBase) ] 
        cols  = [ g*self.base + c for g in shuffle(rBase) for c in shuffle(rBase) ]
        nums  = shuffle(range(1, self.base*self.base+1))

        # produce board using randomized baseline pattern
        self.answer = [ [nums[pattern(r,c)] for c in cols] for r in rows ]
        self.quest = deepcopy(self.answer)


    def generateQue(self):
        squares = self.side * self.side
        empties = squares * 3//4
        for p in sample(range(squares),empties):
            self.quest[p//self.side][p%self.side] = 0
            self.fixed[p//self.side][p%self.side] = 0
            self.empty_count += 1

 
    def printBoard(self):
        print("===[Board]===")
        numSize = len(str(self.side))
        for line in self.quest:
            print(*(f"{n or '.':{numSize}} " for n in line))
        print("=== === ===")
        print()


    def isDone(self):
        if self.empty_count == 0:
            return True
        else:
            return False

    def changeValue(self, x, y, value) -> bool:

        if not( 0<=x<9 and 0<=y<9 and 0<value<10):
            # print("Wrong input. Try Again")
            return False
        # print(self.fixed[x][y])
        elif self.fixed[x][y]:
            return False
        
        if not self.quest[x][y]:
            self.empty_count -= 1    

        self.quest[x][y] = value
        return True
    

    def calcScore(self, board):
        score = 0
        for i in range(9):
            if len(set(board[i])) == 9:
                score += 1
            if len(set([board[j][i] for j in range(9)])) == 9:
                score += 1
        for i in range(3):
            for j in range(3):
                if len(set([board[3*i+k][3*j+l] for k in range(3) for l in range(3)])) == 9:
                    score += 1
        return score




if __name__ == "__main__":
    sudo = Sudoku()

    while True:
        print("===[Input x, y and value]===")
        print("ex - 2 3 9")
        x, y, value = map(int, input().split())

        if not( 0<=x<9 and 0<=y<9 and 0<value<10):
            print("Wrong input. Try Again")
            continue

        
        result = sudo.changeValue(x,y,value)

        print()
        if result : 
            sudo.printBoard()
        else:
            print("That position cannot be modified. Try Again\n")
        


