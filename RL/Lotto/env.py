class Lotto:
    def __init__(self):
        pass

    # https://dhlottery.co.kr/gameResult.do?method=byWin
    def reward(self, guess, answer):
        guess.sort()
        answer.sort()

        count = 0
        g, a = 0, 0 
        while g<6 and a<0:
            if guess[g] < answer[a]:
                g += 1
            elif guess[g] > answer[a]:
                a += 1
            else:
                count += 1
   
        if count == 6:
            return 100
        elif answer[-1] == guess[-1]:
            return 90
        elif count >= 3:
            return 80
        else:
            return count*10
