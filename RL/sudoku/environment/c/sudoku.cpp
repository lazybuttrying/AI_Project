#include <cstdio>


extern "C" {
    int calc_score(int sudoku[9][9][1]) {
        int score = 0;
        
        int i=0, j=0;
        int rtotal[9] = {0,};
        int ctotal[9] = {0,};
        int ztotal[9] = {0,};
        int rsum=0, csum=0, zsum=0;

        for (i = 0; i < 9; i++) {
            for (j = 0; j < 9; j++) {
                if (sudoku[i][j][0])
                    rtotal[j] = 1;
                if (sudoku[j][i][0])
                    ctotal[j] = 1;                
            }
            
            rsum=0;
            csum=0;

            for (j = 0; j < 9; j++) {
                rsum += rtotal[j];
                csum += ctotal[j];
                rtotal[j] = 0;
                ctotal[j] = 0;
            }

            if (rsum == 9)
                score += 1;
            if (csum == 9)
                score += 1;
        }

        for (i=0; i<3; i++) {
            for (j=0; j<3; j++) {

                for (int k=0; k<3; k++) {
                    for (int l=0; l<3; l++) {
                        if (sudoku[i*3+k][j*3+l][0])
                            ztotal[k*3+l] = 1;
                    }
                }

                zsum = 0;
                for (int k=0; k<9; k++) {
                    zsum += ztotal[k];
                    ztotal[k]=0;
                }
                if (zsum == 9)
                    score += 1;
                    
            }
        }

        return score;
    }
}