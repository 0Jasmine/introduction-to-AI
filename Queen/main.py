import numpy as np           # 提供维度数组与矩阵运算
import copy                  # 从copy模块导入深度拷贝方法
from board import Chessboard

# 基于棋盘类，设计搜索策略
class Game:
    def __init__(self, show = True):
        """
        初始化游戏状态.
        """
        
        self.chessBoard = Chessboard(show)
        self.solves = []
        self.gameInit()
        
    # 重置游戏
    def gameInit(self, show = True):
        """
        重置棋盘.
        """
        
        self.Queen_setRow = [-1] * 8
        self.chessBoard.boardInit(False)
        
    ##############################################################################
    ####                请在以下区域中作答(可自由添加自定义函数)                 #### 
    ####              输出：self.solves = 八皇后所有序列解的list                ####
    ####             如:[[0,6,4,7,1,3,5,2],]代表八皇后的一个解为                ####
    ####           (0,0),(1,6),(2,4),(3,7),(4,1),(5,3),(6,5),(7,2)            ####
    ##############################################################################
    #                                                                            #
    def _setQueen(self, x, y, show = False):
        '''
        落子之后更新棋盘
        '''
        
        if self.chessBoard.setLegal(x, y):
            self.chessBoard.queenMatrix[x][y] =1                                       #落子位置
            for i in range(8):
                self.chessBoard.unableMatrix[x][i] =-1                                 #更新行
                self.chessBoard.unableMatrix[i][y] =-1                                 #更新列
            for i in range(-7,8):
                if self.chessBoard.isOnChessboard(x+i,y+i):
                    self.chessBoard.unableMatrix[x+i][y+i] =-1                         #更新正对角线
                if self.chessBoard.isOnChessboard(x+i,y-i):
                    self.chessBoard.unableMatrix[x+i][y-i] =-1                         #更新反对角线
            self.chessBoard.chessboardMatrix = self.chessBoard.unableMatrix +2*self.chessBoard.queenMatrix   #更新棋盘
            self.chessBoard.printMatrix[1:9,1:9] = self.chessBoard.chessboardMatrix
            if show:
                self.chessBoard.printChessboard(False)
            return True
        else:
            return False    
    
    def resolve(self,result:list):
        """这个函数就是实现的核心，通过递归、回溯获得所有解

        Args:
            result (list): 存放尝试的解
        """
        if(len(result)==8): 
            self.solves.append(copy.deepcopy(result))
            return 
        temp= copy.deepcopy(self.chessBoard.unableMatrix)
        for i in range(8):
            if i not in result:
                ret=self._setQueen(len(result),i)
                if ret:  # 如果该落子方法可行则递归继续尝试
                    result.append(i)
                    self.resolve(result)
                    result.pop()    # 还原
                    self.chessBoard.unableMatrix=copy.deepcopy(temp)
                    self.chessBoard.queenMatrix[len(result)][i] = 0    
                
        
    def run(self, row=0):
        self.resolve([])

    #                                                                            #
    ##############################################################################
    #################             完成后请记得提交作业             ################# 
    ##############################################################################
    
    def showResults(self, result):
        """
        结果展示.
        """
        
        self.chessBoard.boardInit(False)
        for i,item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i,item,False)
        
        self.chessBoard.printChessboard(False)
    
    def get_results(self):
        """
        输出结果(请勿修改此函数).
        return: 八皇后的序列解的list.
        """
        
        self.run()
        return self.solves

