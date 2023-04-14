# 导入随机包
import random
from game import Game
from math import log
from board import Board
from copy import deepcopy


#定义无穷
_infinite = 0x7ffffff

#定义超参数
parameterC = 1.2

class TreeNode:
    """
    蒙特卡洛树的树节点类
    记录它的行为，它的后续行为，遍历次数以及权重
    """
    def __init__(self,color,action:str="",weight = _infinite) -> None:
        """初始化

        Args:
            color (str): 记录棋手
            action (str, optional): 下棋位置. Defaults to "".
            weight (_type_, optional): 权重. Defaults to _infinite.
        """
        self.color = color
        self.action = action
        self.children = []
        self.checkTimes = 0
        self.weight = weight
        
    def _selection(self):
        """节点权重的计算选择

        Returns:
            TreeNode: 返回下一步选择的下棋位置
        """
        if len(self.children)==0: return None
        
        indexForChild = 0
        maxValue = -0xffffffff
        UCBValue = 0
        for index,child in enumerate(self.children):
            if  not child.checkTimes:
                child.weight = 0
                return child
            UCBValue = child.weight/child.checkTimes + (2*log(self.checkTimes)/child.checkTimes)**0.5
            if UCBValue > maxValue : 
                maxValue = UCBValue
                indexForChild = index
        
        return self.children[indexForChild]  

class Tree:
    """
    蒙特卡洛树
    """
    def __init__(self,board:Board,color,C:int=1.2,training:int = 400) -> None:
        """初始化树

        Args:
            board (Board): 棋盘
            color (str): 棋手
            C (int, optional): 超参数. Defaults to 1.2.
            training (int, optional): 单步训练次数. Defaults to 400.
        """
        global parameterC
        parameterC = C
        self.board = board
        self.root = TreeNode(color=color,weight=0)
        actions = list(self.board.get_legal_actions(self.root.color))
        for action in actions:
            self.root.children.append(TreeNode(color=self.root.color,action=action))
        selected = self.root._selection()
        while(self.root.checkTimes<training and selected):
            board = deepcopy(self.board._board)
            weight = self.simulation(selected)
            self.root.weight += weight
            self.root.checkTimes+=1
            self.board._board = board
    
    def selection(self) -> str:
        """训练结束返回接口

        Returns:
            str: 下棋位置
        """
        return self.root._selection().action
    
    def simulation(self,node:TreeNode):
        """模拟接口

        Args:
            node (TreeNode): 对某个节点进行模拟

        Returns:
            int: 模拟后的权重
        """
        self.board._move(node.action,node.color)
        if len(node.children)==0:
            self.expansion(node)
        if len(node.children)==0 :
            node.color = 'X' if node.color =='O' else 'O'
            self.expansion(node)
            if len(node.children)==0:
                node.weight = [2,0,1][self.board.get_winner()[0]]
                node.checkTimes+=1
                if self.root.color == 'O':
                    node.weight = 2 - node.weight
                    return node.weight
                else :
                    return node.weight
            else:
                selected = node._selection()
                weight = self.simulation(selected)
                selected.weight+=weight
                node.checkTimes+=1
                return weight
        else:
            selected = node._selection()
            weight = self.simulation(selected)
            node.weight+=weight
            node.checkTimes+=1
            return weight
    
    def expansion(self,node:TreeNode):
        actions = list(self.board.get_legal_actions('X' if node.color=='O' else 'O'))
        for action in actions:
            node.children.append(TreeNode('X' if node.color=='O' else 'O',action=action))
           
      
        

class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    def get_move(self, board:Board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------
        
        t = Tree(board=deepcopy(board),color=self.color)
        action = t.selection()
        # ------------------------------------------------------------------------

        return action
