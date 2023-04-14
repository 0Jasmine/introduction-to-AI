from kanren import run, eq, membero, var, conde      
from kanren.core import lall                        
import time
                                                                           #
countries=["英国人","西班牙人","日本人","意大利人","挪威人"]
jobs=["油漆工","摄影师","外交官","小提琴家","医生"]
colors=["红色","白色","蓝色","黄色","绿色"]
pets=["狗","蜗牛","斑马","马","狐狸"]
drinks=["矿泉水","牛奶","茶","橘子汁","咖啡"]

def left(a,b,arr:list):
    return membero((a,b),zip(arr,arr[1:]))
def right(a,b,arr:list):
    return membero((b,a),zip(arr,arr[1:]))
def next(a,b,arr:list):
    return  conde([left(a,b,arr)],[right(a,b,arr)])
        
class Agent:
    """
    推理智能体.
    """
    
    def __init__(self):
        """
        智能体初始化.
        """
        
        self.units = var()             
        self.rules_zebraproblem = None  
        self.solutions = None           # 存储结果
        
    def define_rules(self):
        """
        定义逻辑规则.
        """
        self.rules_zebraproblem = lall(
        (eq, (var(), var(), var(), var(), var()), self.units),         # self.units共包含五个unit成员，即每一个unit对应的var都指代一座房子(国家，工作，饮料，宠物，颜色) 
        (membero,(countries[0], var(), var(), var(), colors[0]), self.units),#1 
        (membero,(countries[1], var(), var(), pets[0], var()), self.units),#2
        (membero,(countries[2], jobs[0], var(), var(), var()), self.units),#3
        (membero,(countries[3], var(), drinks[2], var(), var()), self.units),#4
        (eq,((countries[4], var(), var(), var(), var()),var(), var() , var(), var()), self.units),#5
        (right,(var(), var(), var(), var(), colors[4]),(var(), var(), var(), var(), colors[1]), self.units), #6
        (membero,(var(), jobs[1], var(), pets[1],var()), self.units), #7
        (membero,(var(), jobs[2], var(), var(), colors[3]), self.units), #8
        (eq,(var(), var(), (var(), var(), drinks[1], var(), var()), var(), var()), self.units),#9
        (membero,(var(), var(),drinks[4], var(), colors[4]), self.units),#10
        (next,(countries[4], var(), var(), var(), var()),(var(), var(), var(), var(), colors[2]),self.units),#11
        (membero,(var(), jobs[3], drinks[3], var(), var()),self.units),
         #12
        (next,(var(), var(), var(), pets[4], var()), (var(), jobs[4], var(), var(), var()),self.units),#13
        (next,(var(), jobs[2], var(), var(), var()), (var(), var(), var(), pets[3], var()), self.units) , #14
        # make sure all the drinks and pets exit
        (membero,(var(), var(),drinks[0], var(), var()), self.units),
        (membero,(var(), var(),var(), pets[2], var()), self.units)
)

    
    def solve(self):
        """
        规则求解器(请勿修改此函数).
        return: 斑马规则求解器给出的答案，共包含五条匹配信息，解唯一.
        """
        
        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions

