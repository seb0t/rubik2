import numpy as np
import random

class RubikEnv:
    def __init__(self):
        self.reset()

        self.moves_dict = {
            "F": (1,0),
            "f": (1,1),
            "D": (2,0),
            "d": (2,1),
            "B": (3,0),
            "b": (3,1),
            "U": (4,0),
            "u": (4,1),
            "L": (5,0),
            "l": (5,1),
            "R": (6,0),
            "r": (6,1),
            "M": (5,2),
            "m": (5,3),
            "E": (2,2),
            "e": (2,3),
            "S": (1,2),
            "s": (1,3)
        }

        self.actionMap = {
            0: "F",
            1: "f",
            2: "D",
            3: "d",
            4: "B",
            5: "b",
            6: "U",
            7: "u",
            8: "L",
            9: "l",
            10: "R",
            11: "r",
            12: "M",
            13: "m",
            14: "E",
            15: "e",
            16: "S",
            17: "s"
        }

    def reset(self):
        self.state = np.array([
            [0 , 0 , 5 , 5 , 5 , 0 , 0],
            [0 , 3 , 1 , 1 , 1 , 4 , 0],
            [0 , 3 , 1 , 1 , 1 , 4 , 0],
            [0 , 3 , 1 , 1 , 1 , 4 , 0],
            [0 , 0 , 2 , 2 , 2 , 0 , 0],
            [3 , 3 , 2 , 2 , 2 , 4 , 4],
            [0 , 0 , 2 , 2 , 2 , 0 , 0],
            [0 , 3 , 6 , 6 , 6 , 4 , 0],
            [0 , 3 , 6 , 6 , 6 , 4 , 0],
            [0 , 3 , 6 , 6 , 6 , 4 , 0],
            [0 , 0 , 5 , 5 , 5 , 0 , 0],
            [0 , 3 , 5 , 5 , 5 , 4 , 0]
        ])

    def execMove(self, move: str):
        side, direction = self.moves_dict[move]
        self.move(side, direction)
    
    def move(self ,side: int, direction: int):
        self.switchPovToSide(side)
        if direction in [0,1]:
            self.rotateMainMatrix(direction)
            self.resetPov(side)
        elif direction in [2,3]:
            self.rotateMiddleLayer(direction - 2)
            if side == 5 :
                if direction == 2:
                    self.resetPov(5)
                    self.resetPov(4)
                else:
                    self.resetPov(5)
                    self.resetPov(2)

            if side == 2 :
                if direction == 2:
                    self.resetPov(2)
                    self.resetPov(5)
                    self.resetPov(2)
                else:
                    self.resetPov(4)
                    self.resetPov(5)
                    self.resetPov(4)

            if side == 1 :
                if direction == 2 :
                    self.resetPov(2)
                    self.resetPov(5)
                else:
                    self.resetPov(2)
                    self.resetPov(6)                  
            

        

    def rotateMainMatrix(self, direction: int):
        mainMatrix = np.array([self.state[0,1:6],
                                self.state[1,1:6],
                                self.state[2,1:6],
                                self.state[3,1:6],
                                self.state[4,1:6]])
        if direction == 0:  # clockwise
            rotatedMatrix = np.rot90(mainMatrix, -1)
        else:  # counter-clockwise
            rotatedMatrix = np.rot90(mainMatrix, 1)

        self.state[0,1:6] = rotatedMatrix[0]
        self.state[1,1:6] = rotatedMatrix[1]
        self.state[2,1:6] = rotatedMatrix[2]
        self.state[3,1:6] = rotatedMatrix[3]
        self.state[4,1:6] = rotatedMatrix[4]

    def rotateMiddleLayer(self, direction: int):

        middleLayer = np.concatenate((self.state[5],self.state[11,1:6][::-1]))
        if direction == 0:  # clockwise
            middleLayer = np.roll(middleLayer, -3)
        else:  # counter-clockwise
            middleLayer = np.roll(middleLayer, 3)

        self.state[5] = middleLayer[:7]
        self.state[11,1:6] = middleLayer[7:][::-1]

    def switchCornersPov(self):
        self.state[0,1] , self.state[1,1] = self.state[1,1] , self.state[0,1]
        self.state[0,5] , self.state[1,5] = self.state[1,5] , self.state[0,5]

        self.state[3,1] , self.state[4,1] = self.state[4,1] , self.state[3,1]
        self.state[3,5] , self.state[4,5] = self.state[4,5] , self.state[3,5]

        self.state[6,1] , self.state[7,1] = self.state[7,1] , self.state[6,1]
        self.state[6,5] , self.state[7,5] = self.state[7,5] , self.state[6,5]

        self.state[9,1] , self.state[10,1] = self.state[10,1] , self.state[9,1]
        self.state[9,5] , self.state[10,5] = self.state[10,5] , self.state[9,5]


    def magicMove(self,side: int):
        leftSide = np.array([self.state[11,1] , self.state[5,0] , self.state[5,1]])
        rightSide = np.array([self.state[5,5] , self.state[5,6] , self.state[11,5]])
        bottomSide = np.array([self.state[5,2] , self.state[5,3] , self.state[5,4]])
        topSide = np.array([self.state[11,4] , self.state[11,3] , self.state[11,2]])


        if side == 5 :  # left
            self.state[11,1] , self.state[5,0] , self.state[5,1] = bottomSide[0] , bottomSide[1] , bottomSide[2]
            self.state[5,5] , self.state[5,6] , self.state[11,5] = topSide[0] , topSide[1] , topSide[2]
            self.state[5,2] , self.state[5,3] , self.state[5,4] = rightSide[0] , rightSide[1] , rightSide[2]
            self.state[11,4] , self.state[11,3] , self.state[11,2] = leftSide[0] , leftSide[1] , leftSide[2]

        elif side == 6 :  # right
            self.state[11,1] , self.state[5,0] , self.state[5,1] = topSide[0] , topSide[1] , topSide[2]
            self.state[5,5] , self.state[5,6] , self.state[11,5] = bottomSide[0] , bottomSide[1] , bottomSide[2]
            self.state[5,2] , self.state[5,3] , self.state[5,4] =  leftSide[0] , leftSide[1] , leftSide[2]
            self.state[11,4] , self.state[11,3] , self.state[11,2] = rightSide[0] , rightSide[1] , rightSide[2]
        
    def switchPovToSide(self, side: int):
        # Placeholder for the actual rotation logic

        if side == 1:  # Front
            return
        elif side == 2:  # Bottom
            self.switchCornersPov()
            self.cutAndStack(3)
        elif side == 3:  # Back
            self.cutAndStack(6)
        elif side == 4:  # Top
            self.switchCornersPov()
            self.cutAndStack(9)
        elif side == 5:  # Left
            self.rotateMainMatrix(0)
            self.cutAndStack(6)
            self.rotateMainMatrix(1)
            self.cutAndStack(6)
            self.magicMove(side)
            self.cutAndStack(9)
            self.switchCornersPov()
        elif side == 6:  # Right
            self.rotateMainMatrix(1)
            self.cutAndStack(6)
            self.rotateMainMatrix(0)
            self.cutAndStack(6)
            self.magicMove(side)
            self.cutAndStack(9)
            self.switchCornersPov()
    
    def cutAndStack(self, rowsBeforeCut: int):
        mainMatrix = self.state[:,1:6].copy()
        topStack = mainMatrix[:rowsBeforeCut , :]
        bottomStack = mainMatrix[rowsBeforeCut: , :]
        self.state[:,1:6] = np.vstack((bottomStack , topStack))

    def resetPov(self , side : int ) :
        if 2 <= side <= 4:
            self.switchPovToSide(6-side) # switch back to original pov , hard to explain why 6-side works but it does
        if side == 5:
            self.switchCornersPov()
            self.cutAndStack(self.state.shape[0] - 9)
            self.magicMove(6)
            self.cutAndStack(self.state.shape[0] - 6)
            self.rotateMainMatrix(0)
            self.cutAndStack(self.state.shape[0] - 6)
            self.rotateMainMatrix(1)
        if side == 6:
            self.switchCornersPov()
            self.cutAndStack(self.state.shape[0] - 9)
            self.magicMove(5)
            self.cutAndStack(self.state.shape[0] - 6)
            self.rotateMainMatrix(1)
            self.cutAndStack(self.state.shape[0] - 6)
            self.rotateMainMatrix(0)

    def convert_matrix_to_cross(self):
        """
        Convert a Rubik's cube state matrix into a cross layout for visualization.
        """

        newMatrix = np.zeros((12, 9), dtype=int)
        
        coordinates = {
            "front" : self.state[1:4, 2:5],
            "bottom": self.state[4:7, 2:5],
            "back"  : self.state[7:10, 2:5],
            "top" : np.vstack((self.state[10:, 2:5], self.state[0, 2:5])), # type: ignore
            "left"  : np.array([
                [self.state[9, 1], self.state[11, 1], self.state[1, 1]],
                [self.state[8, 1], self.state[5, 0], self.state[2, 1]],
                [self.state[7, 1], self.state[5, 1], self.state[3, 1]]
            ]), # type: ignore
            "right" : np.array([
                [self.state[1, 5], self.state[11, 5], self.state[9, 5]],
                [self.state[2, 5], self.state[5, 6], self.state[8, 5]],
                [self.state[3, 5], self.state[5, 5], self.state[7, 5]]
            ])
        }    

        newMatrix[0:3,3:6] = coordinates["top"]
        newMatrix[3:6,0:3] = coordinates["left"]
        newMatrix[3:6,3:6] = coordinates["front"]
        newMatrix[3:6,6:9] = coordinates["right"]
        newMatrix[6:9,3:6] = coordinates["bottom"]
        newMatrix[9:12,3:6]= coordinates["back"]

        return newMatrix
    
    def shuffle(self, num_moves: int = 20) -> list[str]:
        moves = list(self.moves_dict.items())
        moveList : list[str] = []
        last_move : tuple[str, tuple[int,int]] = ("", (-1,-1))
        for _ in range(num_moves):
            newMove = random.choice(moves)
            while newMove[1][0] == last_move[1][0] \
                  and newMove[1][1] != last_move[1][1]:
                newMove = random.choice(moves)
            self.execMove(newMove[0])
            last_move = newMove
            moveList.append(newMove[0])
        return moveList
        

    def is_solved(self) -> bool:
        newMatrix = self.convert_matrix_to_cross()
        if np.all(newMatrix[0:3,3:6] == newMatrix[1,4]) and \
              np.all(newMatrix[3:6,0:3] == newMatrix[4,1]) and \
                np.all(newMatrix[3:6,3:6] == newMatrix[4,4]) and \
                  np.all(newMatrix[3:6,6:9] == newMatrix[4,7]) and \
                    np.all(newMatrix[6:9,3:6] == newMatrix[7,4]) and \
                      np.all(newMatrix[9:12,3:6] == newMatrix[10,4]):   
                
            return True 
        return False
    
    def step(self, action: int):

        original_state = self.state.copy()
        reward = -1
        done = False

        move = self.actionMap[action]
        self.execMove(move)
        
        if self.is_solved():
            reward = 100
            done = True

        return original_state, self.state, reward, done

        
