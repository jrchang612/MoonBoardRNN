import numpy as np
import os
import copy
import pickle
import heapq

global hyperparameter
hyperparameter = [1, 1]


"""For Sequence Generation"""

class beta:
    """ climbing beta is the climber's feeling of the best hand sequence
    class beta describe how the climber create the beta. It stored information including
    moonboard holds information.
    the ongoing built hand sequence (LRLRLRRR) and the hold climber used
    """
    def __init__(self, avalible_holds):
        """Creates a new beta with the specified attributes."""
        self.totalNumOfHold = np.size(avalible_holds, axis=0)
        self.allHolds = avalible_holds
        self.handSequence = []        # Use the number 0,1,2,3,4,5... sequence
        self.handOperator = [] # op: Right hand / Left Hand
        self.holdsNotUsed = []           # You can then use this list to decide where to go next
        self.holdsNotUsed.extend(range(self.totalNumOfHold))
        self.isFinished = False          # True when go to finish hold (neglect matching)
        self.overallSuccess = 1      # Can evaluate difficulty of this beta
        self.singleMoveSuccessRate = []  # singleMoveDifficulty
        self.tryout = 0                  # Try one additional point. Record successRate
        self.touchEndHold = 0
    
    def addStartHolds(self, zeroOrOne):
        opList = ["LH", "RH"]
        startHoldList = self.getStartHold()
        if len(startHoldList) == 1:
            self.handSequence.append(int(self.getOrderFromHold(startHoldList[0])))   # Add a new hold into beta!
            self.handSequence.append(int(self.getOrderFromHold(startHoldList[0])))
            self.handOperator.extend(opList) 
            self.holdsNotUsed.remove(self.getOrderFromHold(startHoldList[0]))   # Not consider match
        if len(startHoldList) == 2:  
            self.handSequence.append(int(self.getOrderFromHold(startHoldList[0])))   # Add a new hold into beta!
            self.handSequence.append(int(self.getOrderFromHold(startHoldList[1])))
            self.handOperator.append(opList[zeroOrOne]) 
            self.handOperator.append(opList[1-zeroOrOne]) # indicate which hand
            self.holdsNotUsed.remove(self.getOrderFromHold(startHoldList[0]))   # Not consider match
            self.holdsNotUsed.remove(self.getOrderFromHold(startHoldList[1]))
            
    def getAllHolds(self):
        """N holds rows, 10 columns np array"""
        return self.allHolds
    
    def tryMove(self, tryOrder, op):
        """Try how addition hold. Return successRate, tryOrder is a number in holdsNotUsed 
           list. op is either "LH" or "RH" """
        tryHold = self.allHolds[tryOrder]
        finalXY = self.getXYFromOrder(tryOrder)    
        if op == "LH":
            originalXY = self.getXYFromOrder(self.getrightHandOrder())
            dontCross = 1  # Penalty of do a big cross. Larger will drop the successRate
            if originalXY[0] < finalXY[0] - 2.5:
                dontCross = 0#2 * (finalXY[0] - 1.5 - originalXY[0])
                
            print("LHEasy? = ", self.successRateByHold(tryHold, "LH"))
            print("RHEasy? = ", self.successRateByHold(self.getrightHandHold(), "RH"))
            print("distanceOfFinalState = ", self.getTwoOrderDistance(self.getrightHandOrder(), tryOrder))
            return self.successRateByHold(tryHold, "LH") * self.successRateByHold(self.getrightHandHold(), "RH") * dontCross / (self.getTwoOrderDistance(self.getrightHandOrder(), tryOrder))**0.5
            
        elif op == "RH":
            originalXY = self.getXYFromOrder(self.getleftHandOrder())
            dontCross = 1
            if  finalXY[0] < originalXY[0] - 2.5:
                dontCross = 0#2 * (originalXY[0] - 1.5 - finalXY[0])
                
            print("RHEasy? = ", self.successRateByHold(tryHold, "RH"))
            print("LHEasy? = ", self.successRateByHold(self.getleftHandHold(), "LH"))
            print("distanceOfFinalState = ", self.getTwoOrderDistance(self.getleftHandOrder(), tryOrder))
            return self.successRateByHold(tryHold, "RH") * self.successRateByHold(self.getleftHandHold(), "LH") * dontCross / (self.getTwoOrderDistance(self.getleftHandOrder(), tryOrder))**0.5
            

    def addNextHand(self, nextHold, op):
        """ nextHold is a hold, whichHand are  LH, LG, RH, RG"""
        
        if self.touchEndHold == 3: 
            self.handSequence.append(self.totalNumOfHold - 1)  
            if self.handSequence[-1] == "LH":
                self.handOperator.append("RH")  
            if self.handSequence[-1] == "RH":
                self.handOperator.append("LH") 
            self.touchEndHold = self.touchEndHold + 1;
            self.isFinished = True
            #print(self.handSequence, self.handOperator)
        elif self.touchEndHold == 1 or self.isFinished == True: 
            pass
        else:
            if nextHold in self.getEndHoldOrder():
                self.touchEndHold = self.touchEndHold + 1;
                
            # Before Update a new hold
            originalCom = self.getCurrentCom()
            dynamicThreshold = hyperparameter[0] * self.lastMoveSuccessRateByHold()  
 
            # Update a new hold
            self.handSequence.append(nextHold)   # Add a new hold into beta!
            self.handOperator.append(op)         # indicate which hand
            if nextHold not in self.getEndHoldOrder():
                self.holdsNotUsed.remove(nextHold)   # Not consider match
            
            # after add a new hold
            if op == "LH":
                remainingHandOrder = self.getrightHandOrder()
            else:
                remainingHandOrder = self.getleftHandOrder()
            
            finalCom = self.getCom(remainingHandOrder, nextHold)
            distance = np.sqrt(((originalCom[0] - finalCom[0]) ** 2)+((originalCom[1] - finalCom[1]) ** 2))
            


            #newMoveSuccessRate = self.lastMoveSuccessRateByHold() * successRateByDistance(distance, dynamicThreshold)
            #self.singleMoveSuccessRate.append(newMoveSuccessRate)


    def setTrueBeta(self):
        self.isTrueBeta = True   
        
    def getleftHandOrder(self):
        lastIndexOfLeft = ''.join(self.handOperator).rindex('L') / 2
        return self.handSequence[int(lastIndexOfLeft)]
    
    def getrightHandOrder(self):
        """ Output the order of the handSequence"""
        lastIndexOfRight = ''.join(self.handOperator).rindex('R') / 2
        return self.handSequence[int(lastIndexOfRight)]
    
    def getleftHandHold(self):
        return self.allHolds[self.getleftHandOrder()]
    
    def getrightHandHold(self):
        """ Output the order of the handSequence"""
        return self.allHolds[self.getrightHandOrder()]
    
    def getOrderFromHold(self, hold):
        """ from a single hold (np array) to an order"""
        return np.where((self.allHolds == hold).all(1))[0] # Use np.where to get row indices
    
    def getCom(self, hold1Order, hold2Order):
        """ Get the coordinate of COM using current both hands order"""
        xCom = (self.allHolds[hold1Order][6] + self.allHolds[hold2Order][6]) / 2
        yCom = (self.allHolds[hold1Order][7] + self.allHolds[hold2Order][7]) / 2
        return (xCom, yCom)
    
    def getXYFromOrder(self, holdOrder):
        return ((self.allHolds[holdOrder][6]), (self.allHolds[holdOrder][7])) 
        
    def getCurrentCom(self):
        """ Get the coordinate of COM based on current hand position"""
        return self.getCom(self.getleftHandOrder(), self.getrightHandOrder())
    
    def getTwoOrderDistance(self, remainingHandOrder, nextHoldOrder):
        """ Given order 2, and 5. Output distance between"""
        originalCom = self.getCurrentCom()
        finalCom = self.getCom(remainingHandOrder, nextHoldOrder)
        return np.sqrt(((originalCom[0] - finalCom[0]) ** 2)+((originalCom[1] - finalCom[1]) ** 2))

    def orderToSeqOrder(self, order):
        """ Transform from rawdataorder to hand order"""
        return self.handSequence.index(order)
    
    def lastMoveSuccessRateByHold(self):
        operatorLeft = self.handOperator[self.orderToSeqOrder(self.getleftHandOrder())]
        operatorRight = self.handOperator[self.orderToSeqOrder(self.getrightHandOrder())]
        return self.successRateByHold(self.getleftHandHold(), operatorLeft) * self.successRateByHold(self.getrightHandHold(), operatorRight)
    def successRateByHold(self, hold, operation):
        """ """
        if operation == "LH": 
            return max((hold[0] + 2 * hold[1] + hold[2] + hold[5]) **1.2  , (hold[2] / 2 + hold[3] + hold[4])) / hyperparameter[1]
        if operation == "RH":
            return max((hold[2] + 2 * hold[3] + hold[4] + hold[5]) **1.2 , (hold[0] + hold[1] + hold[2] / 2)) / hyperparameter[1]
        
    def getStartHold(self):
        """return startHold list with 2 element"""
        startHoldList = []
        for hold in self.allHolds:
            if hold[8] == 1:
                startHoldList.append(hold)
        return startHoldList

    def getEndHoldOrder(self):
        endHoldOrderList = []
        for hold in self.allHolds:
            if hold[9] == 1:
                endHoldOrderList.append(self.getOrderFromHold(hold))
        if len(endHoldOrderList) == 1:
            endHoldOrderList = [endHoldOrderList[0], endHoldOrderList[0]]
        return endHoldOrderList
    
    def getholdsNotUsed(self):
        return self.holdsNotUsed  
    
    def overallSuccessRate(self):
        numOfHand = len(self.handSequence)
        overallScore = 1
        successScoreSequence = []
        for i, order in enumerate(self.handSequence):
            successScore1 = self.successRateByHold(self.allHolds[order], self.handOperator[i])
            overallScore = overallScore * successScore1

            if i != numOfHand-1:
                targetXY = self.getXYFromOrder(self.handSequence[i+1])
                #update last L/R hand
                if self.handOperator[i] == "RH":
                    lastrightHandXY = self.getXYFromOrder(self.handSequence[i]) 
                if self.handOperator[i] == "LH":    
                    lastleftHandXY = self.getXYFromOrder(self.handSequence[i])
                    
                if i >= 1:
                    if i == 1 and self.handSequence[0] == self.handSequence[1]:  ## not sure
                        targetXY = (targetXY[0], targetXY[1] - 1)
                    
                    if self.handOperator[i+1] == "RH": #targetXY[0] < originalXY[0] - 2.5 and
                        originalXY = lastleftHandXY
                        successScore2 = makeGaussian(targetXY, 3, (originalXY[0] , originalXY[1]), "LH")
                        overallScore = overallScore * successScore2
                        successScore = successScore1 * successScore2
                    elif self.handOperator[i+1] == "LH": #originalXY[0] < targetXY[0] - 2.5 and
                        originalXY = lastrightHandXY
                        successScore2 = makeGaussian(targetXY, 3, (originalXY[0], originalXY[1]), "RH")
                        overallScore = overallScore * successScore2
                        successScore = successScore1 * successScore2
                else:
                    successScore = successScore1
                
            else:
                successScore = successScore1
            
            successScoreSequence.append(successScore)
        self.overallSuccess = overallScore    
        self.successScoreSequence = successScoreSequence
        
        return overallScore ** (3/numOfHand)
    
def makeGaussian(targetXY, fwhm = 3, center = None, lasthand = "LH"):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = targetXY[0]
    y = targetXY[1]

    x0 = center[0]
    y0 = center[1]
    if lasthand == "RH":
        firstGauss = np.exp(-4*np.log(2) * ((x-(x0-3))**2 + (y-(y0+1.5))**2) / fwhm**2)
        secondGauss = np.exp(-4*np.log(2) * ((x-(x0+1))**2 + (y-(y0+0.5))**2) / fwhm**2) * 0.4
        thirdGauss =  np.exp(-4*np.log(2) * ((x-(x0))**2 + (y-(y0+1))**2) / fwhm**2) * 0.3
    if lasthand == "LH":
        firstGauss = np.exp(-4*np.log(2) * ((x-(x0+3))**2 + (y-(y0+1.5))**2) / fwhm**2)
        secondGauss = np.exp(-4*np.log(2) * ((x-(x0-1))**2 + (y-(y0+0.5))**2) / fwhm**2) * 0.4
        thirdGauss =  np.exp(-4*np.log(2) * ((x-(x0))**2 + (y-(y0+1))**2) / fwhm**2) * 0.3
    return  firstGauss + secondGauss

def rindex(lst, item):
    def index_ne(x):
        return lst[x] != item
    try:
        return next(dropwhile(index_ne, reversed(range(len(lst)))))
    except StopIteration:
        raise ValueError("rindex(lst, item): item not in list")  
        
def addNewBeta(status, printOut = True):
    """ Add one move to expand the candidate list and pick the largest 8"""
    tempstatus = []
    operationList = ["RH", "LH"]
    for betaPre in status:  # betaPreviousTwoCandidates       
        distanceScore = []
        for nextHoldOrder in betaPre.holdsNotUsed:
            
            originalCom = betaPre.getCurrentCom() 
            dynamicThreshold = hyperparameter[0] * betaPre.lastMoveSuccessRateByHold()  
            finalXY = betaPre.getXYFromOrder(nextHoldOrder)
    
            distance = np.sqrt(((originalCom[0] - finalXY[0]) ** 2)+((originalCom[1] - finalXY[1]) ** 2))
            distanceScore.append(successRateByDistance(distance, dynamicThreshold))

        # Find the first and second smallest distance
        largestIndex = heapq.nlargest(min(8, len(distanceScore)), range(len(distanceScore)), key=distanceScore.__getitem__)
        
        goodHoldIndex = [betaPre.holdsNotUsed[i] for i in largestIndex]
        
        added = False
        for possibleHold in goodHoldIndex:
            for op in operationList:
                if betaPre.isFinished == False:
                    tempstatus.append(copy.deepcopy(betaPre))
                    tempstatus[-1].addNextHand(possibleHold, op)
                elif added == False:
                    tempstatus.append(copy.deepcopy(betaPre))
                    added = True
    
    # trim tempstatus to pick the largest 3 
    finalScore = []       
    for i in tempstatus:
        finalScore.append(i.overallSuccessRate())    
    largestIndex = heapq.nlargest(8, range(len(finalScore)), key=finalScore.__getitem__) 
       
    if printOut == True:
        for i in largestIndex:
            print(tempstatus[i].handSequence, tempstatus[i].handOperator, tempstatus[i].overallSuccessRate()) 

    return [tempstatus[i] for i in largestIndex]

def successRateByDistance(distance, dynamicThreshold):
    """ Relu"""
    if distance < dynamicThreshold:
        return 1 - distance / dynamicThreshold
    if distance > dynamicThreshold:
        return 0

def produce_sequence(keyNum, X_dict, n_return = 1):
    '''
    Input: 
    - keyNum: the ID number of the problem
    - X_dict: the dictionary that contains the problem
    - n_return: number of best sequences to return
    Output:
    - a dictionary with key = the ranking of the output. Specific information of each output can be extracted using
      status.handSequence, status.handOperator, status.overallSuccessRate(), status.successScoreSequence
    '''
    moonboardTest = X_dict[keyNum]
    testbeta = beta(moonboardTest.T)
    status = [beta(moonboardTest.T), beta(moonboardTest.T)]
    status[0].addStartHolds(0)
    status[1].addStartHolds(1)
    operationList = ["RH", "LH"]
    tempstatus = []
    tempstatus2 = []
    tempstatus3 = []
    distanceScore = []
    
    # Run the algorithm for 6 times
    totalRun = status[0].totalNumOfHold - 1
    for i in range(totalRun):  # how many new move you wan to add
        status = addNewBeta(status, printOut = False)
        finalScore = [] 
        for i in status:   
            finalScore.append(i.overallSuccessRate())
        largestIndex = heapq.nlargest(4, range(len(finalScore)), key=finalScore.__getitem__)
        if (status[largestIndex[0]].isFinished and status[largestIndex[1]].isFinished) == True:
            break
    
    # last sorting for the best 5
    finalScore = [] 
    for i in status:   
        finalScore.append(i.overallSuccessRate())   
    largestIndex = heapq.nlargest(n_return, range(len(finalScore)), key=finalScore.__getitem__)
    
    # produce output
    output = {}
    print ("After Beamer search, the most possible hand sequence and the successRate:")
    for i in largestIndex:
        print(status[i].handSequence, status[i].handOperator, status[i].overallSuccessRate())
        print(status[i].successScoreSequence)
        output[i] = status[i]
    
    return output

"""Common Function"""

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

"""For Final Preprocessing Step"""
def get_grade_map():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6A': 0,
        '6A+': 1,
        '6B': 2,
        '6B+': 3,
        '6C': 4,
        '6C+': 5,
        '7A': 6,
        '7A+': 7,
        '7B': 8,
        '7B+': 9,
        '7C': 10,
        '7C+': 11,
        '8A': 12,
        '8A+': 13,
        '8B': 14,
        '8B+': 15,
    }
    return grade_map

def get_grade_map_new():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6B': 0,
        '6B+': 1,
        '6C': 2,
        '6C+': 3,
        '7A': 4,
        '7A+': 5,
        '7B': 6,
        '7B+': 7,
        '7C': 8,
        '7C+': 9,
        '8A': 10,
        '8A+': 11,
        '8B': 12,
        '8B+': 13,
    }
    return grade_map