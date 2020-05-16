# Authors: Aaron Wu, Howard Tai

# Script containing helper functions for accessing and navigating the MoonBoard web-site

import numpy as np
import os
import copy
import pickle
import heapq
import matplotlib.pyplot as plt
import pandas as pd

global hyperparameter, RightHandfeature_dict, LeftHandfeature_dict, operationList, MoonBoard_2016_withurl

hyperparameter = [1, 1]
operationList = ["RH", "LH"]

cwd = os.getcwd()
parent_wd = cwd.replace('/preprocessing', '')
left_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016LeftHand.csv'
right_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016RightHand.csv'
url_data_path = parent_wd + '/raw_data/moonGen_scrape_2016_cp'

LeftHandfeatures = pd.read_csv(left_hold_feature_path, dtype=str)
RightHandfeatures = pd.read_csv(right_hold_feature_path, dtype=str)
# convert features from pd dataframe to dictionary of left and right hand
RightHandfeature_dict = {}
LeftHandfeature_dict = {}
for index in RightHandfeatures.index:
    LeftHandfeature_item = LeftHandfeatures.loc[index]
    LeftHandfeature_dict[(int(LeftHandfeature_item['X_coord']), int(LeftHandfeature_item['Y_coord']))] = np.array(
        list(LeftHandfeature_item['Difficulties'])).astype(int)
    RightHandfeature_item = RightHandfeatures.loc[index]
    RightHandfeature_dict[(int(RightHandfeature_item['X_coord']), int(RightHandfeature_item['Y_coord']))] = np.array(
        list(RightHandfeature_item['Difficulties'])).astype(int)
with open(url_data_path, 'rb') as f:
    MoonBoard_2016_withurl = pickle.load(f)
    
# ----------------------------------------------------------------------------------------------------------------------
# General utility functions
# ----------------------------------------------------------------------------------------------------------------------
def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None


def load_pickle(file_name):
    """
    Loads data from pickle format
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def remove_duplicates(repeats_data):
    """
    Removes repeats from a list
    """
    repeats = []
    for r in repeats_data:
        if r not in repeats:
            repeats.append(r)
    return repeats

def get_grade_map():
    """
    Defines a mapping of Fontainebleau grades to integer values
    """
    grade_map = {
        '6B': 0,  # V4
        '6B+': 0, # V4
        '6C': 1,  # V5
        '6C+': 1, # V5
        '7A': 2,  # V6
        '7A+': 3, # V7
        '7B': 4,  # V8
        '7B+': 4, # V8
        '7C': 5,  # V9
        '7C+': 6, # V10
        '8A': 7,  # V11
        '8A+': 8, # V12
        '8B': 9,  # V13
    }
    return grade_map

def get_grade_FtToV():
    grade_FtToV = {
        0: 0, #original 6B, new grade V4
        1: 0,
        2: 0,
        3: 0,
        4: 1, # original 6C, they are all V5
        5: 1, # original 6C+, they are all V5
        6: 2,
        7: 3, 
        8: 4, # original 7B, they are all V8
        9: 4, # original 7B+, they are all V8
        10: 5,
        11: 6,
        12: 7,
        13: 8, 
        14: 9, #original 8A+, new grade V13. Delete V14 because many people joke on V14
    }
    return grade_FtToV
# ----------------------------------------------------------------------------------------------------------------------
# sequence generation related function
# ----------------------------------------------------------------------------------------------------------------------

class beta:
    """ climbing beta is the climber's feeling of the best hand sequence.
    This class beta stored information including 1. moonboard holds information.
    2. the ongoing built hand sequence (LRLRLRRR) 3. and the hand sequence climber used
    We can add new sequence and evaluate the successful rate of this beta to find out the best one.
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
        """ Specifically add the first two hold as the starting hold. Consider one hold start situation"""
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
        """ return all avalible holds. N holds rows, 10 columns np array"""
        return self.allHolds
    
    def addNextHand(self, nextHold, op):
        """ Operation to make add the next hold. Append handsequence and hand operation. nextHold is a hold. op is "LH" or "RH" """     
        if self.touchEndHold == 3: 
            self.handSequence.append(self.totalNumOfHold - 1)  
            if self.handSequence[-1] == "LH":
                self.handOperator.append("RH")  
            if self.handSequence[-1] == "RH":
                self.handOperator.append("LH") 
            self.touchEndHold = self.touchEndHold + 1;
            self.isFinished = True

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

    def getXYFromOrder(self, holdOrder):
        """return a coordinate tuple giving holdOrder (a num in processed data)"""
        return ((self.allHolds[holdOrder][6]), (self.allHolds[holdOrder][7])) 
    
    def getleftHandOrder(self):
        """ Return a num of the last left hand hold's oreder (in processed data from bottom to top)"""
        lastIndexOfRight = ''.join(self.handOperator).rindex('R') / 2
        return self.handSequence[int(lastIndexOfRight)]
    
    def getrightHandOrder(self):
        """ Return a num of the last right hand hold's oreder (in processed data from bottom to top)"""
        lastIndexOfRight = ''.join(self.handOperator).rindex('R') / 2
        return self.handSequence[int(lastIndexOfRight)]

    def getleftHandHold(self):
        """ Return a np array of the last right hand hold (in processed data from bottom to top)"""
        return self.allHolds[self.getleftHandOrder()]
    
    def getrightHandHold(self):
        """ Return a np array of the last right hand hold (in processed data from bottom to top)"""
        return self.allHolds[self.getrightHandOrder()]
    
    def getOrderFromHold(self, hold):
        """ from a single hold (np array) to an order"""
        return np.where((self.allHolds == hold).all(1))[0] # Use np.where to get row indices
    
    def getCom(self, hold1Order, hold2Order):
        """ Get the coordinate of COM using current both hands order"""
        xCom = (self.allHolds[hold1Order][6] + self.allHolds[hold2Order][6]) / 2
        yCom = (self.allHolds[hold1Order][7] + self.allHolds[hold2Order][7]) / 2
        return (xCom, yCom)

        
    def getCurrentCom(self):
        """ Get the coordinate of COM based on current hand position"""
        return self.getCom(self.getleftHandOrder(), self.getrightHandOrder())
    
    def getTwoOrderDistance(self, remainingHandOrder, nextHoldOrder):
        """ Given order 2, and 5. Output distance between"""
        originalCom = self.getCurrentCom()
        finalCom = self.getCom(remainingHandOrder, nextHoldOrder)
        return np.sqrt(((originalCom[0] - finalCom[0]) ** 2)+((originalCom[1] - finalCom[1]) ** 2))

    def orderToSeqOrder(self, order):
        """ Transform from order (in the all avalible holds sequence) to hand order (in the hand sequence)"""
        return self.handSequence.index(order)
    
    def lastMoveSuccessRateByHold(self):
        operatorLeft = self.handOperator[self.orderToSeqOrder(self.getleftHandOrder())]
        operatorRight = self.handOperator[self.orderToSeqOrder(self.getrightHandOrder())]
        return self.successRateByHold(self.getleftHandHold(), operatorLeft) * self.successRateByHold(self.getrightHandHold(), operatorRight)
    
    def successRateByHold(self, hold, operation):
        """ Evaluate the difficulty to hold on a hold applying LH or RH (op)"""
        if operation == "LH": 
            return LeftHandfeature_dict[(hold[6], hold[7])] #Chiang's evaluation
            # Duh's evaluation
            #return max((hold[0] + 2 * hold[1] + hold[2] + hold[5]) **1.2  , (hold[2] / 2 + hold[3] + hold[4])) / hyperparameter[1]  
        if operation == "RH":
            return RightHandfeature_dict[(hold[6], hold[7])] #Chiang's evaluation
            #return max((hold[2] + 2 * hold[3] + hold[4] + hold[5]) **1.2 , (hold[0] + hold[1] + hold[2] / 2)) / hyperparameter[1]
        
    def getStartHold(self):
        """return startHold list with 2 element of np array"""
        startHoldList = []
        for hold in self.allHolds:
            if hold[8] == 1:
                startHoldList.append(hold)
        return startHoldList

    def getEndHoldOrder(self):
        """return endHold list with 2 element of np array"""
        endHoldOrderList = []
        for i in range(self.totalNumOfHold):
            if self.allHolds[i][9] == 1:
                endHoldOrderList.append(i)
        if len(endHoldOrderList) == 1:
            endHoldOrderList.append(self.totalNumOfHold)
        return endHoldOrderList
    
    def overallSuccessRate(self):
        """return the overall successful rate using the stored beta hand sequence"""
        numOfHand = len(self.handSequence)
        overallScore = 1;
        for i, order in enumerate(self.handSequence): 
            overallScore = overallScore * self.successRateByHold(self.allHolds[order], self.handOperator[i])
  
        for i in range (numOfHand - 1):  # Penalty of do a big cross. Larger will drop the successRate   
            targetXY = self.getXYFromOrder(self.handSequence[i+1]) 
            
            #update last L/R hand
            if self.handOperator[i] == "RH":
                lastrightHandXY = self.getXYFromOrder(self.handSequence[i]) 
            if self.handOperator[i] == "LH":    
                lastleftHandXY = self.getXYFromOrder(self.handSequence[i])
                
            if i == 1 and self.handSequence[0] == self.handSequence[1]:  ## not sure
                targetXY = (targetXY[0], targetXY[1] - 1)
            
            if i >= 1 and self.handOperator[i+1] == "RH": 
                originalXY = lastleftHandXY
                overallScore = overallScore * makeGaussian(targetXY, 3, (originalXY[0] , originalXY[1]), "LH")
            if i >= 1 and self.handOperator[i+1] == "LH": 
                originalXY = lastrightHandXY
                overallScore = overallScore * makeGaussian(targetXY, 3, (originalXY[0], originalXY[1]), "RH")
        self.overallSuccess = overallScore    
        
        return overallScore ** (3/numOfHand) 
    
    def setTrueBeta(self):
        self.isTrueBeta = True  
        
    def getholdsNotUsed(self):
           return self.holdsNotUsed      
    
    ############ Below are some other function that is not useful in the current version
    
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

    
def makeGaussian(targetXY, fwhm = 3, center = None, lasthand = "LH"):
    """ Make a square gaussian filter to evaluate how possible of the relative distance between hands
    from target hand to remaining hand (center)
    fwhm is full-width-half-maximum, which can be thought of as an effective distance of dynamic range.
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

def successRateByDistance(distance, dynamicThreshold):
    """ Relu funtion to get the successrate """
    if distance < dynamicThreshold:
        return 1 - distance / dynamicThreshold
    if distance >= dynamicThreshold:
        return 0
    
def rindex(lst, item):
    """ return the index of item in an array counting from the rear"""
    def index_ne(x):
        return lst[x] != item
    try:
        return next(dropwhile(index_ne, reversed(range(len(lst)))))
    except StopIteration:
        raise ValueError("rindex(lst, item): item not in list")  

def addNewBeta(status, printOut = True):
    """ Add one move to expand the candidate list and pick the largest 8"""
    tempstatus = []
    for betaPre in status:  # betaPreviousTwoCandidates       
        distanceScore = []
        for nextHoldOrder in betaPre.holdsNotUsed:
            
            originalCom = betaPre.getCurrentCom() 
            dynamicThreshold = hyperparameter[0] * betaPre.lastMoveSuccessRateByHold()  
            finalXY = betaPre.getXYFromOrder(nextHoldOrder)
    
            distance = np.sqrt(((originalCom[0] - finalXY[0]) ** 2)+((originalCom[1] - finalXY[1]) ** 2))
            distanceScore.append(successRateByDistance(distance, dynamicThreshold))  # evaluate success rate simply consider the distance (not consider left and right hand)

        # Find the first and second smallest distance in the distanceScore
        largestIndex = heapq.nlargest(min(8, len(distanceScore)), range(len(distanceScore)), key=distanceScore.__getitem__)
        
        #goodHoldIndex = [betaPre.holdsNotUsed[largestIndex[0]], betaPre.holdsNotUsed[largestIndex[1]], betaPre.holdsNotUsed[largestIndex[2]]]  #[#3,#5] holds
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
                   
    
    # trim tempstatus to pick the largest 8
    finalScore = []       
    for i in tempstatus:
        finalScore.append(i.overallSuccessRate())    
    largestIndex = heapq.nlargest(8, range(len(finalScore)), key=finalScore.__getitem__) 
       
    if printOut == True:
        for i in largestIndex:
            print(tempstatus[i].handSequence, tempstatus[i].handOperator, tempstatus[i].overallSuccessRate()) 

    return [tempstatus[i] for i in largestIndex] 

#=====================================================
def oppositehand(astring):
    if astring == "LH":
        return "RH"
    if astring == "RH":
        return "LH"

def holdScoreUseCordAndOp (coordination, operation):
    """ return the difficulty of each hold using LH or RH"""
    if operation == "RH": 
        return RightHandfeature_dict[coordination]
    if operation == "LH": 
        return LeftHandfeature_dict[coordination]

def moveGenerator(beta, string_mode = True):
    """ generate the final output of move sequence as a list of dictionary.
    Length of the list: how many moves in this climb to the target hold. Target holds run from the third order hold to the last hold
    Dictionary involves all information needed to evaluate grade/ analyze style for human. This is a basic building block of the route.
    TargetHoldString : "A10" for example
    TargetHoldHand: "RH" for example 
    TargetHoldScore: the difficulty to hold on the target hold applying the "RH" operation
    RemainingHoldString : "A10" for example
    RemainingHoldHand: 
    RemainingHoldScore 
    MovingHoldString : A10 for example
    MovingHoldHand: 
    MovingHoldScore: 
    dxdyMtoT: vector Target - moving hand. This distance's physical meaning is the real hand traveling range during the move
    dxdyRtoT: vector Target - Remaining hand. This distance's physical meaning is the inter distance between two remaining hand after finish the move
    FootPlacement: [0,0,0,0,1,1,0] means there is hold on region 5 and 6. 
    MoveSuccessRate: estimation of how easy of this move 
    if coordinate_mode = True, String will be coordinate form and 
    """
    
    outputDictionaryList = []
    numOfMoves = len(beta.handSequence) - 2 # calculate from the third hold to the end hold (no final match)
    # loop over holds from third one to the finish hold (rank from 3 to end). In each move, this is the hold defined as target hold
    for rank in range(2,  len(beta.handSequence)):
        # Renew a dictionary
        moveDictionary = {}
        
        # Define target hold
        order = beta.handSequence[rank] # order is the original order (int) of hold read from bottom to top
        targetHoldHand = beta.handOperator[rank]
        coordinateOfTarget = beta.getXYFromOrder(order) 
        
        if string_mode == False:
            moveDictionary["TargetHoldString"] = coordinateOfTarget
            if targetHoldHand == "LH":
                moveDictionary["TargetHoldHand"] = 0   # LH ->0
            else: moveDictionary["TargetHoldHand"] = 1  # RH -> 1 
        else: 
            moveDictionary["TargetHoldString"] = coordinateToString(coordinateOfTarget)
            moveDictionary["TargetHoldHand"] = targetHoldHand
        moveDictionary["TargetHoldScore"] = holdScoreUseCordAndOp(coordinateOfTarget, targetHoldHand)   # Could you file I/O excile file L/R hand difficulty?
        
        # Define remaining hold
        listBeforeTargetHold = beta.handOperator[0:rank]

        remainingHoldHand = oppositehand(targetHoldHand)
        order = int(''.join(listBeforeTargetHold).rindex(remainingHoldHand)/2) # remaining hold is the last hold with opposite hand in the sequence before Target hand
        coordinateOfRemaining = beta.getXYFromOrder(beta.handSequence[order]) 
    
        if string_mode == False:
            moveDictionary["RemainingHoldString"] = coordinateOfRemaining
            moveDictionary["RemainingHoldHand"] = 1 - moveDictionary["TargetHoldHand"]
        else: 
            moveDictionary["RemainingHoldString"] = coordinateToString(coordinateOfRemaining)
            moveDictionary["RemainingHoldHand"] = remainingHoldHand
        moveDictionary["RemainingHoldScore"] = holdScoreUseCordAndOp(coordinateOfRemaining, remainingHoldHand)
        moveDictionary["dxdyRtoT"] = (coordinateOfTarget[0] - coordinateOfRemaining[0], coordinateOfTarget[1] - coordinateOfRemaining[1])
        
        # Define moving hold
        movingHoldHand = targetHoldHand
        order = int(''.join(listBeforeTargetHold).rindex(movingHoldHand)/2) # remaining hold is the last hold with opposite hand in the sequence before Target hand
        coordinateOfMoving = beta.getXYFromOrder(beta.handSequence[order]) 
        
        if string_mode == False:
            moveDictionary["MovingHoldString"] = coordinateOfMoving
            moveDictionary["MovingHoldHand"] = moveDictionary["TargetHoldHand"]
        else: 
            moveDictionary["MovingHoldString"] = coordinateToString(coordinateOfMoving)
            moveDictionary["MovingHoldHand"] = movingHoldHand
        moveDictionary["MovingHoldScore"] = holdScoreUseCordAndOp(coordinateOfMoving, movingHoldHand)
        moveDictionary["dxdyMtoT"] = (coordinateOfTarget[0] - coordinateOfMoving[0], coordinateOfTarget[1] - coordinateOfMoving[1])
        
        # Define foot region location
        x0, y0 = int(coordinateOfRemaining[0]), int(coordinateOfRemaining[1])
        region0 = [(x,y) for x in range(x0 - 4, x0 - 1) for y in range(y0 - 3, y0 - 1)]
        region1 = [(x,y) for x in range(x0 - 1, x0 + 2) for y in range(y0 - 3, y0 - 1)]
        region2 = [(x,y) for x in range(x0 + 2, x0 + 5) for y in range(y0 - 3, y0 - 1)]
        region3 = [(x,y) for x in range(x0 - 5, x0 - 1) for y in range(y0 - 6, y0 - 3)]
        region4 = [(x,y) for x in range(x0 - 1, x0 + 2) for y in range(y0 - 6, y0 - 3)]
        region5 = [(x,y) for x in range(x0 + 2, x0 + 6) for y in range(y0 - 6, y0 - 3)]
        region6 = [(x,y) for x in range(x0 - 2, x0 + 3) for y in range(y0 - 9, y0 - 6)]

        # check is there foot holds in the region
        footholdList = [0] * 7 
        regionList = [region0, region1, region2, region3, region4, region5, region6]
        for hold in beta.allHolds:
            holdx, holdy = hold[6], hold[7]
            for i in range(7):
                if (holdx, holdy) in regionList[i]:
                    footholdList[i] = 1 
            # deal with additional footholds        
            if region1[0][1] < 0: # if the lowest hold in region1 is < 0, we can use additional footholds (region's first element start from the lowest)
                footholdList[0] = 1
                footholdList[1] = 1
                footholdList[2] = 1
            elif region4[0][1] < 0: # if the lowest hold in region1 is < 0, we can use additional footholds
                footholdList[3] = 1
                footholdList[4] = 1
                footholdList[5] = 1 
            elif region6[0][1] < 0: # if the lowest hold in region1 is < 0, we can use additional footholds
                footholdList[6] = 1
            
        moveDictionary["FootPlacement"] = footholdList  
        
        # Add the singlemoveSuccessRate
        if coordinateOfMoving == coordinateOfRemaining:  ## If start from the match position
            pass # May need special consideration when match hand
        if targetHoldHand == "RH": 
            scoreFromDistance =  makeGaussian(coordinateOfTarget, 3, coordinateOfRemaining, "LH")
        if targetHoldHand == "LH": 
            scoreFromDistance =  makeGaussian(coordinateOfTarget, 3, coordinateOfRemaining, "RH")
            
        scoreFromfoot = 1    
        if sum(footholdList) < 1: scoreFromfoot = 0.5  
 
        moveSuccessRate = moveDictionary["RemainingHoldScore"] * moveDictionary["TargetHoldScore"] * scoreFromDistance * scoreFromfoot
        moveDictionary["MoveSuccessRate"] = moveSuccessRate
        
        # Finish fill in all components of a move
        outputDictionaryList.append(moveDictionary)
    return outputDictionaryList

def moveGeneratorForAllProblem(processed_data, save_path, print_result = False):
    """
    Apply moveGenerator on all problems, and transform the output into X vector move sequence for RNN
    Input:
    - processed_data: the processed data that is scraped from MoonBoard
    - i.e. '/preprocessing/processed_data_xy_mode'  MoonBoard_2016_raw["X_dict_benchmark_withgrade"]
    
    Output:
    dictionary-- key: x_vector
    x_vector dim is 22 * numOfMoves
    22 is the embedding feature vector's dimension. In other word, every climbing move can be characterized using 22 dimension vector
    so we will have:
    vector1- vector2- vector3....last vector = move1- move2- move3... last move. This sequence will be feed into RNN as X
    """
    output = {}
    list_fail = []
    
    for key in processed_data.keys():
        # create x_vector
        try:
            beamerBeta = produce_sequence(key, processed_data, n_return = 1)[0]
            numOfMoves = len(beamerBeta.handSequence) - 2
            movesInfoList = moveGenerator(beamerBeta, string_mode = False)
            x_vectors = np.zeros((22, numOfMoves))
            
            for orderOfMove, moveInfoDict in enumerate(movesInfoList):   
                #print(x_vectors[0:2, orderOfMove])
                #print(moveInfoDict['TargetHoldString'])
                
                x_vectors[0:2, orderOfMove] = moveInfoDict['TargetHoldString'] 
                x_vectors[2, orderOfMove] = moveInfoDict['TargetHoldHand'] # only express once
                x_vectors[3, orderOfMove] = moveInfoDict['TargetHoldScore']
                x_vectors[4:6, orderOfMove] = moveInfoDict['RemainingHoldString']
                x_vectors[6, orderOfMove] = moveInfoDict['RemainingHoldScore']
                x_vectors[7:9, orderOfMove] = moveInfoDict['dxdyRtoT']
                x_vectors[9:11, orderOfMove] = moveInfoDict['MovingHoldString']
                x_vectors[11, orderOfMove] = moveInfoDict['MovingHoldScore']
                x_vectors[12:14, orderOfMove] = moveInfoDict['dxdyMtoT']
                x_vectors[14:21, orderOfMove] = moveInfoDict['FootPlacement']
                x_vectors[21, orderOfMove] = moveInfoDict['MoveSuccessRate']
                
            if print_result:
                print('Complete %s' %key)
            output[key] = x_vectors  
        except:
            print('Raw data with key %s contains error' %key)
            list_fail.append(key)

    save_pickle(output, save_path)
    print('result saved.')
    return output, list_fail

def gradeTransFromFontToV(processed_data_Y, save_path):
    """ 
    convert font scale grade to v scale grade. Drop V14. 
    input:     
    - processed_data: the processed Y data that is scraped from MoonBoard
    - i.e. '/preprocessing/processed_data_xy_mode'  MoonBoard_2016_raw["Y_dict_benchmark_withgrade"]
    
    There are three element: grade, isBenchMark, usergrade
    We will take the average of grade and usergrade. Round up to increase more higher grade's population
    Should we drop out unrepeat problem?
    
    output: This Y file is ready for many to 1 RNN
    """
    output = {}
    fail_list = []
    grade_FtToV = get_grade_FtToV()
    try:
        processed_data_Y[processed_data_Y.keys()[0]][2]
        for key in processed_data_Y.keys():
            try:
                fontGrade = int(np.ceil(processed_data_Y[key][0] + processed_data_Y[key][2]) / 2)  # so V5 and V6 will return V6.
                vGrade = grade_FtToV[fontGrade]
                output[key] = vGrade
            except:
                fail_list.append(key)
                print('key %s failed.' %key)
    except:
        for key in processed_data_Y.keys():
            try:
                fontGrade = int(processed_data_Y[key][0])
                vGrade = grade_FtToV[fontGrade]
                output[key] = vGrade
            except:
                fail_list.append(key)
                print('key %s failed.' %key)
        
    save_pickle(output, save_path)
    print('result saved.')
    return output, fail_list

def produce_sequence(keyNum, X_dict, n_return = 1, printout = False):
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
    moonboardTestUrl = MoonBoard_2016_withurl[keyNum]
    testbeta = beta(moonboardTest.T)
    status = [beta(moonboardTest.T), beta(moonboardTest.T)]
    status[0].addStartHolds(0)
    status[1].addStartHolds(1)
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
    if printout == True:
        print(moonboardTestUrl["url"])
        print ("After Beamer search, the most possible hand sequence and the successRate:")
    for i in largestIndex:
        output[i] = status[i]
        if printout == True:
            print([coordinateToString(status[i].getXYFromOrder(j)) for j in status[i].handSequence])
            print(status[i].handSequence, status[i].handOperator, status[i].overallSuccessRate())
        #print(status[i].successScoreSequence)
    
    return output

def coordinateToString(coordinate):
    """ convert (9.0 ,4.0) to "J5" """
    alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    return str(alphabateList[int(coordinate[0])]) + str(int(coordinate[1]) + 1)

# =============================================================
# Normalization
# =============================================================
def normalization(input_set):
    mu_x = 5.0428571
    sig_x = 3.079590
    mu_y = 9.8428571
    sig_y = 4.078289957
    mu_hand = 4.2428571
    sig_hand = 2.115829552
    mu_diff = 12.118308
    sig_diff = 11.495348196
    
    mu_vec = np.array([mu_x, mu_y, 0, mu_hand, mu_x, mu_y, mu_hand, 0, 0, mu_x, mu_y, mu_hand, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu_diff])
    sig_vec = np.array([sig_x, sig_y, 1, sig_hand, sig_x, sig_y, sig_hand, sig_x, sig_y, sig_x, sig_y, sig_hand, sig_x, sig_y, 1, 1, 1, 1, 1, 1, 1, sig_diff])
    
    mask = np.zeros_like(input_set['X'])
    for i in range(len(mask)):
        mask[i, 0:int(input_set['tmax'][i]), :] = 1
    
    X_normalized = np.copy(input_set['X'])
    X_normalized -= mu_vec
    X_normalized /= sig_vec
    X_normalized *= mask
    
    output_set = input_set
    output_set['X'] = X_normalized
    return output_set

def normalization_move(input_set):
    mu_x = 5.0428571
    sig_x = 3.079590
    mu_y = 9.8428571
    sig_y = 4.078289957
    mu_hand = 4.2428571
    sig_hand = 2.115829552
    mu_diff = 12.118308
    sig_diff = 11.495348196
    
    mu_vec = np.array([mu_x, mu_y, 0, mu_hand, mu_x, mu_y, mu_hand, 0, 0, mu_x, mu_y, mu_hand, 0, 0, 0, 0, 0, 0, 0, 0, 0, mu_diff])
    sig_vec = np.array([sig_x, sig_y, 1, sig_hand, sig_x, sig_y, sig_hand, sig_x, sig_y, sig_x, sig_y, sig_hand, sig_x, sig_y, 1, 1, 1, 1, 1, 1, 1, sig_diff])
    
    X_normalized = np.copy(input_set['X'])
    X_normalized -= mu_vec
    X_normalized /= sig_vec
    
    output_set = input_set
    output_set['X'] = X_normalized
    return output_set