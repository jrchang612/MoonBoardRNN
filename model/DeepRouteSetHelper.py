'''
Helper funcitons for DeepRoutSet 
Yi-Shiou Duh (allenduh@stanford.edu)
'''

import numpy as np
import os
import copy
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cbook as cbook
import re
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Input, LSTM, Reshape, Lambda, RepeatVector
from keras import backend as K

cwd = os.getcwd()
parent_wd = cwd.replace('/model', '')
benchmark_handString_seq_path = parent_wd + '/preprocessing/benchmark_handString_seq_X'
benchmarkNoGrade_handString_seq_path = parent_wd + '/preprocessing/benchmarkNoGrade_handString_seq_X'
nonbenchmark_handString_seq_path = parent_wd + '/preprocessing/nonbenchmark_handString_seq_X'
nonbenchmarkNoGrade_handString_seq_path = parent_wd + '/preprocessing/nonbenchmarkNoGrade_handString_seq_X'
url_data_path = parent_wd + '/raw_data/moonGen_scrape_2016_cp'

with open(benchmark_handString_seq_path, 'rb') as f:
    benchmark_handString_seq = pickle.load(f)
with open(benchmarkNoGrade_handString_seq_path, 'rb') as f:
    benchmarkNoGrade_handString_seq = pickle.load(f)
with open(nonbenchmark_handString_seq_path, 'rb') as f:
    nonbenchmark_handString_seq = pickle.load(f)
with open(nonbenchmarkNoGrade_handString_seq_path, 'rb') as f:
    nonbenchmarkNoGrade_handString_seq = pickle.load(f)        
with open(url_data_path, 'rb') as f:
    MoonBoard_2016_withurl = pickle.load(f)
    
# Feed in the hold feature.csv files
left_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016LeftHand.csv'
right_hold_feature_path = parent_wd + '/raw_data/HoldFeature2016RightHand.csv'

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

"""Now add more problems and add more Benchmark problem to emphasize Benchmark"""
def collectHandStringIntoList(levelProblemList, duplicateBenchMark = 6):
    ## Add other nonbenchmark into StringList
    ## Need to pass throuh certain filter
    ## 1. Can't be longer than 11 hands
    ## 2. Be in goodProblem key list
    handStringList = []
    levelProblemList = levelProblemList
    # Add duplicated benchMark in the training example to emphasize benchMark
    for i in range(duplicateBenchMark):
        for key in benchmark_handString_seq.keys():
            if key in levelProblemList:
                handStringList.append(benchmark_handString_seq[key])
    for key in benchmarkNoGrade_handString_seq.keys():
        if key in levelProblemList:
            if len(benchmarkNoGrade_handString_seq[key]) < 12:
                handStringList.append(benchmarkNoGrade_handString_seq[key])
    for key in nonbenchmark_handString_seq.keys():
        if key in levelProblemList:
            if len(nonbenchmark_handString_seq[key]) < 12:
                handStringList.append(nonbenchmark_handString_seq[key])
    for key in nonbenchmarkNoGrade_handString_seq.keys():
        if key in levelProblemList:
            if len(nonbenchmarkNoGrade_handString_seq[key]) < 12:
                handStringList.append(nonbenchmarkNoGrade_handString_seq[key]) 
    return handStringList            

def loadSeqXYFromString (stringList, holdStr_to_holdIx, m, numOfPossibleHolds, maxNumOfHands = 12):
    """Input with HandSting list ['J5-LH', 'J5-RH', 'I9-LH', 'J10-RH', 'H12-RH', 'C13-LH', 'D15-LH', 'E18-RH']
       Different training sample have different length, so padded with 0 ("End") up to maxNumOfHands
       OutPut X, Y HandString matrix to feed in RNN
       OutPut shape X (Training sample, Tx, numOfPossibleHolds + "End") = (numOfTrainingSample, 12, n_values)
       OutPut shape Y (Tx, Training sample, numOfPossibleHolds + "End") = (12, numOfTrainingSample, n_values)
    """
    n_values = numOfPossibleHolds + 1 # including "End"
    X = np.zeros((m, maxNumOfHands, n_values), dtype=np.bool)
    Y = np.zeros((m, maxNumOfHands, n_values), dtype=np.bool)
    for ixOfSample in range(m):
        # Extract a seq like ['J5-LH', 'J5-RH', 'I9-LH', 'J10-RH', 'H12-RH', 'C13-LH', 'D15-LH', 'E18-RH']
        one_Seq = stringList[ixOfSample]
        
        # Convert each string to index
        ixList = [holdStr_to_holdIx[string] for string in one_Seq]
        
        # Specify one hot X, Y and take care of padding. Note that Y[n] = X[n+1]
        # Pad 0 after the end hold up to maxNumOfHands
        for j in range(maxNumOfHands):
            if j >= len(ixList): # condition to pad
                Y[ixOfSample, j, 0] = 1 # Pad 0
                if j+1 < maxNumOfHands:
                    X[ixOfSample, j+1, 0] = 1  # Pad 0          
            else:
                idx = ixList[j]  # Y[n] = X[n+1]
                X[ixOfSample, j+1, idx] = 1
                Y[ixOfSample, j, idx] = 1
                
    Y = np.swapaxes(Y,0,1)  # Y is different shape than X
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), n_values 

def routeSetmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the climbing move data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    reshapor = Reshape((1, n_values))                  
    LSTM_cell = LSTM(n_a, return_state = True)        
    densor = Dense(n_values, activation='softmax')
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):      
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda z: z[:, t, :])(X)   
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)  # from (?, n_values) to (?, 1, n_values)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs = x, initial_state = [a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs = [X, a0, c0], outputs = outputs)
    
    return model
            
""" Finally sanity check"""
def sanityCheckAndOutput(indices, holdIx_to_holdStr, handStringList, outputExactFromDatabase = False, printError = False):
    lastString = ""
    outputListInString = []
    outputListInIx = []
    passCheck = True
    for i in range(12):
        if lastString != "End": 
            # Check is there repetitive holds. Unlike music generation, don't allow the same hold, same hand happen consequtively.
            if holdIx_to_holdStr[int(indices[i])] == lastString:# and i < 3:
                passCheck = False
                if printError: print("Repeat hand error", outputListInString)
                return passCheck, outputListInString, outputListInIx
            # Check is this problem end (filter out problem with no ?18-LH / ?18-RH )   
            if holdIx_to_holdStr[int(indices[i])] == "End" and "18" not in lastString:
                passCheck = False
                if printError: print("No end error", outputListInString)
                return passCheck, outputListInString, outputListInIx
            # Check if the last(i=12), is this problem end at 18
            if i == 11 and "18" not in holdIx_to_holdStr[int(indices[i])]:
                passCheck = False
                if printError: print("No end error", outputListInString)
                return passCheck, outputListInString, outputListInIx
            
            if holdIx_to_holdStr[int(indices[i])] != "End":
                outputListInString.append(holdIx_to_holdStr[int(indices[i])]) 
                outputListInIx.append(int(indices[i])) 
            lastString = holdIx_to_holdStr[int(indices[i])]
            
    outputListInStringSet = set(outputListInString) # Delete repetitive string
    outputListInIxSet = set(outputListInIx) # Delete repetitive string
    
    # Check if the second hold match with third hold
    stringOfSecondHold = holdIx_to_holdStr[int(indices[1])].split("-")[0]
    stringOfThirdHold = holdIx_to_holdStr[int(indices[2])].split("-")[0]
    if stringOfSecondHold == stringOfThirdHold:
        passCheck = False
        if printError: print("Warning: Second match hand with third", outputListInString)
        return passCheck, outputListInString, outputListInIx
    
    # Check if the first and second hold have the same op
    opOfFirstHold = holdIx_to_holdStr[int(indices[0])].split("-")[1]
    opOfSecondHold = holdIx_to_holdStr[int(indices[1])].split("-")[1]
    if opOfFirstHold == opOfSecondHold:
        passCheck = False
        if printError: print("Warning: Same op for first and second", outputListInString)
        return passCheck, outputListInString, outputListInIx
    
    # Check if the first hold start too high
    stringOffirstHold = holdIx_to_holdStr[int(indices[0])].split("-")[0]
    stringOfsecondHold = holdIx_to_holdStr[int(indices[1])].split("-")[0]
    
    # Splitting text and number in string  
    heightOfFirstHold = [re.findall(r'(\w+?)(\d+)', stringOffirstHold.split("-")[0])[0]] 
    heightOfSecondHold = [re.findall(r'(\w+?)(\d+)', stringOfsecondHold.split("-")[0])[0]] 
    if int(heightOfFirstHold[0][1]) > 6 or int(heightOfSecondHold[0][1]) > 7:
        passCheck = False
        if printError: print("Warning: Too high start", outputListInString)
        return passCheck, outputListInString, outputListInIx
    
    # Check is this already in database
    for item in handStringList:
        if set(outputListInStringSet) == set(item):
            passCheck = outputExactFromDatabase
            print("Same", item)
            return passCheck, outputListInString, outputListInIx
    # Optional: Calculate cos similarity
    if passCheck == True:
        return passCheck, outputListInString, outputListInIx
            
""" Draw a moonboard problem on the layout"""
def plotAProblem(stringList, title = None, key = None, save = None):    
    image_file = cbook.get_sample_data(parent_wd + "/raw_data/moonboard2016Background.jpg")
    plt.rcParams["figure.figsize"] = (30,10)
    img = plt.imread(image_file)
    x = []
    y = []
    for hold in stringList:
        # Using re.findall() 
        # Splitting text and number in string  
        res = [re.findall(r'(\w+?)(\d+)', hold.split("-")[0])[0]] 
        
        alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] 
        ixInXAxis = alphabateList.index(res[0][0]) 
     
        x = x + [(90 + 52 * ixInXAxis)]# * img.shape[0] / 1024]
        y = y + [(1020 - 52 * int(res[0][1]))]# * img.shape[1] / 1024]

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1, dpi = 100)
    ax.set_aspect('equal')
    plt.axis('off')

    # Show the image
    ax.imshow(img)

    # Now, loop through coord arrays, and create a circle at each x,y pair
    count = 0
    for xx,yy in zip(x,y):
        if yy == 84:
            circ = plt.Circle((xx,yy), 30, color = 'r', fill=False, linewidth = 2)
        elif count < 2:
            circ = plt.Circle((xx,yy), 30, color = 'g', fill=False, linewidth = 2)
        else:
            circ = plt.Circle((xx,yy), 30, color = 'b', fill=False, linewidth = 2)
        ax.add_patch(circ)
        count = count + 1
        
    if title:
        plt.title(title)
    if save:
        plt.savefig(key + '.jpg', dpi = 200)
    # Show the image
    plt.show()

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

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None

def moveGeneratorFromStrList (betaStringList, string_mode = True):
    """ generate the final output of move sequence as a list of dictionary.
    Input :
    ['F5-LH', 'F5-RH', 'E8-LH', 'H10-RH', 'E13-LH', 'I14-RH', 'E15-LH', 'G18-RH']
    
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
    # From List of string to hand sequence and op sequence
    handSequence = []
    handOperatorSequence = []
    xSequence = []
    ySequence = []
    for hold in betaStringList:
        characterAndNum = [re.findall(r'(\w+?)(\d+)', hold.split("-")[0])[0]] 
        handOp = hold.split("-")[1]

        alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] 
        
        handOperatorSequence.append(handOp)
        xSequence.append(alphabateList.index(characterAndNum[0][0]) )
        ySequence.append(int(characterAndNum[0][1]) - 1)

    
    outputDictionaryList = []
    numOfMoves = len(handOperatorSequence) - 2 # calculate from the third hold to the end hold (no final match)
    # loop over holds from third one to the finish hold (rank from 3 to end). In each move, this is the hold defined as target hold
    for rank in range(2,  len(handOperatorSequence)):
        # Renew a dictionary
        moveDictionary = {}
        
        # Define target hold
        targetHoldHand = handOperatorSequence[rank]
        coordinateOfTarget = (xSequence[rank], ySequence[rank]) 
        
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
        listBeforeTargetHold = handOperatorSequence[0:rank]

        remainingHoldHand = oppositehand(targetHoldHand)
        order = int(''.join(listBeforeTargetHold).rindex(remainingHoldHand)/2) # remaining hold is the last hold with opposite hand in the sequence before Target hand
        coordinateOfRemaining = (xSequence[order], ySequence[order])
    
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
        coordinateOfMoving = (xSequence[order], ySequence[order])
        
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
        for holdx, holdy in zip(xSequence, ySequence):
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

def moveGeneratorForAllGeneratedProblem(listOfGeneratedHandSequence, save_path, keyNamePre = "DeepRouteSet_v1_id", print_result = False):
    """
    Apply moveGenerator on all generated problems, and transform the output into X vector move sequence for RNN
    Input:
    - processed_data: the processed data that is scraped from MoonBoard
    - i.e. '/preprocessing/processed_data_xy_mode'  MoonBoard_2016_raw["X_dict_benchmark_withgrade"]
    
    Output:
    dictionary-- key: x_vector
    x_vector dim is 20 * numOfMoves
    20 is the embedding feature vector's dimension. In other word, every climbing move can be characterized using 20 dimension vector
    so we will have:
    vector1- vector2- vector3....last vector = move1- move2- move3... last move. This sequence will be feed into RNN as X
    """
    output = {}
    list_fail = []
    
    listOfSavedSequence = []
    countUniqueProblem = 0
    
    for oneSequence in listOfGeneratedHandSequence:# processed_data.keys():
        # create x_vector
        try:
            if oneSequence not in listOfSavedSequence:
                countUniqueProblem = countUniqueProblem + 1
                keyName = keyNamePre + str(countUniqueProblem)
                
                movesInfoList = moveGeneratorFromStrList(oneSequence, string_mode = False)
                listOfSavedSequence.append(oneSequence)

                x_vectors = np.zeros((22, len(movesInfoList)))
                for orderOfMove, moveInfoDict in enumerate(movesInfoList):   
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
                output[keyName] = x_vectors  
        except:
            print(oneSequence)
    save_pickle(output, save_path)
    print('result saved. Store ', countUniqueProblem, 'out of', len(listOfGeneratedHandSequence))
    return output, listOfSavedSequence

def coordinateToString(coordinate):
    """ convert (9.0 ,4.0) to "J5" """
    alphabateList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    return str(alphabateList[int(coordinate[0])]) + str(int(coordinate[1]) + 1)
