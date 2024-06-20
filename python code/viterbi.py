import csv
import numpy as np
import sys
import time
import copy
from IPython.display import clear_output

#Setting up the dictionary
posDictionary = {}
posDictionary['ADJ'] = 0
posDictionary['ADP'] = 1
posDictionary['ADV'] = 2
posDictionary['AUX'] = 3
posDictionary['CCONJ'] = 4
posDictionary['DET'] = 5
posDictionary['INTJ'] = 6
posDictionary['NOUN'] = 7
posDictionary['NUM'] = 8
posDictionary['PART'] = 9
posDictionary['PRON'] = 10
posDictionary['PROPN'] = 11
posDictionary['PUNCT'] = 12
posDictionary['SCONJ'] = 13
posDictionary['SYM'] = 14
posDictionary['VERB'] = 15
posDictionary['X'] = 16

reverseDictionary = {}
reverseDictionary['0'] = 'ADJ'
reverseDictionary['1'] = 'ADP'
reverseDictionary['2'] = 'ADV'
reverseDictionary['3'] = 'AUX'
reverseDictionary['4'] = 'CCONJ'
reverseDictionary['5'] = 'DET'
reverseDictionary['6'] = 'INTJ'
reverseDictionary['7'] = 'NOUN'
reverseDictionary['8'] = 'NUM'
reverseDictionary['9'] = 'PART'
reverseDictionary['10'] = 'PRON'
reverseDictionary['11'] = 'PROPN'
reverseDictionary['12'] = 'PUNCT'
reverseDictionary['13'] = 'SCONJ'
reverseDictionary['14'] = 'SYM'
reverseDictionary['15'] = 'VERB'
reverseDictionary['16'] = 'X'

#Opening the training set file 
with open("ud21_for_POS_TAGGING-180325-train.txt", encoding="utf-8") as f:
	reader = csv.reader(f, delimiter="\t", quotechar=None)
	wordsArray = list(reader)

#Creating the Part of Speech list: dimension=367760x1
posList=[]
for i in range(0,len(wordsArray)):
	posList.append(wordsArray[i][1])

#Creating the Part of Speech set: dimension=17x1
posSet = sorted(set(posList))

#Creating:
#          1: the array containing P(t_0 | start)
#          2: the array containing P(end | t_last)

tagStart = np.zeros(len(posSet))
totalPunct = 0
for i in range(1, len(wordsArray)-1):
	#Everytime we find a PUNCT we consider the next tag(wordsArray[i+1][1])
	if(wordsArray[i][0] == '.'):
		totalPunct += 1
		tagStart[posDictionary[wordsArray[i+1][1]]] += 1

for i in range(0, len(tagStart)):
	tagStart[i] /= totalPunct-1    

#Creating:
#          1: the matrix that count C(t_{i}, t_{i-1}): dimension=17x17
#          2: the array that count C(t_{i}): dimension=17x1
print("PoS tag set is: ")
print(posSet)
tagCount = np.zeros(len(posSet))
tagConjCount = np.zeros((len(posSet), len(posSet)))
tagCount[posDictionary[posList[0]]] +=1

for i in range(1, len(posList)):
	currentTag = posList[i]
	prevTag = posList[i-1]
	
	tagCount[posDictionary[currentTag]] +=1
	tagConjCount[posDictionary[currentTag], posDictionary[prevTag]] +=1 

#Creating array containing the conditional probability
tagCondProb = np.zeros((len(posSet), len(posSet)))
for i in range(0, len(posSet)):
	for j in range(0, len(posSet)):
		tagCondProb[i][j] = tagConjCount[i][j]/tagCount[i]

#Building wordStructure to make efficient the buildEmission function
wordsArrayCopy = np.array(copy.copy(wordsArray))
for i in range(0, len(wordsArrayCopy)):
	wordsArrayCopy[i][0] = wordsArrayCopy[i][0].lower()

wordSet = sorted(set(wordsArrayCopy[:,0]))
wordStructure = np.zeros((len(wordSet), len(posSet)))

for i in range(0, len(wordsArrayCopy)):
	if(i%10000==0):
		clear_output()
		print(str(i)+"/"+str(len(wordsArrayCopy)))
	position = wordSet.index(wordsArrayCopy[i][0])
	wordStructure[position][posDictionary[wordsArrayCopy[i][1]]] += 1
	
clear_output()
print(str(len(wordsArrayCopy))+"/"+str(len(wordsArrayCopy)))

#Normalizing
for i in range(0,len(wordSet)):
	occur = 0
	for j in range(0, len(posSet)):
		occur += wordStructure[i][j]
	wordStructure[i] = wordStructure[i] / occur

#Building emission probability matrix
def buildEmission(phrase, posSet):
	emissionMatrix = np.zeros((len(phrase), len(posSet)))
	#for every word in the training set
	for j in range(0, len(phrase)):
		if(phrase[j].lower() in wordSet):
			currentWordPos = wordSet.index(phrase[j].lower())
			emissionMatrix[j] = wordStructure[currentWordPos]
		else:
			emissionMatrix[j] = 1/len(posSet)
	return emissionMatrix

#print(buildEmission(['È', "la", "spada", "laser", "di", "tuo", "padre"], posSet))

#Defining Viterbi Algorithm
def viterbiAlgorithm(phrase, posSet):
	viterbiMatrix = np.zeros((len(posSet), len(phrase)))
	backpointerMatrix = np.zeros((len(posSet), len(phrase)))
	emissionMatrix = buildEmission(phrase, posSet)
	
	#Initialization step
	for i in range(0, len(posSet)):
		viterbiMatrix[i][0] = tagStart[i] * emissionMatrix[0][i]
		  
	#Recursion step
	for i in range(1, len(phrase)):
		for j in range(0, len(posSet)):
			maxProb = -1
			maxIndex = -1
			for k in range(0, len(posSet)):
				currentProb = viterbiMatrix[k][i-1] * tagCondProb[j][k] * emissionMatrix[i][j]
				
				if(currentProb>maxProb):
					maxProb = currentProb
					maxIndex = k
			viterbiMatrix[j][i]=maxProb
			backpointerMatrix[j][i] = maxIndex
		  
	return [viterbiMatrix, backpointerMatrix]

def viterbi(phrases, posSet):
	phrasesArray =[]
	for p in range(0, len(phrases)):
		if(p % 100 == 0):
			clear_output()
			print(p)
		
		viterbiResults = viterbiAlgorithm(phrases[p], posSet)

		#Getting the max probability index from Viterbi Matrix
		maxIndexProb = np.argmax(viterbiResults[0][:,len(phrases[p])-1])
		
		phraseClassArray = []
		
		phraseClassArray.append(reverseDictionary[str(maxIndexProb)])
		currentBackpointerIndex = int(viterbiResults[1][maxIndexProb][len(phrases[p])-1])

		#Getting the PoS sequence
		for i in range(len(phrases[p])-2, -1, -1):
			phraseClassArray.insert(0, reverseDictionary[str(currentBackpointerIndex)])
			currentBackpointerIndex = int(viterbiResults[1][currentBackpointerIndex][i])
			
		phrasesArray.append(phraseClassArray)
	
	return phrasesArray

#Testing the algorithm on the three sample phrase given by the teacher
phrasesSample = [['È', "la", "spada", "laser", "di", "tuo", "padre"], ["Ha", "fatto", "una", "mossa", "leale"], ["Gli", "ultimi", "avanzi", "della", "vecchia", "Repubblica", "sono", "stati", "spazzati", "via"]] 

start_time_total = time.time()
resSample = viterbi(phrasesSample, posSet)
end_time_total = time.time()
print("TOTAL time: {}".format(end_time_total - start_time_total))

print(resSample)

#ACCURACY
#Opening the test set file 
with open("ud21_for_POS_TAGGING-180325-test.txt", encoding="utf-8") as f:
	reader = csv.reader(f, delimiter="\t", quotechar=None)
	wordsArrayTest = list(reader)

#Creating the Part of Speech list: dimension=20254x1
posListTest=[]
for i in range(0,len(wordsArrayTest)):
	posListTest.append(wordsArrayTest[i][1])

phrases=[]
tagsList = []
currentPhrase = []
currentTagList = []
for i in range(0, len(wordsArrayTest)):
	#Everytime we find a '.' we consider the current phrase as ended
	if(wordsArrayTest[i][0] == '.'):
		currentPhrase.append(wordsArrayTest[i][0])
		currentTagList.append(posListTest[i])
		phrases.append(currentPhrase)
		tagsList.append(currentTagList)
		currentPhrase =[]
		currentTagList=[]
	else:
		currentPhrase.append(wordsArrayTest[i][0])
		currentTagList.append(posListTest[i])

#Number of phrases in the test set
print("There are " + str(len(phrases)) + " in the test set")

#Executing Viterbi algorithm over the test set
start_time_long = time.time()
res = viterbi(phrases, posSet)
end_time_long = time.time()
print("TOTAL time: {}".format(end_time_long - start_time_long))

#Accuracy calculation
correct = 0
total = 0
for i in range(0, len(res)):
	for j in range(0, len(res[i])):
		if(res[i][j] == tagsList[i][j]):
			correct += 1
		total +=1
print("Correct is "+ str(correct))
print("Total is "+str(total))
print("Accuracy is " + str(correct/total))

#Creating the array that contains the most frequent tag for every word
wordsMostFreqUse = []
for i in range(0, len(wordSet)):
	if(i%500==0):
		clear_output()
		print(str(i)+"/"+str(len(wordSet)))
		
	wordsMostFreqUse.append([wordSet[i], np.argmax(wordStructure[i])])

clear_output()
print(str(len(wordSet))+"/"+str(len(wordSet)))

#BASELINE ACCURACY
baselineRes = []
for i in range(0, len(wordsArrayTest)):
	currentWord = wordsArrayTest[i][0].lower()
	
	if(currentWord in wordSet):
		currentWordPos = wordSet.index(currentWord)
		baselineRes.append(wordsMostFreqUse[currentWordPos][1])
	else:
		#If the current word never occured in training set, we'll consider it as NOUN
		baselineRes.append(7)

#Baseline accuracy calculation
correct = 0
for i in range(0, len(baselineRes)):
	if(baselineRes[i]==posDictionary[wordsArrayTest[i][1]]):
		correct += 1
print("Baseline Accuracy is " + str(correct/len(baselineRes)))





#-------------------------------------#
#         DIRECT TRANSLATION          #
#-------------------------------------#

itaEngDict = {}
itaEngDict['È0'] = "is"
itaEngDict['È1'] = "'s"
itaEngDict['la0'] = 'of'
itaEngDict['la1'] = 'it'
itaEngDict['spada0'] = 'saber'
itaEngDict['laser0'] = 'light'
itaEngDict['di0'] = 'of'
itaEngDict['di1'] = "'s"
itaEngDict['tuo0'] = 'your'
itaEngDict['padre0'] = 'father'

itaEngDict['Ha0'] = 'he'
itaEngDict['fatto0'] = 'made'
itaEngDict['una0'] = 'a'
itaEngDict['mossa0'] = 'move'
itaEngDict['leale0'] = 'fair'

itaEngDict['Gli0'] = 'the'
itaEngDict['ultimi0'] = 'last'
itaEngDict['avanzi0'] = 'remnants'
itaEngDict['della0'] = 'of'
itaEngDict['vecchia0'] = 'old'
itaEngDict['Repubblica0'] = 'republic'
itaEngDict['sono0'] = 'have'
itaEngDict['stati0'] = 'been'
itaEngDict['spazzati0'] = 'swept'
itaEngDict['via0'] = 'away'


#PoS rules
def invertingPosRules(phrase, posTagArray, i, translated):
	#Handle Saxon genitive
	if((i<len(phrase)-2) & (i>0)):
		if((posTagArray[i]=='ADP') & (posTagArray[i+1]=='DET') & (posTagArray[i+2]=='NOUN')):
			j=i-1
			while(posTagArray[j] == 'NOUN'): 
				j-=1
			translated.insert(j+1, itaEngDict[phrase[i+1]+'0'])
			translated.insert(j+2, itaEngDict[phrase[i+2]+'0'])
			translated.insert(j+3, itaEngDict[phrase[i]+'1'])
			return 3
		
	# Handle subject's Saxon genitive("È la" --> "It's")  
	if(i<len(phrase)-1):
		if((posTagArray[i]=='AUX') & (posTagArray[i+1]=='DET')):
			translated.append(itaEngDict[phrase[i+1]+'1'])
			translated.append(itaEngDict[phrase[i]+'1'])
			return 2
	
	#Invert noun and adjective("mossa leale" --> "fair move")
	if(i<len(phrase)-1):
		if((posTagArray[i]=='NOUN') & (posTagArray[i+1]=='ADJ')):
			translated.append(itaEngDict[phrase[i+1]+'0'])
			translated.append(itaEngDict[phrase[i]+'0'])
			return 2
	translated.append(itaEngDict[phrase[i]+'0'])
	return 1

#DIRECT TRANSLATION
def translatePhrase(phrase, posTagArray):
	translated = []
	i=0
	while(i<len(phrase)):
		incr = invertingPosRules(phrase, posTagArray, i, translated)
		i+= incr
	return translated      

##Testing the translation on the three sample phrase given by the teacher 
phrases2Translate = [ [phrasesSample[0], resSample[0]], [phrasesSample[1], resSample[1]], [ phrasesSample[2], resSample[2] ] ]
print(phrases2Translate)

#Translate!
for i in range(0, len(phrases2Translate)):
    translated = translatePhrase(phrases2Translate[i][0], phrases2Translate[i][1])
    for j in range(0, len(translated)):
        sys.stdout.write(translated[j]+' ')
        sys.stdout.flush()
    print()