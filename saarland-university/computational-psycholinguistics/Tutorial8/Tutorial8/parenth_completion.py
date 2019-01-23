'''
Created on Jul 13, 2017

@author: jesus
'''
#testString="(TOP (S (NP (DT The) (NN man) (NP/NP (VP (VBN held) (PP (IN at) (NP (DT the) (NN hospital))))\n"
import sys

def addClosingPars(inputString):
    openning= inputString.count("(")
    closing= inputString.count(")")
    missing=openning-closing
    if missing==0:
         return 0
    else:
         return inputString.strip()+")"*missing+"\n"


def getNewFile(filenameInput,filenameOutput):
    FILEIN=open(filenameInput,'r')
    
    with open(filenameOutput,'w') as outputFile:
        line = FILEIN.readline()
        while line:
            addedClose=addClosingPars(line)
            if addedClose==0:
                outputFile.write(line)
                print line
            else:
                outputFile.write(addedClose)
                        
            line = FILEIN.readline()
    FILEIN.close()
    
print len(sys.argv)
if len(sys.argv)==3:
    fileIn=sys.argv[1]
    fileOut=sys.argv[2]
     
print fileIn
print fileOut
getNewFile(fileIn,fileOut)
