import os,math
import csv
import argparse

#Naive Bayes Implementation for Spam-Ham
def naiveBayes(test_file,train_file,output_file):

    groundTruth = {}
    spamDict = {}
    hamDict = {}

    for line in train_file:
        line_content = line[0].split(" ")
        groundTruth[line_content[0]] = line_content[1]

        if line_content[1] == "ham":
            for i in range(2, len(line_content) - 1, 2):
                if line_content[i] in hamDict.keys():
                    hamDict[line_content[i]] = float(hamDict[line_content[i]]) + float(line_content[i + 1])
                else:
                    hamDict[line_content[i]] = float(line_content[i + 1])

        elif line_content[1] == "spam":
                for i in range(2, len(line_content) - 1, 2):
                    if line_content[i] in spamDict.keys():
                        spamDict[line_content[i]] = float(spamDict[line_content[i]]) + float(line_content[i + 1])
                    else:
                        spamDict[line_content[i]] = float(line_content[i + 1])

    hamwords_count=sum(hamDict.values())  #No. of words in all ham emails
    spamword_count=sum(spamDict.values()) #No. of words in all emails

    probhamdict={}
    probspamdict={}

    for k,v in hamDict.items():
        probhamdict[k]=math.log10(v)-math.log10(hamwords_count) #probability of each word in all ham emails

    for k,v in spamDict.items():
        probspamdict[k] = math.log10(v)-math.log10(spamword_count) #probability of each word in all spam emails

    spamCount=0
    hamCount=0

    #Find number of spam emails and ham emails
    for hs in groundTruth.values():
        if hs=='spam':
            spamCount+=1
        elif hs=='ham':
            hamCount+=1

    # Calculation of prior probabilities
    prior_spam =  float(float(spamCount) / float((spamCount+hamCount)))
    prior_ham = float(float(hamCount) / float((spamCount + hamCount)))

    #Model is ready. Now, onto Test Data
    predicted = {}
    test_emails={}

    for line in test_file:
        line_content = line[0].split(" ")
        test_emails[line_content[0]] = line_content[1]

        probability_ham=0.0
        probability_spam = 0.0

        #Summation in Bayes Formula
        for i in range(2, len(line_content) - 1, 2):
            if line_content[i] in probhamdict:
                probability_ham += probhamdict[line_content[i]]
            else:
                probability_ham += math.log10(float(float(1.0)/float(len(probhamdict))))
        probability_ham=math.log10(prior_ham )+probability_ham

        for i in range(2, len(line_content) - 1, 2):
            if line_content[i] in probspamdict:
                probability_spam += probspamdict[line_content[i]]
            else:
                probability_spam += math.log10(float(float(1.0) /float(len(probspamdict))))
        probability_spam = math.log10(prior_spam) + probability_spam

        if probability_ham > probability_spam:
            predicted[line_content[0]] = "ham"
        else:
            predicted[line_content[0]] = "spam"

    #Checking for Test Accuracy
    test_accuracy = 0
    for k,v in test_emails.items():
        if test_emails[k] == predicted[k]:
            test_accuracy+=1

    print (test_accuracy/float(len(test_emails)))*100

    with open(output_file, 'w') as file:
        fieldnames = ['email_id', 'class']
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=',', lineterminator='\n')

        for k,v in predicted.items():
            writer.writerow({'email_id': k, 'class': v})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", type=str, help="path of training files")
    parser.add_argument("-f2", type=str, help="path of testing files")
    parser.add_argument("-o", type=str, help="path of output file")

    args = parser.parse_args()
    training_data_filepath = args.f1
    test_data_filepath = args.f2
    output_filepath = args.o

    train_file = csv.reader(open(training_data_filepath, 'r'))
    test_file = csv.reader(open(test_data_filepath, 'r'))

    naiveBayes(test_file,train_file,output_filepath)