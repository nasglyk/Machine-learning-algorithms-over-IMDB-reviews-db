from pickle import FALSE
import tensorflow as tf
from tensorflow import keras
import math

# ID3

#----------------------------
#diavasma tou arxeiou
num_of_trees = 0
number_of_words = input("Please enter the number of words:\n")
skip_top_number_of_words = input("Please enter the number of top words that you want to skip:\n")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=int(number_of_words)+1,
    skip_top=int(skip_top_number_of_words)+1, # svisame ena +1
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=int(number_of_words)+int(skip_top_number_of_words)+1,
    index_from=0
)

word_index = keras.datasets.imdb.get_word_index()


x=[0 for i in range (len(x_train))]
   


voc={'del': 0}
voc = dict(sorted(word_index.items(), key=lambda item: item[1]))

# dimiourgia toy leksilogiou
d = dict(list(voc.items())[int(skip_top_number_of_words):int(number_of_words)+int(skip_top_number_of_words)])
#print(d)
#print ( d["the"])


#---------- dimiourgoume ton pinaka x gia to x_train
for y in range(len(x_train)):
    x[y]=[0 for i in range (int(number_of_words)-int(skip_top_number_of_words))]
#print(x[0])



inverted_word_index = dict((i, word) for (word, i) in word_index.items())
decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])

for n in range(len(x)):
  
    for y in x_train[n]:
        
       
        if y!= int(number_of_words)+int(skip_top_number_of_words)+1:
            x[n][y-int(skip_top_number_of_words)-1] =1




#------------- dimiourgoume ton pinaka x2 gia to x_test    
x2=[0 for i in range (len(x_test))]        
for y in range(len(x_test)):
    x2[y]=[0 for i in range (int(number_of_words))]       
  
for n in range(len(x2)):
  
    for y in x_test[n]:
        
       
        if y!= int(number_of_words)+int(skip_top_number_of_words)+1:
            x2[n][y-int(skip_top_number_of_words)-1] =1 
#------------------------

x3=[]
y_train2=[]
for i in x:
    x3.append(i)
for y in y_train:
    y_train2.append(y)





#-----------------------------------
#sinartisi pou upologizei thn entropia 2 metavliton
def twoCEntropy( cProb):
    #print(cProb)
    if(cProb ==0 or cProb==1):
        return 0.0
    else:
        return -(cProb * math.log2(cProb)) - ((1.0 - cProb)* math.log2(1.0 - cProb))
    


#-----------------------------------
#sinartisi pou upologizei enan pinaka me ta information gain ton idiotiton mas
def calculateIG(table , train):
    numOfExamples = len(table)
    numOfFeatures = len(table[1]) 
    IG = [0 for x in range(numOfFeatures)]

    positives =0
    #count how many are c=1
    for i in train:
        if i == 1 :
            positives = positives +1 
            #print(positives)
    PC1 = positives/ numOfExamples
    HC = twoCEntropy(PC1)

    PX1 = [0 for x in range(numOfFeatures)]
    PC1X1 = [0 for x in range(numOfFeatures)]
    PC1X0 = [0 for x in range(numOfFeatures)]
    HCX1 = [0 for x in range(numOfFeatures)]
    HCX0 = [0 for x in range(numOfFeatures)]

    for i in range (numOfFeatures):

        cX1=0
        cC1X1=0
        cC1X0 = 0
        for j in range (numOfExamples):
            if (table[j][i]== 1) :
                cX1 = cX1 +1
            if (table[j][i] ==1 and train[j] == 1 ):
                cC1X1 = cC1X1 +1
            if (table[j][i] ==0 and train[j] == 1 ):
                cC1X0 = cC1X0 +1   
        PX1[i]= cX1/ numOfExamples
        if ( cX1 ==0 ):
            PC1X1[i] =0.0
        else:
            PC1X1[i]=cC1X1/cX1

        if(cX1 == numOfExamples) :
            PC1X0[i] = 0.0
        else: 
            PC1X0[i] =  cC1X0 / (numOfExamples - cX1) 

        HCX1[i] = twoCEntropy(PC1X1[i])
	    #HCX0[i] = twoCEntropy(PC1X0[i])  
        HCX0[i]= twoCEntropy(PC1X0[i]) 

        IG[i] = HC - ( (PX1[i] * HCX1[i]) + ( (1.0 - PX1[i]) * HCX0[i]) )   

        

    return IG


IG2 = [0 for x in range(len(x[1]))]

#pinakas pou elegxoume poia feature exoun xrisimopoihthei
xrisimopoihmenes_theseis=[0 for x in range(len(x[1]))]

most_common=[0 for x in range(len(x[1]))]

#------------------------------------------
# vriskoume to most informative feature 
def most_informative_feature(train_data,y_train):
    max_info_gain=-1
    max_info_feature= None
    IG2=calculateIG(train_data,y_train)
    
    
    for i in range(len(IG2)):
        
        if (IG2[i]>= max_info_gain and xrisimopoihmenes_theseis[i]==0) :
            max_info_gain=IG2[i]
            max_info_feature = i # h thesh thw leksis me to megalytero IG 
            
    xrisimopoihmenes_theseis[max_info_feature]=1       
    return max_info_feature    


def make_sub_tree(feature_name, train_data, y_train,most_common):
    count_1=0
    count_0=0
    #ypologizoume plithos twn reviews pou exoun to sygkekrimeno feature kai aytwn pou den to exoun 
    for i in train_data:
        
        if i[feature_name]==1:
            count_1=count_1+1
        if i[feature_name]==0:
            count_0=count_0+1

    tree={}  

    pinakas1=[]
    apotelesmata_p1=[]
    pinakas2=[]
    apotelesmata_p2=[]
    #ksexorizoume poia reviews exoun to feature kai poia oxi 
    for i in range(len(train_data)):
        if train_data[i][feature_name]==1:
            pinakas1.append(train_data[i])
            apotelesmata_p1.append(y_train[i])
        if train_data[i][feature_name]==0:
            pinakas2.append(train_data[i]) 
            apotelesmata_p2.append(y_train[i])  
    assigned_to_node_class0= False
    assigned_to_node_class1= False

    class0_count0=0
    class0_count1=0
    class1_count0=0
    class1_count1=0

    for j in apotelesmata_p1:
        if j==1:
            class0_count1=class0_count1+1
        if j==0:
            class0_count0=class0_count0+1
    for j in apotelesmata_p2:
        if j==1:
            class1_count1=class1_count1+1
        if j==0:
            class1_count0=class1_count0+1

   
    count_y0=0
    count_y1=0


    for i in y_train:
        if i==0:
            count_y0=count_y0+1
        if i==1:
            count_y1=count_y1+1
    if count_y1>count_y0:
        for key in d:
            if d[key] -int(skip_top_number_of_words)== feature_name+1 :
                most_common[d[key]-1-int(skip_top_number_of_words)] = 1
    else:
        for key in d:
            if d[key] -int(skip_top_number_of_words)== feature_name+1 :
                most_common[d[key]-1-int(skip_top_number_of_words)] = 0        
        


    if count_1!=0:
        if class0_count0/count_1>=0.7 and count_1!=0: #diwxnw oles tiw grammes pou exoun 1 se ayto to feature
            for key in d:
                if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                    tree[key] = 0
                
            for g in range(len(train_data)):
                if train_data[g]!='no':
                    if train_data[g][feature_name]==1:
                        train_data[g]='no'
                        
                        y_train[g]=2


            assigned_to_node_class0= True
        
        if class0_count1/count_1>=0.7 and count_1!=0: #diwxnw oles tiw grammes pou exoun 1 se ayto to feature
            for key in d:
                if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                    
                    tree[key] = 1
                
            for g in range(len(train_data)):
                if train_data[g]!='no':
                    if train_data[g][feature_name]==1:
                        train_data[g]='no'
                        y_train[g]=2
        

            assigned_to_node_class0= True
    if count_0!=0:
        if class1_count0/count_0>=0.7 and count_0!=0: #diwxnw oles tiw grammes pou exoun 0 se ayto to feature
            for key in d:
                if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                    tree['not ' + key] = 0
            
            for g in range(len(train_data)):
                if train_data[g]!='no':
                    if train_data[g][feature_name]==0:
                        train_data[g]='no'
                        y_train[g]=2
        
            assigned_to_node_class1= True
    
        if class1_count1/count_0>=0.7 and count_0!=0: #diwxnw oles tiw grammes pou exoun 0 se ayto to feature
            for key in d:
                if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                    tree[ 'not ' + key] = 1
                
        
            for g in range(len(train_data)):
                if train_data[g]!='no':
                    if train_data[g][feature_name]==0:
                        train_data[g]='no'
                        y_train[g]=2

            assigned_to_node_class1= True

    if not assigned_to_node_class1:
        for key in d:
            if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                tree['not ' + key] = "?"
                
            
    if not assigned_to_node_class0:
        for key in d:
            if d[key]-int(skip_top_number_of_words) == feature_name+1 :
                tree[ key] = "?"
        

    return tree,train_data,y_train


dokimi=[[1, 0, 1, 1, 0, 1, 0, 0, 1, 1],[1, 1, 0, 0, 0, 1, 0, 0, 1, 1],[1, 1, 0, 1, 1, 1, 0, 0, 1, 1],[1, 0, 0, 0, 0, 1, 0, 0, 1, 1]]
dokimi2=[0,0,1,1]



def generate_tree(root, prev_feature_value, train_data, y_train,most_common):
    
    alitheia=False
    for i in xrisimopoihmenes_theseis:
        if i==0:
            alitheia=True
    if len(train_data)!=0 and alitheia==True :
        max_info_feature = most_informative_feature(train_data,y_train)
        tree, train_data, y_train = make_sub_tree(max_info_feature, train_data,y_train,most_common)
        
        next_root= None

        if prev_feature_value != None:
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature]= tree
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature]=tree
            next_root= root[max_info_feature]

        for node, branch in list(next_root.items()):
            real_node = node
            
            if len(node)>=4:
                if node[0]+node[1]+node[2]+node[3]=='not ':
                    node=0
            else:
                node=1
            
           
            if branch == "?" : 
                
                feature_value_data=[]
                feature_value_data_result=[]
                
                for item in range(len(train_data)):
                    
                    if train_data[item]!='no':
                        if train_data[item][max_info_feature]==node:
                            
                            feature_value_data.append(train_data[item])
                            feature_value_data_result.append(y_train[item])
                
                generate_tree(next_root,real_node, feature_value_data,feature_value_data_result,most_common)



def id3(train_data_m,y_train,most_common):
    num_of_trees = 0
    train_data=[]
    for i in train_data_m:
        train_data.append(i)
    
    tree={}
    generate_tree(tree,None,train_data_m,y_train,most_common)
    return tree

tree3=id3(x,y_train,most_common)


#katigoriopoisi enos review os thetiko h arnhtiko
def decision(tree, instance,timi):
    if not isinstance(tree, dict): #an einai fyllo 
        return tree,timi #epistrepse thn timh
    elif (int(number_of_words) - int(skip_top_number_of_words) != 0 ) : 
        
        root_node = next(iter(tree)) #pare to prwto
        feature_value = instance[root_node] #value tou  feature
        timi=''
        for key in d:
            if d[key]-int(skip_top_number_of_words) == root_node + 1:

                timi=key
                    
                if feature_value==0:
                    timi='not '+key
        
        if timi in tree[root_node]: 
            return decision(tree[root_node][timi], instance,timi) #goto next feature
        else:
            return None,None
    else:
        return None,None       


#ipologismos pososton kai metriseon 
def evaluate(tree,test_data_m, y_test):
    TP=0
    FP=0
    FN=0
    correct_predict=0
    wrong_predict=0
    precision=0
    recall=0
    F1=0
    for i in range(len(test_data_m)):
        
        result,timi= decision(tree,test_data_m[i],0)
        #an to fyllo toy dentroy einai ? apofasizoume an einai thetiko h arntitiko me vash tou an ta perissotera review einai thetika h arnhtika sta reviews pou antistixousan se ayto to fyllo
        if result=='?':
            if len(timi)>=4:
                if timi[0]+timi[1]+timi[2]+timi[3]=='not ':
                   s=timi.split() 
                   timi=s[1]
            result=most_common[d[timi]-1-int(skip_top_number_of_words)]
           
        if result == y_test[i] :
            correct_predict=correct_predict+1
        else:
            wrong_predict=wrong_predict+1
            
        if result==1 and y_test[i]==1:
            
            TP=TP+1
        if result==1 and y_test[i]==0:
            FP=FP+1
        if result==0 and y_test[i]==1:
            FN=FN+1    
             
    accuracy= correct_predict / (correct_predict + wrong_predict)
    if (TP+FP)!=0:
        precision=TP/(TP+FP)
    if (TP+FN)!=0:
        recall=TP/(TP+FN)
    if precision+recall!=0:
        F1=(2*precision*recall)/(precision+recall)
    
    return accuracy,precision ,recall,F1


accuracy1,precision1,recall1,F1_1= evaluate(tree3,x2,y_test)
print("accuracy test",accuracy1)

accuracy2,precision2,recall2,F1_2= evaluate(tree3,x3,y_train2)
print("accuracy train",accuracy2)
print("precision train",precision2)
print("recall train",recall2)
print("F1 train",F1_2)

