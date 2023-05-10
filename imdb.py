import tensorflow as tf
from tensorflow import keras


# Bayes


#------------------------------------------
#  diavasma toy arxeioy 



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

#-------------to leksilogio mas
d = dict(list(voc.items())[int(skip_top_number_of_words):int(number_of_words)+int(skip_top_number_of_words)])



# ------------ gia to x_train dimioyrgoyme ton pinaka x
for y in range(len(x_train)):
    x[y]=[0 for i in range (int(number_of_words))]


for n in range(len(x)):
  
    for y in x_train[n]:
        
       
        if y!= int(number_of_words)+int(skip_top_number_of_words)+1:
            x[n][y-int(skip_top_number_of_words)-1] =1 
            


#------------- gia to x_test dimiourgoume ton pinaka x2    
x2=[0 for i in range (len(x_test))]        
for y in range(len(x_test)):
    x2[y]=[0 for i in range (int(number_of_words))]       
  
for n in range(len(x2)):
  
    for y in x_test[n]:
        
       
        if y!= int(number_of_words)+int(skip_top_number_of_words)+1:
            x2[n][y-int(skip_top_number_of_words)-1] =1 
#------------------------



# dhmioyrgoyme th sinartisi poy ypologizei tis pithanotites gia an iparxei h den iparxei h leksh kai an einai thetiko h arnhtiko review
def find_possibility(x,y_train):
    possibility=[0 for i in range (len(d))]


    for y in range(len(possibility)):
        possibility[y]=[0 for i in range (4)]



    yes_exists = 0
    yes_den_exists =0
    no_exists=0
    no_den_exists =0
    sumyes=0
    sumno=0

    for i in range(len(possibility)):
    
        for y in range(len(y_train)):
            if y_train[y]==1: 
                if x[y][i]==1:
                    yes_exists += 1
                
                else:
                    yes_den_exists +=1    
            if y_train[y]==0:
                if x[y][i]==1:
                    no_exists +=1
                
                else:
                    no_den_exists +=1

        sumyes=yes_exists + yes_den_exists
        sumno = no_den_exists + no_exists
        possibility[i][0]= (yes_exists +1 )/ (sumyes +2)
        possibility[i][1]= (yes_den_exists +1) / (sumyes +2)
        possibility[i][2]= (no_exists +1) / (sumno +2)
        possibility[i][3]= (no_den_exists +1) / (sumno + 2)
        yes_exists = 0
        yes_den_exists =0
        no_exists=0
        no_den_exists =0
        sumyes=0
        sumno=0
    return possibility
        

#--------------------------------------
# ypologizoyme tis pithanotites ena review na einai thetiko h arnhtiko 
def posibility_of_pos_neg(y_train):
    count_yes=0
    count_no=0
    for j in y_train:
        if j==1:
            count_yes=count_yes+1
        if j==0:
            count_no=count_no+1

    p_yes=count_yes/(count_yes+count_no)
    p_no=count_no/(count_yes+count_no)
    return p_yes,p_no



#----------------------------------
#ypologizoyme tis telikew pithanotitew kai katatasoume ta review se katigories ipologizontas kai ta pososta epitixias
def results(x,y_train,possibility):
    

    apotelesmata2=[0 for i in range (len(x))]
    p_yes,p_no=posibility_of_pos_neg(y_train)
    n=0
    for y in x:

        p_yes_y= 1
        p_no_y=1
        for i in range(len(y)):
            if y[i]==1:
                p_yes_y=p_yes_y * possibility[i][0]
            if y[i]==0:
                p_yes_y=p_yes_y * possibility[i][1]

        for i in range(len(y)):
            if y[i]==1:
                p_no_y=p_no_y * possibility[i][2]
            if y[i]==0:
                p_no_y=p_no_y * possibility[i][3]
        p_yes_y=p_yes_y * p_yes
        p_no_y = p_no_y * p_no
        if (p_yes_y > p_no_y):
            apotelesmata2[n]=1
        else: 
            apotelesmata2[n]=0
        n=n+1
    
    count_correct=0
    count_all=0
    TP=0
    FP=0
    FN=0
    for i in range(len(x)):
    
        count_all=count_all+1
        if (apotelesmata2[i]==y_train[i]):
            count_correct=count_correct+1
        if apotelesmata2[i]==1 and y_train[i]==1:
            TP=TP+1
        if apotelesmata2[i]==1 and y_train[i]==0:
            FP=FP+1
        if apotelesmata2[i]==0 and y_train[i]==1:
            FN=FN+1

    





    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=(2*precision*recall)/(precision+recall)
    return count_correct/count_all,precision,recall,F1


#  ypologizoume tis pithanotites gia an iparxei h den iparxei kathe leksh eno einai thetiko h arnhtiko review
possibility=find_possibility(x,y_train)
# ypologizoume ta telika apotelesmata gia to test
accuracy_test,precision_test,recall_test,F1_test=results(x2,y_test,possibility)
print("test acc",accuracy_test)
#ypologizoume ta telika apotelesmata gia to train
accuracy_train,precision_train,recall_train,F1_train=results(x,y_train,possibility)
print("train acc",accuracy_train)
print("train prec",precision_train)
print("train recall",recall_train)
print("train F1",F1_train)

