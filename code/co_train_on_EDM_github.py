# -*- coding: utf-8 -*-
"""
@author: Stamatis Karlos (stkarlos@upatras.gr) - Gerorge Kostopoulos - Sotiris Kotsiantis

This variant of Co-training demands Python 2.7 to run properly.
Prerequisite libraries: pandas, numpy, sklearn, xlwt
"""
import sys
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support

### imported learners
from sklearn.ensemble import ExtraTreesClassifier as EXTRA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import GradientBoostingClassifier as GraB 
from sklearn import tree
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.svm import SVC

### export final results to xls format
import xlwt

#%% write the necessary defs

def split_dataset(position, x, y):

    if  position > x.shape[1] - 1 :
        print 'wrong position id'
        return [] , []
    
    else:
        x1 = x.iloc[ : , 0:position]
        x1['class'] = y
        x2 = x.iloc[ : , position: ]
        x2['class'] = y
    return x1,x2
    
def replace_non_numeric(df,one_category):
    #replace a binary categorical attribute with 0s and 1s 

	df = df.apply(lambda sex: 0 if sex == one_category else 1)
	return df
    
def apply_classifier(x, y, upool, my_class):
    #apply the chosen classifier, depicted as my class, over the current unlabeled pool

    my_class.fit(x,y)
    results_proba = my_class.predict_proba(upool)
    results = my_class.predict(upool)
    return results_proba , results

def Mined_Instances_for_Binary_Class(x):
    # x must be a pandas.core.series.Series object

    cardinalities = x.value_counts()
    
    class1_cardinality = cardinalities.loc[0]
    class2_cardinality = cardinalities.loc[1]

    amount1 = max( int( round ( float(class1_cardinality) / class2_cardinality ) ), int( round ( float(class2_cardinality) / class1_cardinality ) ) )
    amount2 = 1
    return amount1 , amount2

#%% pre-process specific EDM datasets
    
# provide current directory which contains the specific datasets

#path = '...'
#os.chdir(path)

# provide random seeds for reproducing the experiment

# here we provide 10 seeds for execuring 10 times a 90-10 train-test scenario
rs1_list = [8, 33, 11, 45, 12345, 5867349, 3, 99, 111, 8945]
rs2_list = [123, 6, 4, 90, 566, 432, 123456, 1, 10354, 333]
rs_shuffle = 23 # control the shuffling process
rs_l = 23       # control the random_state parameter of the used classifiers
scenario = ''
iterations = input('give the desired number of iterations: (20 iterations were selected for original experiment) ')

# choice of 0 iterations would return us the supervised metrics over the initial L subset
if iterations == 0:
    scenario = 'initial'
else:
    scenario = 'final'

original_test_size = 0.1 

# initial training subsset is split to L and U subsets, such us Lpool + Upool = 1 -> 100%
upool_size_list = [0.85, 0.90, 0.975]  # it refers to the percentage size of the L subset

files = ['edu1','edu2','edu3','edu4']
for upool_size in upool_size_list:
 book = xlwt.Workbook(encoding="utf-8")
 
 for ff in files:
    df = pd.read_csv(ff + '.csv')

# convert categorical attributes to numeric - should be replaced accordingly for different datasets

    df['Final'] = replace_non_numeric(df['Final'] , 'fail')
    df['Gender'] = replace_non_numeric(df['Gender'] , 'M')
    
    if ff == 'edu1':

        df['OCS1'] = replace_non_numeric(df['OCS1'] , 'absent')
        position_of_split = 4

    elif ff == 'edu2':

        df['OCS1'] = replace_non_numeric(df['OCS1'] , 'absent')
        df['OCS2'] = replace_non_numeric(df['OCS2'] , 'absent')
        position_of_split = 6
         
    elif ff == 'edu3':

        df['OCS1'] = replace_non_numeric(df['OCS1'] , 'absent')
        df['OCS2'] = replace_non_numeric(df['OCS2'] , 'absent')
        df['OCS3'] = replace_non_numeric(df['OCS3'] , 'absent')
        position_of_split = 8
        
    else : #it stands for the case of ff == 'edu4'

        df['OCS1'] = replace_non_numeric(df['OCS1'] , 'absent')
        df['OCS2'] = replace_non_numeric(df['OCS2'] , 'absent')
        df['OCS3'] = replace_non_numeric(df['OCS3'] , 'absent')
        df['OCS4'] = replace_non_numeric(df['OCS4'] , 'absent')
        position_of_split = 10
        
    y_data = df.iloc[ : , -1] 
    x_data = df.iloc[ : , 0 : -1]

    print 'shape of data is ', x_data.shape, 'while shape of target is ', y_data.shape 

#%% Select base learners
     
    learnersX = [KNN(n_neighbors = 5) , EXTRA(n_estimators= 30, random_state = rs_l) , RF(n_estimators=30, random_state = rs_l) , GNB() , GraB(random_state = rs_l)]
    learnersY = [KNN(n_neighbors = 5) , EXTRA(n_estimators= 30, random_state = rs_l) , RF(n_estimators=30, random_state = rs_l) , GNB() , GraB(random_state = rs_l)]   

    for ww in range(0,len(rs1_list)):

       flag = False # a parameter for writing helpful headings to xls files    
       w_count = 1
       sheet1 = book.add_sheet('sheet' + str(ww) + '_' + ff)
       rs1 = rs1_list[ww]
       rs2 = rs2_list[ww]

       for w in learnersX:
        
        print ('Dataset: %s, Labeled Ratio = %d %%, Learner of two views: %s, fold: %d' %(ff, 100*(1-upool_size), str(w)[0 : str(w).find('(')], ww))
        learner1 = learner2 = learner_intermediate = w
 
        ind = StratifiedShuffleSplit(n_splits=1, test_size = original_test_size, random_state = rs1)
        for t, test in ind.split(x_data,y_data):
            pass
            
        x_test = x_data.loc[test]
        y_test = y_data.loc[test]

        x = x_data.drop(test)
        y = y_data.drop(test)

        names = list(x)

        X1 , X2 = split_dataset(position_of_split, x, y)
        x1 = X1.copy()
        x2 = X2.copy()
        
        # generate the indices for L1, L2, U1 and U2
        ind = StratifiedShuffleSplit(n_splits = 1, test_size = upool_size, random_state = rs2)
        for L_x1, U_x1 in ind.split(x1,y): 
            pass
              
        for L_x2, U_x2 in ind.split(x2,y):
            pass        

        y1 = x1['class']
        x1 = x1.drop(['class'], axis=1)
        y2 = x2['class']
        x2 = x2.drop(['class'], axis=1)

        x1_df = x1.as_matrix()
        x2_df = x2.as_matrix()
        y1_df = y1.as_matrix()
        y2_df = y2.as_matrix()
        x1_labeled   = x1.iloc[L_x1]
        x2_labeled   = x2.iloc[L_x2]
        y1_labeled   = y1.iloc[L_x1]
        y2_labeled   = y2.iloc[L_x2]
        x1_unlabeled = x1.iloc[U_x1]
        x2_unlabeled = x2.iloc[U_x2]
        y1_unlabeled = y1.iloc[U_x1]
        y2_unlabeled = y2.iloc[U_x2]

        amount1 , amount2 = Mined_Instances_for_Binary_Class(y1_labeled)
        
        median_indices = [] # hold the indices of the extracted unlabeled instances
        median = []         # hold the accuracy values of the third learner during intermediate iterations

        yL = y.iloc[L_x1]
        xL = x.iloc[L_x1]
        learner_intermediate.fit(xL, yL)
        median.append(learner_intermediate.score(x_test,y_test))
        
        for j in range(1,iterations+1):

            tt1 = []
            tt2 = []
            
            res_prob1 = np.empty(y1_unlabeled.shape[0])
            res_prob2 = np.empty(y1_unlabeled.shape[0])
            res1 = np.empty(y1_unlabeled.shape[0])
            res2 = np.empty(y1_unlabeled.shape[0])
    
            res_prob1 , res1 = apply_classifier(x1_labeled, y1_labeled, x1_unlabeled, learner1)
            res_prob2 , res2 = apply_classifier(x2_labeled, y2_labeled, x2_unlabeled, learner2)

            l_pos = []
            l_neg = []
            for i in res_prob1:
                l_pos.append(i[0] - i[1]) #fail
                l_neg.append(i[1] - i[0]) #success
            
            view1_class_0 = sorted(range(len(l_pos)), key=lambda x:l_pos[x])
            view1_class_1 = sorted(range(len(l_neg)), key=lambda x:l_neg[x])
    
            view1_s0 = view1_class_0[-1 * amount1: ]
            view1_s1 = view1_class_1[-1 * amount2: ]

            l_pos = []
            l_neg = []
            for i in res_prob2:
                l_pos.append(i[0] - i[1]) #fail
                l_neg.append(i[1] - i[0]) #success
            
            view2_class_0 = sorted(range(len(l_pos)), key=lambda x:l_pos[x])
            view2_class_1 = sorted(range(len(l_neg)), key=lambda x:l_neg[x])
    
            view2_s0 = view2_class_0[-1 * amount1: ]
            view2_s1 = view2_class_1[-1 * amount2: ]
    

            # augment the labeled instances of each learner with the most confident instances according to the other learner for both classes 
            x1_labeled = x1_labeled.append(x1_unlabeled.iloc[view2_s0 + view2_s1])
            y1_labeled = y1_labeled.append(y1_unlabeled.iloc[view2_s0 + view2_s1])
            y1_labeled.iloc[-1 * (amount1 + amount2) : ] = [0]*amount1 + [1]*amount2
    
            x2_labeled = x2_labeled.append(x2_unlabeled.iloc[view1_s0 + view1_s1])
            y2_labeled = y2_labeled.append(y2_unlabeled.iloc[view1_s0 + view1_s1])
            y2_labeled.iloc[-1 * (amount1 + amount2) : ] = [0]*amount1 + [1]*amount2 

            x1_unlabeled = x1_unlabeled.drop(x1_unlabeled.index[list(set(view1_s0 + view1_s1 + view2_s0 + view2_s1))])
            y1_unlabeled = y1_unlabeled.drop(y1_unlabeled.index[list(set(view1_s0 + view1_s1 + view2_s0 + view2_s1))])
            
            x2_unlabeled = x2_unlabeled.drop(x2_unlabeled.index[list(set(view1_s0 + view1_s1 + view2_s0 + view2_s1))])
            y2_unlabeled = y2_unlabeled.drop(y2_unlabeled.index[list(set(view1_s0 + view1_s1 + view2_s0 + view2_s1))])
    
           
            median_indices = list(y1_labeled.index) + list(y2_labeled.index)
            median_indices = list(set(median_indices)) # reduce the indices that appear two times
            
            # record the accuracy per 5 iterations
            if j % 5 == 0: 
                learner_intermediate.fit(x.loc[median_indices] , y.loc[median_indices])
                median.append(learner_intermediate.score(x_test,y_test))
    
#%% info about the experiment

        final_indices = []
        final_indices = list(y1_labeled.index) + list(y2_labeled.index)
        final_indices = list(set(final_indices))

#%% write appropriate heading
        
        cc = 0
        pos = 11
        if flag == False: 
            
            sheet1.write(0, 0, str('name of learner1'))
            sheet1.write(1, 0, str('name of learner2'))
            sheet1.write(2, 0, str('name of intermediate learner'))
            sheet1.write(4, 0, str('amount of initial L'))
            sheet1.write(5, 0, str('number of iterations'))
            sheet1.write(6, 0, str('amount of added instances'))

            for ii in range(0,len(median)):

                sheet1.write(pos + cc, 0, str('accuracy of intermediate learner during %d iteration' %(ii * 5)))
                cc = cc + 1 

            pos = pos + 2

            sheet1.write(pos + cc, 0 , str("Metrics computed from the corresponding final learner"))
            for jj in learnersY:
                
                a = str(jj)[0 : str(jj).find('(')]
                sheet1.write(pos + cc + 1, 0 , str("Acc of %s" %a))
                sheet1.write(pos + cc + 2, 0 , str("Fscore of %s" %a))
                cc += 2

            flag = True

#%% collect final results

        xAug = np.empty(len(final_indices))
        yAug = np.empty(len(final_indices))
        xAug = x.loc[final_indices]
        yAug = y.loc[final_indices]

        c = w_count
        sheet1.write(0, c, str(learner1)[0 : str(learner1).find('(')])                         # name of learner1
        sheet1.write(1, c, str(learner2)[0 : str(learner2).find('(')])                         # name of learner2
        sheet1.write(2, c, str(learner_intermediate)[0 : str(learner_intermediate).find('(')]) # name of learner_intermediate
        sheet1.write(4, c, len(L_x1))                                                          # amount of initial L
        sheet1.write(5, c, iterations)                                                         # number of iterations
        sheet1.write(6, c, len(final_indices) - len(L_x1))                                     # amount of added instances

        pos = 11
        cc = 0
        for ii in range(0,len(median)):

            sheet1.write(pos + cc, c, median[ii])
            cc = cc + 1 
        
        pos = 13

        for jj in learnersY:
            classifier_final = jj
            classifier_final.fit(xAug , yAug)

            precision , recall , fscore , method = precision_recall_fscore_support(y_test, classifier_final.predict(x_test) , average = 'weighted' , pos_label = None)
            
            sheet1.write(pos + cc + 1, c , classifier_final.score(x_test,y_test))
            sheet1.write(pos + cc + 2, c , float(fscore.mean()))
            cc += 2
        
        w_count += 1
 
 book.save("Co_train_educational_" + scenario + "_" + str(upool_size) + ".xls")