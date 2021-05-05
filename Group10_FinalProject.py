# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:21:16 2021

@author: lkl444
"""

def PlotData(dataset,x_names,y_names,x_label,y_label,figtitle,fignum=1):
    #Create Subplots
    #Import necessary packages
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Create label for the x
    xlabel = ['','','','','']+x_label
    #Create label for the y
    ylabel = [y_label[0]]+['','','','']+[y_label[1]]+['','','','']
    #Define what input data you're looking at
    xname = x_names+x_names
    #Define what output data you're looking at
    yname = [y_names[0]]*5+[y_names[1]]*5
    #Define marker size and color
    markersize=2
    pltcolors = sns.color_palette("Paired")
    #Create new figure
    plt.figure(fignum)
    #Plot each subplot in a loop
    for i in range(10):
        plt.subplot(2,5,(i+1))
        #If input is not height or weight, use jitter function
        if (i+1)%5 == 4 or (i+1)%5 == 0:
            #loop through each subject as a different color
            for name, group in dataset:
                plt.scatter(x=group[xname[i]],y=group[yname[i]],color=pltcolors[name],s=markersize)
        else:
            #loop through each subject as a different color
            for name, group in dataset:
                sns.stripplot(x=group[xname[i]],y=group[yname[i]],color=pltcolors[name],jitter=0.3,size=markersize)
        #remove y axis for plots not on left size
        if (i+1)%5 != 1:
            frame1=plt.gca()
            frame1.axes.get_yaxis().set_ticks([])
        #label graphs
        plt.xlabel(xlabel[i],fontweight='bold')
        plt.ylabel(ylabel[i], fontweight='bold')
    #Save figure
    figname = figtitle+'.png'    
    plt.savefig(figname)
    return

def histogram(dataset_AM,dataset_JL):
    #Use pandas to look at histogram of output data to check for normality
    #Import necessary packages
    import matplotlib.pyplot as plt
    plt.figure(10)
    dataset_AM.hist('AngMomCor')
    plt.savefig('histogramAngMomCor.png')
    plt.figure(11)
    dataset_AM.hist('AngMomTrans')
    plt.savefig('histogramAngMomTrans.png')
    plt.figure(12)
    dataset_JL.hist('intptpm')
    plt.savefig('histogramintptpm.png')
    plt.figure(13)
    dataset_JL.hist('extptpm')
    plt.savefig('histogramextptpm.png')
    return

def importandclean(fileAM = 'SummaryAM.txt',fileJM='SummaryJL.txt'):
    #Cleans and organizes data
    import pandas
    
    #use pandas to read data
    #Load angular momentum data
    dataset_AM = pandas.read_csv(fileAM, sep='\t')
    #Load joint loading data
    dataset_JL = pandas.read_csv(fileJM, sep='\t')
    #CLEAN DATA with pandas
    #Data is normalized by height and mass in these files- we need to change that first
    dataset_AM['AngMomCor']=dataset_AM['AngMomCor']*dataset_AM['height']*dataset_AM['mass']
    dataset_AM['AngMomTrans'] = dataset_AM['AngMomTrans']*dataset_AM['height']*dataset_AM['mass']
    #Create index column
    dataset_AM['index']=list(range(len(dataset_AM)))
    dataset_JL['index']=list(range(len(dataset_JL)))
    
    #Convert data to appropriate form before machine learning algorithm
    dataset_AM = pandas.concat((dataset_AM,pandas.get_dummies(dataset_AM['activity'],drop_first=True)),axis=1)
    dataset_AM = pandas.concat((dataset_AM,pandas.get_dummies(dataset_AM['speed'],drop_first=True)),axis=1)
    dataset_AM = pandas.concat((dataset_AM,pandas.get_dummies(dataset_AM['stiffness'],drop_first=True)),axis=1)
    dataset_AM = pandas.concat((dataset_AM,pandas.get_dummies(dataset_AM['sub'],drop_first=True)),axis=1)
    dataset_JL = pandas.concat((dataset_JL,pandas.get_dummies(dataset_JL['activity'],drop_first=True)),axis=1)
    dataset_JL = pandas.concat((dataset_JL,pandas.get_dummies(dataset_JL['speed'],drop_first=True)),axis=1)
    dataset_JL = pandas.concat((dataset_JL,pandas.get_dummies(dataset_JL['stiffness'],drop_first=True)),axis=1)
    dataset_JL = pandas.concat((dataset_JL,pandas.get_dummies(dataset_JL['sub'],drop_first=True)),axis=1)
    
    #Separate Data by subject
    groups_AM = dataset_AM.groupby("sub")
    groups_JL = dataset_JL.groupby("sub")
    return dataset_AM, dataset_JL, groups_AM, groups_JL

def createalgorithm(dataset_AM,dataset_JL,printsummary='no'):
    #Creates the algorithms
    import statsmodels.formula.api as sm
    #Ignore warnings for machine learning
    import warnings
    warnings.filterwarnings("ignore")
    #Use statsmodels to create a linear regression of all models
    #First model predicts Coronal Angular Momentum
    form1 = 'AngMomCor~mass+height+OUT+ST+SLW+SSW+B+C'
    lm1 = sm.ols(formula=form1, data=dataset_AM).fit()
    #Second model predicts Transverse Angular Momentum
    form2 = 'AngMomTrans~mass+height+OUT+ST+SLW+SSW+B+C'
    lm2 = sm.ols(formula=form2, data=dataset_AM).fit()
    #Third model predicts Internal Peak Transverse Plane Moment
    form3 = 'intptpm~mass+height+OUT+ST+SLW+SSW+B+C'
    lm3 = sm.ols(formula=form3, data=dataset_JL).fit()
    #Fourth model predicts External Peak Transverse Plane Moment
    form4 = 'extptpm~mass+height+OUT+ST+SLW+SSW+B+C'
    lm4 = sm.ols(formula=form4, data=dataset_JL).fit()
    #Create a linear mixed effects model for each variable
    #Fifth model predicts Coronal Angular Momentum
    form5 = 'AngMomCor~OUT+ST+SLW+SSW+B+C+mass*height'
    lm5 = sm.mixedlm(formula=form5,data=dataset_AM,groups=dataset_AM['sub'],re_formula='~SLW+SSW')
    mdf5 = lm5.fit()
    #Sixth model predicts Transverse Angular Momentum
    form6 = 'AngMomTrans~OUT+ST+SLW+SSW+B+C+mass*height'
    lm6 = sm.mixedlm(formula=form6,data=dataset_AM,groups=dataset_AM['sub'],re_formula='~SLW+SSW')
    mdf6 = lm6.fit()
    #Seventh model predicts Internal Peak Transverse Plane Moment
    form7 = 'intptpm~OUT+ST+SLW+SSW+B+C+mass*height'
    lm7 = sm.mixedlm(formula=form7,data=dataset_JL,groups=dataset_JL['sub'],re_formula='~SLW+SSW')
    mdf7 = lm7.fit()
    #Eighth model predicts External Peak Transverse Plane Moment
    form8 = 'extptpm~OUT+ST+SLW+SSW+B+C+mass*height'
    lm8 = sm.mixedlm(formula=form8,data=dataset_JL,groups=dataset_JL['sub'],re_formula='~SLW+SSW')
    mdf8 = lm8.fit()
    
    if printsummary == 'yes':
        print(lm1.summary())
        print(lm2.summary())
        print(lm3.summary())
        print(lm4.summary())
        print(mdf5.summary())
        print(mdf6.summary())
        print(mdf7.summary())
        print(mdf8.summary())
    return lm1, lm2, lm3, lm4, mdf5, mdf6, mdf7, mdf8

def CVError(dataset_AM, dataset_JL, groups_AM, groups_JL):
    #Import packages
    import numpy as np
    from sklearn.model_selection import KFold
    #Calculates Model Error
    from statsmodels.tools.eval_measures import rmse
    #Creates the algorithms
    import statsmodels.formula.api as sm
    #Ignore warnings for machine learning
    import warnings
    warnings.filterwarnings("ignore")
    
    #Define formulas
    form1 = 'AngMomCor~mass+height+OUT+ST+SLW+SSW+B+C'
    form2 = 'AngMomTrans~mass+height+OUT+ST+SLW+SSW+B+C'
    form3 = 'intptpm~mass+height+OUT+ST+SLW+SSW+B+C'
    form4 = 'extptpm~mass+height+OUT+ST+SLW+SSW+B+C'
    form5 = 'AngMomCor~OUT+ST+SLW+SSW+B+C+mass*height'
    form6 = 'AngMomTrans~OUT+ST+SLW+SSW+B+C+mass*height'
    form7 = 'intptpm~OUT+ST+SLW+SSW+B+C+mass*height'
    form8 = 'extptpm~OUT+ST+SLW+SSW+B+C+mass*height'
    
    #define number of folds
    kf = KFold(n_splits = 10,shuffle=True)
    train_idx = np.empty(10,dtype=object)
    test_idx = np.empty(10,dtype=object)
    train_idx_JL = np.empty(10,dtype=object)
    test_idx_JL = np.empty(10,dtype=object)
    #create indices for test and train
    #has to have even percentage of each subject
    #AM test train splits
    for name, group in groups_AM:
        splits = group.to_numpy(dtype=None,copy=False)
        i=0
        for train_index, test_index in kf.split(splits):
            a = group['index']
            a = a.to_numpy(dtype=None,copy=False)
            a_train = a[train_index]
            a_test = a[test_index]
            if not train_idx[i] is None:
                train_idx[i]=np.concatenate([train_idx[i],a_train])
                test_idx[i]=np.concatenate([test_idx[i],a_test])
            else:
                train_idx[i]=a_train
                test_idx[i]=a_test
            i=i+1
    #JL test train splits
    for name, group in groups_JL:
        splits = group.to_numpy(dtype=None,copy=False)
        i=0
        for train_index, test_index in kf.split(splits):
            a = group['index']
            a = a.to_numpy(dtype=None,copy=False)
            a_train = a[train_index]
            a_test = a[test_index]
            if not train_idx_JL[i] is None:
                train_idx_JL[i]=np.concatenate([train_idx_JL[i],a_train])
                test_idx_JL[i]=np.concatenate([test_idx_JL[i],a_test])
            else:
                train_idx_JL[i]=a_train
                test_idx_JL[i]=a_test
            i=i+1
    #Rerun each model for each split
    rmse1_CV=[]
    rmse2_CV=[]
    rmse3_CV=[]
    rmse4_CV=[]
    rmse5_CV=[]
    rmse6_CV=[]
    rmse7_CV=[]
    rmse8_CV=[]
    for i in range(10):
        train = np.sort(train_idx[i])
        test = np.sort(test_idx[i])
        train_JL = np.sort(train_idx_JL[i])
        test_JL = np.sort(test_idx_JL[i])
        ds_train = dataset_AM.loc[train]
        ds_test = dataset_AM.loc[test]
        ds_train_JL = dataset_JL.loc[train_JL]
        ds_test_JL = dataset_JL.loc[test_JL]
        lm1_CV = sm.ols(formula=form1, data=ds_train).fit()
        lm2_CV = sm.ols(formula=form2, data=ds_train).fit()
        lm3_CV = sm.ols(formula=form3, data=ds_train_JL).fit()
        lm4_CV = sm.ols(formula=form4, data=ds_train_JL).fit()
        lm5_CV = sm.mixedlm(formula=form5,data=ds_train,groups=ds_train['sub'],re_formula='~SLW+SSW')
        mdf5_CV = lm5_CV.fit()
        lm6_CV = sm.mixedlm(formula=form6,data=ds_train,groups=ds_train['sub'],re_formula='~SLW+SSW')
        mdf6_CV = lm6_CV.fit()
        lm7_CV = sm.mixedlm(formula=form7,data=ds_train_JL,groups=ds_train_JL['sub'],re_formula='~SLW+SSW')
        mdf7_CV = lm7_CV.fit()
        lm8_CV = sm.mixedlm(formula=form8,data=ds_train_JL,groups=ds_train_JL['sub'],re_formula='~SLW+SSW')
        mdf8_CV = lm8_CV.fit()
        #First model predicts Coronal Angular Momentum
        Y1_CV = ds_test['AngMomCor']
        X1_CV = ds_test[['mass','height','OUT','ST','SLW','SSW','B','C']]
        #Second model predicts Transverse Angular Momentum
        Y2_CV = ds_test['AngMomTrans']
        X2_CV = ds_test[['mass','height','OUT','ST','SLW','SSW','B','C']]
        #Third model predicts Internal Peak Transverse Plane Moment
        Y3_CV = ds_test_JL['intptpm']
        X3_CV = ds_test_JL[['mass','height','OUT','ST','SLW','SSW','B','C']]
        #Fourth model predicts External Peak Transverse Plane Moment
        Y4_CV = ds_test_JL['extptpm']
        X4_CV = ds_test_JL[['mass','height','OUT','ST','SLW','SSW','B','C']]
        ypred1_CV = lm1_CV.predict(X1_CV)
        ypred2_CV = lm2_CV.predict(X2_CV)
        ypred3_CV = lm3_CV.predict(X3_CV)
        ypred4_CV = lm4_CV.predict(X4_CV)
        ypred5_CV = mdf5_CV.predict(X1_CV)
        ypred6_CV = mdf6_CV.predict(X2_CV)
        ypred7_CV = mdf7_CV.predict(X3_CV)
        ypred8_CV = mdf8_CV.predict(X4_CV)
        rmse1_CV.append(rmse(Y1_CV, ypred1_CV))
        rmse2_CV.append(rmse(Y2_CV, ypred2_CV))
        rmse3_CV.append(rmse(Y3_CV, ypred3_CV))
        rmse4_CV.append(rmse(Y4_CV, ypred4_CV))
        rmse5_CV.append(rmse(Y1_CV, ypred5_CV))
        rmse6_CV.append(rmse(Y2_CV, ypred6_CV))
        rmse7_CV.append(rmse(Y3_CV, ypred7_CV))
        rmse8_CV.append(rmse(Y4_CV, ypred8_CV))
    
    print('Model 1 Cross Validation =',sum(rmse1_CV)/len(rmse1_CV))
    print('Model 2 Cross Validation =',sum(rmse2_CV)/len(rmse2_CV))
    print('Model 3 Cross Validation =',sum(rmse3_CV)/len(rmse3_CV))
    print('Model 4 Cross Validation =',sum(rmse4_CV)/len(rmse4_CV))
    print('Model 5 Cross Validation =',sum(rmse5_CV)/len(rmse5_CV))
    print('Model 6 Cross Validation =',sum(rmse6_CV)/len(rmse6_CV))
    print('Model 7 Cross Validation =',sum(rmse7_CV)/len(rmse7_CV))
    print('Model 8 Cross Validation =',sum(rmse8_CV)/len(rmse8_CV))
    return rmse1_CV, rmse2_CV, rmse3_CV, rmse4_CV, rmse5_CV, rmse6_CV, rmse7_CV, rmse8_CV

def predictedvsactual(dataset_AM,dataset_JL,lm1,lm2,lm3,lm4,mdf5,mdf6,mdf7,mdf8):
    #Import necessary packages
    import matplotlib.pyplot as plt
    #Use pandas to separate into x/y
    #First model predicts Coronal Angular Momentum
    Y1 = dataset_AM['AngMomCor']
    X1 = dataset_AM[['mass','height','OUT','ST','SLW','SSW','B','C']]
    #Second model predicts Transverse Angular Momentum
    Y2 = dataset_AM['AngMomTrans']
    X2 = dataset_AM[['mass','height','OUT','ST','SLW','SSW','B','C']]
    #Third model predicts Internal Peak Transverse Plane Moment
    Y3 = dataset_JL['intptpm']
    X3 = dataset_JL[['mass','height','OUT','ST','SLW','SSW','B','C']]
    #Fourth model predicts External Peak Transverse Plane Moment
    Y4 = dataset_JL['extptpm']
    X4 = dataset_JL[['mass','height','OUT','ST','SLW','SSW','B','C']]
    
    #Generate predictions for each model so we can calculate errors
    ypred1 = lm1.predict(X1)
    ypred2 = lm2.predict(X2)
    ypred3 = lm3.predict(X3)
    ypred4 = lm4.predict(X4)
    ypred5 = mdf5.predict(X1)
    ypred6 = mdf6.predict(X2)
    ypred7 = mdf7.predict(X3)
    ypred8 = mdf8.predict(X4)
    
    ## Predicted vs Actual
    plt.figure(100)
    plt.scatter(Y1,ypred1)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Model- Coronal Plane WBAM')
    plt.plot([2,12],[2,12],color='red')
    plt.savefig('lm_angmomcor_predvsobs.png')
    
    plt.figure(101)
    plt.scatter(Y2,ypred2)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Model- Transverse Plane WBAM')
    plt.plot([0.5,4.0],[0.5,4.0],color='red')
    plt.savefig('lm_transmomcor_predvsobs.png')
    
    plt.figure(102)
    plt.scatter(Y3,ypred3)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Model- Internal PTPM')
    plt.plot([40,140],[40,140],color='red')
    plt.savefig('lm_intptpm_predvsobs.png')
    
    plt.figure(103)
    plt.scatter(Y4,ypred4)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Model- External PTPM')
    plt.plot([-22,5],[-22,5],color='red')
    plt.savefig('lm_extptpm_predvsobs.png')
    
    plt.figure(104)
    plt.scatter(Y1,ypred5)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Mixed Effects Model- Coronal Plane WBAM')
    plt.plot([3,13],[3,13],color='red')
    plt.savefig('lme_angmomcor_predvsobs.png')
    
    plt.figure(105)
    plt.scatter(Y2,ypred6)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Mixed Effects Model- Transverse Plane WBAM')
    plt.plot([1.5,3.75],[1.5,3.75],color='red')
    plt.savefig('lme_transmomcor_predvsobs.png')
    
    plt.figure(106)
    plt.scatter(Y3,ypred7)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Mixed Effects Model- Internal PTPM')
    plt.plot([40,140],[40,140],color='red')
    plt.savefig('lme_intptpm_predvsobs.png')
    
    plt.figure(107)
    plt.scatter(Y4,ypred8)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title('Linear Mixed Effects Model- External PTPM')
    plt.plot([-30,5],[-30,5],color='red')
    plt.savefig('lme_extptpm_predvsobs.png')
    return

def prediction(lm1,lm2,lm3,lm4,mass,height,activity='st',speed='ssw',stiffness='a'):
    #Cleans and organizes data
    import pandas
    #Define categorical variables for activity- default is st
    #If user defines wrong input, it will default to st
    if activity == 'in':
        st = 0
        out = 0
    elif activity == 'out':
        st = 0
        out  = 1
    else:
        st = 1
        out = 0
    #Define categorical variables for speed- default is ssw
    #If user defines wrong input, it will default to ssw
    if speed == 'fsw':
        slw = 0
        ssw = 0
    elif speed == 'slw':
        slw = 1
        ssw = 0
    else:
        slw = 0
        ssw = 1
    #Define categorical variables for stiffness- default is a
    #If user defines wrong input, it will default to a
    if stiffness == 'b':
        b = 1
        c = 0
    elif stiffness == 'c':
        b = 0
        c = 1
    else:
        b = 0
        c = 0
    
    data = {'mass':[mass],'height':[height],'OUT':[out],'ST':[st],'SLW':[slw],'SSW':[ssw],'B':[b],'C':[c]}
    X = pandas.DataFrame(data,columns = ['mass','height','OUT','ST','SLW','SSW','B','C'])
    ypred1 = lm1.predict(X)
    ypred2 = lm2.predict(X)
    ypred3 = lm3.predict(X)
    ypred4 = lm4.predict(X)
    yp1=ypred1.tolist()
    yp2=ypred2.tolist()
    yp3=ypred3.tolist()
    yp4=ypred4.tolist()
    print('The predicted Coronal Plane WBAM is ',round(yp1[0],2),'.',sep='')
    print('The predicted Transverse Plane WBAM is ',round(yp2[0],2),'.',sep='')
    print('The predicted Internal PTPM is ',round(yp3[0],2),'.',sep='')
    print('The predicted External PTPM is ',round(yp4[0],2),'.',sep='')
    return ypred1, ypred2, ypred3, ypred4

def main():
    #clean data
    dataset_AM, dataset_JL, groups_AM, groups_JL = importandclean(fileAM = 'SummaryAM_DummyFile.txt',fileJM='SummaryJL_DummyFile.txt')
    
    #Plot inputs versus outputs
    #This is for angular momentum data
    PlotData(groups_AM,['speed','stiffness','activity','mass','height'],['AngMomCor','AngMomTrans'],['Speed','Stiffness','Activity','Mass','Height'],['Coronal WBAM','Transverse WBAM'],'AngMomvsInputs',1)
    #This is for joint loading data
    PlotData(groups_JL,['speed','stiffness','activity','mass','height'],['intptpm','extptpm'],['Speed','Stiffness','Activity','Mass','Height'],['Internal PTPM','External PTPM'],'PTPMvsInputs',2)
    
    #Create histogram of outputs
    histogram(dataset_AM,dataset_JL)
    
    #Create algorithms
    lm1, lm2, lm3, lm4, mdf5, mdf6, mdf7, mdf8 = createalgorithm(dataset_AM,dataset_JL,'no')
    
    #Calcualte error
    CVError(dataset_AM, dataset_JL, groups_AM, groups_JL)
    
    # #plot predicted versus actual
    predictedvsactual(dataset_AM,dataset_JL,lm1,lm2,lm3,lm4,mdf5,mdf6,mdf7,mdf8)
    
    #use simple model to predict on new person
    #activity can be: 
        #'st': straight (default)
        #'out':prosthesis on outside of turn
        #'in':prosthesis on inside of turn
    #speed can be: 
        #'ssw': self selected (default) 
        #'fsw': slow
        #'slw': fast
    #stiffness can be: 
        #'ssw': 'a': low (default) 
        #'fsw': 'b': medium
        #'slw': 'c': high
    #mass is 80 kg and height is 1.7 m
    #For this subject, they are walking straight at a self selected pace with stiffness B
    prediction(lm1,lm2,lm3,lm4,mass=80,height=1.7,activity='st',speed='ssw',stiffness='b')

if __name__ == '__main__':
    main()
