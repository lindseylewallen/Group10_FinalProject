# Group10_FinalProject

The SummaryJL_DummyFile.txt and SummaryAM_DummyFile.txt are the two fake datasets that we created for this project. This data will be fed into the machine learning algorithms. All pictures of outputs come from using this code with these two fake datasets.

The Final_Code.py file allows for estimating PTPM and WBAM for Amputees. The functions in this code are explained in detail below:

main() is the main function that allows us to predict PTPM and WBAM for amputees.

importandclean() allows us to import and clean our text files full of WBAM data and PTPM data. As a default, this program loads the original 'SummaryAM.txt" and 'SummaryJL.txt' files. You can specify different files for this function to grab. This outputs cleaned full datasets and datasets grouped by subject number. This function should never be commented out. There is no output for this function.

histogram() allows us to look at a histogram of the output data to check for normality. It requires cleaned full datasets (from importandclean()) as inputs. This can be commented out if you do not want to look at the histogram. The output graphs for this function are saved in our files as:
histogramAngMomCor.png
histogramAngMomTrans.png
histogramintptpm.png
histogramextptpm.png

PlotData() plots inputs versus outputs. This requires a cleaned dataset (from importandclean()), x variable names, y variable names, xlabels for the data and ylabels for the data, and a figure title to run. This can be commented out if you do not want to look at the histogram. The output graphs for this function are saved in our files as:
PTPMvsInputs.png
AngMomvsInputs.png

createlagorithm() takes in cleaned angular momentum and joint loading datasets and outputs 8 machine learning algorithms. It gives 4 linear models and 4 linear mixed effects models, one for each output. It has an optional argument "print summary" that allows you to look at the summary for each machine learning algorithm. This function should never be commented out.

CVError() takes in all outputs from the importandclean() function and prints and returns 10-fold cross validation error for each of the 8 models. This can be commented out if you do not want to look at the error. The output for this function is saved in our files as:
CVError.png

predictedvsactual() takes all 8 linear models and 2 cleaned datasets to create predicted versus actual plots. All datapoints should fall close to the 45 degree lines. This can be commented out if you do not want to look at these plots. The output graphs for this function are saved in our files as:
lm_angmomcor_predvsobs.png
lm_transmomcor_predvsobs.png
lm_intptpm_predvsobs.png
lm_extptpm_predvsobs.png
lme_angmomcor_predvsobs.png
lme_transmomcor_predvsobs.png
lme_intptpm_predvsobs.png
lme_extptpm_predvsobs.png

prediction() takes 4 linear models, one for each output, and inputs to the machine learning algorithms (mass, height, activity, speed, and stiffness) and returns a prediction for a new subject. This is the main function that you will want to modify for our code. You can change the machine learning algorithm that you wish to use or modify each input. The inputs can be specified as follows:
activity can be set to: 
  'st': straight (default if input is something other than these three values or no activity is given)
  'out':prosthesis on outside of turn
  'in':prosthesis on inside of turn
speed can be set to: 
  'ssw': self selected (default if input is something other than these three values or no speed is given)
  'fsw': slow
  'slw': fast
stiffness can be set to: 
  'ssw': 'a': low (default if input is something other than these three values or no stiffness is given)
  'fsw': 'b': medium
  'slw': 'c': high
The output for this function is saved in our files as:
prediction.png
