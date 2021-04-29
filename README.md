# Group10_FinalProject

The SummaryJL_DummyFile.txt and SummaryAM_DummyFile.txt are the two fake datasets that we created for this project. This data will be fed into the machine learning algorithms.

The Final_Code.py file allows for estimating PTPM and WBAM for Amputees. This code is explained in detail below:

main() is the main function that allows us to predict PTPM and WBAM for amputees.

importandclean() allows us to import and clean our text files full of WBAM data and PTPM data. As a default, this program loads the original 'SummaryAM.txt" and 'SummaryJL.txt' files. You can specify different files for this function to grab. This outputs cleaned full datasets and datasets grouped by subject number.

histogram() allows us to look at a histogram of the output data to check for normality. It requires cleaned full datasets (from importandclean()) as inputs.

PlotData() plots inputs versus outputs. This requires a cleaned dataset (from importandclean()), x variable names, y variable names, xlabels for the data and ylabels for the data, and a figure title to run.

createlagorithm() takes in cleaned angular momentum and joint loading datasets and outputs 8 machine learning algorithms. It gives 4 linear models and 4 linear mixed effects models, one for each output. It has an optional argument "print summary" that allows you to look at the summary for each machine learning algorithm.

CVError() takes in all outputs from the importandclean() function and prints and returns 10-fold cross validation error for each of the 8 models.

predictedvsactual() takes all 8 linear models and 2 cleaned datasets to create predicted versus actual plots. All datapoints should fall close to the 45 degree lines.

prediction() takes 4 linear models, one for each output, and inputs to the machine learning algorithms (mass, height, out, st, slw, ssw, b, and c) and returns a prediction for a new subject. You can modify each of these machine learning inputs or change the machine learning algorithm that you wish to use.
