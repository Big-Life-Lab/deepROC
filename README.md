# DeepROC
Deep ROC Analysis Toolkit

Code for deep ROC analysis by André Carrington and Yusuf Sheikh.  This supercedes the Partial-AUC-C toolkit with additions and corrections.

André M. Carrington, Douglas G. Manuel, Paul W. Fieguth, Tim Ramsay, Venet Osmani, Bernhard Wernly, Carol Bennett, Steven Hawken, Olivia Magwood, Yusuf Sheikh, Matthew McInnes and Andreas Holzinger. Deep ROC Analysis and AUC as Balanced Average Accuracy for Improved Classifier Selection, Audit and Explanation. IEEE Transactions on Pattern Analysis and Machine Intelligence, Early Access, Jan 25, 2022. doi:10.1109/TPAMI.2022.3145392 https://doi.org/10.1109/TPAMI.2022.3145392

Note: refactoring is underway.
  
## Instructions
Ensure you have a Python 3.8 interpreter.  
Copy files in Python3.8 folder to a local folder.  
Create a subfolder called output.

In your code:  
import deepROC as ac  
  
Then call the deepROC function with appropriate inputs:  
results, EQresults = ac.deepROC(*kwargs)  
  
An example (under extras) of calling the function can be found in:  
test_deepROC.py  
  
Analysis can be performed with:  
analyze.py  
reanalyze.py  
analyze-table.py    

## Examples/explanations of inputs to deepROC
To be completed.  
