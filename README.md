# DeepROC
Deep ROC Analysis Toolkit

Code for deep ROC analysis by André Carrington and Yusuf Sheikh.  This supercedes the Partial-AUC-C toolkit with additions and corrections.

Manuscript submitted:
André M. Carrington, Douglas G. Manuel, Paul W. Fieguth, Venet Osmani, Bernhard Wernly, Steven Hawken, Matthew McInnes, Tim Ramsay, Carol Bennett, Olivia Magwood, Yusuf Sheikh and Andreas Holzinger. Deep ROC Analysis and AUC as Balanced Average Accuracy to Improve Model Selection, Understanding and Interpretation. Submitted, 2021.

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
analyze15.py  
compare15.py  
analyze-table.py  

## Examples/explanations of inputs to deepROC
To be completed.  
