# Deep ROC Analysis
  
The second release of Deep ROC Analysis code for Python 3, implements ROC and AUC related measures which are of general interest: the AUC and C statistic for binary outcomes, are generalized from measuring a whole ROC curve, to a part [1][2].  An alternative, called the Partial AUC, is misleading (and a misnomer) because it only measures sensitivity--it does not represent AUC in a part.  The Concordant Partial AUC and Partial C statistic are proper generalizations.  
  
pip install deeproc  
See: https://pypi.org/project/deeproc/  
  
This code is a complete refactoring of the earlier release, from procedural to object-oriented, and written to simplify use, understanding, extension and management.  
  
The code consists of four classes: SimpleROC, FullROC, DeepROC and ConcordanceMatrixPlot. Please read "Deep ROC Code Documentation.docx" for further information.  
  
[1] Carrington AM, Manuel DG, Fieguth PW, Ramsay T, Osmani V, Wernly B, Bennett C, Hawken S, Magwood O, Sheikh Y, McInnes M, Holzinger A. Deep ROC Analysis and AUC as Balanced Average Accuracy for improved classifier selection, audit and explanation. IEEE Transactions on Pattern Analysis and Machine Intelligence. Online ahead of print. January 25, 2022. doi:10.1109/TPAMI.2022.3145392 PMID:35077357  
  
[2] Carrington AM, Fieguth PW, Qazi H, Holzinger A, Chen HH, Mayr F and Manuel DG. A new concordant partial AUC and partial C statistic for imbalanced data in the evaluation of machine learning algorithms, BMC Medical Informatics and Decision Making 20, 4 (2020) doi:10.1186/s12911-019-1014-6 PMID:31906931  

