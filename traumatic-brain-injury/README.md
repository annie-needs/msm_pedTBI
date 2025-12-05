# Using BCR sequences for classification of pediatric TBI

editing the max snippet model to use the BCR sequences from pediatric TBI patients. copied files from breast cancer directory to begin working with TBI data. 

Change Log:
- I edited the load_cdr3 function for my data. 
- changed the code to save the best_val_loss model (rather than the last epoch model)
- implemented early stopping 
- moved the optimizer.step() command to after the training block (rather than after validation block) to update the model weights before validating
- Added LR scheduler 
- Changed output weights, bias, and report to be based on validation fit with lowest entropy loss (rather than training fit with lowest entropy loss)
- changed report.py to say true negative rates (tnrs/TNRS) (rather than false positive rate) because that is what was actually being calculated.  False positive rate is just = (1 - true negative rate). True negative rate is the a metric of specificity. 

Scenario Log:
1. base (k-mer length = 4)
2. base with k-mer length = 5
3. base with k-mer length = 6
4. base with k-mer length = 7