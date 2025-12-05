# Using BCR sequences for classification of pediatric TBI

editing the max snippet model to use the BCR sequences from pediatric TBI patients. copied files from breast cancer directory to begin working with TBI data. 

Change Log:
- I edited the load_cdr3 function for my data. 
- changed the code to save the best_val_loss model (rather than the last epoch model)
- implemented early stopping 
- moved the optimizer.step() command to after the training block (rather than after validation block) to update the model weights before validating
- Added LR scheduler 
- Changed output weights, bias, and report to be based on validation fit with lowest entropy loss (rather than training fit with lowest entropy loss)

