# Using BCR sequences for classification of pediatric TBI

editing the max snippet model to use the BCR sequences from pediatric TBI patients. copied files from breast cancer directory to begin working with TBI data. 

Change Log:
- I edited the load_cdr3 function for my data. 
- changed the code to save the best_val_loss model (rather than the last epoch model) > then changed to save thte best_train_loss model
- implemented early stopping > then removed while testing
- moved the optimizer.step() command to after the training block (rather than after validation block) to update the model weights before validating
- Added LR scheduler > then removed because performance got worse
- Changed output weights, bias, and report to be based on validation fit with lowest entropy loss (rather than training fit with lowest entropy loss) > then switched back because with LOOV the validation set can get lucky. 
- changed report.py to say true negative rates (tnrs/TNRS) (rather than false positive rate) because that is what was actually being calculated.  False positive rate is just = (1 - true negative rate). True negative rate is the a metric of specificity. 

Scenario Log:
1. base with k-mer length = 4 (this is the base code)
2. base with k-mer length = 5
3. base with k-mer length = 6 (d1, d2, & d9 study data)
4. base with k-mer length = 7
5. base with k-mer length = 6 & d1 study data, removing control kmers from cases
6. base with k-mer length = 6 & d2 study data, removing control kmers from cases
7. base with k-mer length = 6 & d9 study data, removing control kmers from cases
8. base with k-mer length = 6 & (d1 & d2) study data, removing control kmers from cases
9. base with k-mer length = 6 & (d1 & d9) study data, removing control kmers from cases
10. base with kmer length = 6 & (d2 & d9) study data, removing control kmers from cases
11. kmer length = 6, removing control kmers from cases
12. kmer length = 6, removing control kmers from cases, removing low quantitiies (<10) <<<----