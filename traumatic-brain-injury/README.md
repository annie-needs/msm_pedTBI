# Using BCR sequences for classification of pediatric TBI


editing the max snippet model to use the BCR sequences from pediatric TBI patients. copied files from breast cancer directory to begin working with TBI data. 

Change Log:
- I edited the load_cdr3 function for my data. 
- changed the code to save the best_val_loss model (rather than the last epoch model) > then changed to save the best_train_loss model
- implemented early stopping > then removed while testing
- moved the optimizer.step() command to after the training block (rather than after validation block) to update the model weights before validating
- Added LR scheduler > then removed because performance got worse
- Changed output weights, bias, and report to be based on validation fit with lowest entropy loss (rather than training fit with lowest entropy loss) > then switched back because with LOOV the validation set can get lucky. 
- changed report.py to say true negative rates (tnrs/TNRS) (rather than false positive rate) because that is what was actually being calculated.  False positive rate is just = (1 - true negative rate). True negative rate is the a metric of specificity. 

Scenario Log (for scenarios 28 - 56: all with sedation cohort as controls):
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
12. kmer length = 6, removing low quantities (<5)
13. kmer length = 6, removing low quantities (<**2**), removing control kmers from cases (changed min_count bc otherwise removed all sequences from several samples.)
14. kmer length = 6, d9 study data, using d1 as controls, removing "control" kmers from cases
15. kmer length = 6, d9 study data, using d1 as controls, removing low quantities (<5) 
16. kmer length = 6, d9 study data, using d1 as controls, removing low quantities (<5), removing "control" kmers from cases
17. kmer length = 6, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
18. kmer length = 6, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample, num_fits reduced by factor of 10 (num_fits = 1,638)
19. kmer length = 6, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample, num_fits reduced by factor of 2 (num_fits = 8,192)
20. kmer length = 6, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample, num_fits increased by factor of 1.5 (num_fits = 24,576)
21. MOTIF length = 4 with window size = 6, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
note: up thru 21, dropping low quantity kmers. Could change to drop low quantity CDR3s (move line up b4 kmer splits)

Making edits based on performance of scenarios 1-21
22. kmer length = 7, d9 study data, using d1 as controls, removing low quantities (<5) [scen15 but k=7]
23. kmer length = 7, d9 study data, using d1 as controls, removing low quantity **CDRs** (<**2**)(changed min_count bc otherwise removed all sequences from several samples.)
24. MOTIF length = 3 with window size = 5, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
25. MOTIF length = 5 with window size = 7, d9 study data, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
26. MOTIF length = 4 with window size = 6, d9 study data, removing low quantity **CDRs** (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
J
27. MOTIF length = 4 with window size = 6, d9 study data, **using d1 as controls**, removing low quantities (<5), removing control kmers from cases but keeping overlap kmers in the holdout sample
28. Motif= 3, Window=4, MinCount=2
29. Motif= 3, Window=4, MinCount=5
30. Motif= 3, Window=5, MinCount=2
31. Motif= 3, Window=5, MinCount=5
32. Motif= 3, Window=6, MinCount=2
33. Motif= 3, Window=6, MinCount=5
34. Motif= 3, Window=7, MinCount=2
35. Motif= 3, Window=7, MinCount=5
36. Motif= 3, Window=8, MinCount=2
37. Motif= 3, Window=8, MinCount=5
38. Motif= 4, Window=4, MinCount=2
39. Motif= 4, Window=4, MinCount=5
40. Motif= 4, Window=5, MinCount=2
41. Motif= 4, Window=5, MinCount=5
42. Motif= 4, Window=6, MinCount=2
43. Motif= 4, Window=6, MinCount=5
44. Motif= 4, Window=7, MinCount=2
45. Motif= 4, Window=7, MinCount=5
46. Motif= 4, Window=8, MinCount=2
47. Motif= 4, Window=8, MinCount=5
48. Motif= 5, Window=5, MinCount=2
49. Motif= 5, Window=5, MinCount=5
50. Motif= 5, Window=6, MinCount=2
51. Motif= 5, Window=6, MinCount=5
52. Motif= 5, Window=7, MinCount=2
53. Motif= 5, Window=7, MinCount=5
54. Motif= 5, Window=8, MinCount=2
55. Motif= 5, Window=8, MinCount=5