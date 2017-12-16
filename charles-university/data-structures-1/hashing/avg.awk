{ sum1 += $0; sum2 += $2; n++ }
(NR%25000==0){ if (n > 0) print (sum1 / n)"\t"(sum2 / n); }