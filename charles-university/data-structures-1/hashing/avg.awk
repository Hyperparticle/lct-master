(NF < 2) {
    if (n > 0) print (sum1 / n)"\t"(sum2 / n);
    sum1 = sum2 = n = 0;
    print $1
}

(NF == 2) {
    sum1 += $0;
    sum2 += $2;
    n++;
}

(NR%10000 == 0) {
    if (n > 0) print (sum1 / n)"\t"(sum2 / n);
    sum1 = sum2 = n = 0;
}
