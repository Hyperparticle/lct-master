#!/bin/bash

 perl -ne 'chomp;if($_){@p=split/\t/;$t++;$c+=$p[1]eq$p[2]} END{print "prec=$c/$t=".($c/$t)."\n"}'
