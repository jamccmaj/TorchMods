#! /usr/bin/env python3

import sys

from datetime import datetime

formatter = "%Y-%m-%d %H:%M:%S"

f = open(sys.argv[1])
o = open(sys.argv[2], "w")

for line in f:
    line = line.strip()
    dt = datetime.strptime(line, formatter)
    hour = [0] * 13
    hour[dt.hour % 12] = 1
    # am or pm
    hour[12] = 0 if dt.hour < 12 else 1
    day = [0] * 7
    day[dt.weekday()] = 1
    month = [0] * 12
    month[dt.month-1] = 1
    md = hour + day + month
    md = [str(x) for x in md]
    md = "\t".join(md)
    print(f"{line}\t{md}", file=o)
f.close()
o.close()
