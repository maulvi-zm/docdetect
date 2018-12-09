# docdetect

[![Build Status](https://travis-ci.org/alessandrozamberletti/docdetect.svg?branch=master)](https://travis-ci.org/alessandrozamberletti/docdetect)
[![Build status](https://ci.appveyor.com/api/projects/status/l1gjc8g7c1q3846j/branch/master?svg=true)](https://ci.appveyor.com/project/alessandrozamberletti/docdetect/branch/master)
[![codecov](https://codecov.io/gh/alessandrozamberletti/docdetect/branch/master/graph/badge.svg)](https://codecov.io/gh/alessandrozamberletti/docdetect)
[![Maintainability](https://api.codeclimate.com/v1/badges/a9aa496faab72437e650/maintainability)](https://codeclimate.com/github/alessandrozamberletti/docdetect/maintainability)

~~Unofficial implementation of~~ Trying to emulate [Fast and Accurate Document Detection for Scanning](https://blogs.dropbox.com/tech/2016/08/fast-and-accurate-document-detection-for-scanning/) without ML
 
2018/10/20: check literature and datasets **done**  
2018/10/23: compare canny+mser+hough and ml approach **done**  
2018/11/01: too many lines from hough with low thr: group similar rho, ~~theta~~ values **done**  
2018/11/02: detect intersections, identify unusual ones **done**  
2018/11/03: 
> iterate through potential document corners, and enumerate all possible quadrilaterals

2018/11/05: cycle graph search (dfs) **done**  
2018/11/08: detect most probable quadrilaters and drop false positive **wip**  
2018/11/09: try to take the largest rectangle, maybe it works **wip**  
2018/11/11: 1st version spike completed 
2018/11/13: implementation started..

```python
edges = docdetect.detect_edges(im)
lines = docdetect.detect_lines(edges)
corners = docdetect.find_corners(lines_unique, im)
rects = docdetect.find_quadrilaterals(corners)
for rect in rect:
    ...
```

# resources  
* https://arxiv.org/pdf/1504.06375.pdf
* https://pdollar.github.io/files/papers/DollarPAMI15edges.pdf
