<div tabindex="-1" id="notebook" class="border-box-sizing">

<div class="container" id="notebook-container">

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [6]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython2">

<pre><span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">scipy.stats</span> <span class="kn">as</span> <span class="nn">stats</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="kn">as</span> <span class="nn">plt</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [8]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython2">

<pre><span class="n">dist</span> <span class="o">=</span> <span class="n">stats</span><span class="o">.</span><span class="n">norm</span><span class="o">.</span><span class="n">rvs</span><span class="p">(</span><span class="n">loc</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span> <span class="n">scale</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">size</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">dist</span><span class="p">,</span> <span class="s">'.'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
E/vdxD5Lzex3E/sslddvbikAAJkh2AEgMykG+8G6G1CTJva7iX2WmtnvJvZZKqnfyc2xAwAGS7Fi
BwAMkFSwm9m1ZnbCzBbN7MN1t6cMZna+mR01syfN7JiZ3dL5/dlm9rCZfavz/9nd6s/MNpvZV83s
SOfnJvT5R83s82b2DTM7bmY/m3u/zeyPO9v2E2b2OTP7oRz7bGb3mNlLnet81n/Xt59mdnsn206Y
2a8Uee1kgt3MNks6IOm9ki6R9Btmdkm9rSrFqqRb3f0Ste/Bc3Onnx+W9Ki7XyTp0c7PublF0vGu
n5vQ509K+kd3/3FJ71C7/9n228zOk/RHkmbc/TJJmyVdpzz7/DeSrt3wu5797Ozj10m6tPNv/qqT
eRNJJtglXSFp0d1PuvuKpHvVvhd8Vtz9BXf/Sue/v6/2jn6e2n39TOdhn5H0a/W0sBxmtlPS+yUd
6vp17n3+EUm/KOlTkuTuK+7+XWXeb7WveP9hM9siaZuk7yjDPrv7v0r6rw2/7tfP/ZLudffX3P3b
khbVzryJpBTs50l6tuvn5zq/y5aZ7ZZ0uaTHJL3V3V/o/OlFSW+tqVll+YSk2yStdf0u9z5foPYN
8z7dmYI6ZGZvUsb9dvfnJX1M0jOSXpD03+7+kDLu8wb9+hk031IK9kYxszdL+oKkD7n797r/1rkf
TzanM5nZPkkvuXur32Ny63PHFkk/Lemv3f1ySf+jDVMQufW7M6e8X+2D2o9JepOZ3dD9mNz63E+Z
/Uwp2J+XdH7Xzzs7v8uOmZ2hdqgfdvf7Or/+TzN7W+fvb5OU07dVXSnpA2b2tNpTbO82s88q7z5L
7arsOXd/rPPz59UO+pz7fY2kb7v7y+5+WtJ9kn5Oefe5W79+Bs23lIL9cUkXmdkFZrZV7YWGB2pu
U3BmZmrPuR539493/ekBSTd2/vtGSfdX3bayuPvt7r7T3Xer/bn+s7vfoIz7LEnu/qKkZ83s4s6v
rpb0pPLu9zOSZs1sW2dbv1rtdaSc+9ytXz8fkHSdmZ1pZhdIukjSlyd+FXdP5n+S3ifpm5KeknRH
3e0pqY8/r/bw7D8kfa3zv/dJ2q72Kvq3JD0i6ey621pS/98l6Ujnv7Pvs6SfkrTQ+by/KGkq935L
+jNJ35D0hKS/lXRmjn2W9Dm11xFOqz06+71B/VT7W+meknRC0nuLvDZXngJAZlKaigEAjIBgB4DM
EOwAkBmCHQAyQ7ADQGYIdgDIDMEOAJkh2AEgM/8HjD6MU/D7UR8AAAAASUVORK5CYII=
)</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [16]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython2">

<pre><span class="k">def</span> <span class="nf">gbm</span><span class="p">(</span><span class="n">dist</span><span class="p">):</span>
    <span class="n">t</span> <span class="o">=</span> 
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">((</span><span class="mf">0.01</span> <span class="o">-</span> <span class="mf">0.5</span><span class="o">*</span><span class="mi">2</span><span class="p">)</span> <span class="o">+</span> <span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">gbm</span><span class="p">(</span><span class="n">dist</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAX4AAAD8CAYAAABw1c+bAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAADlVJREFUeJzt21+MnXWdx/H3Z1tQwY3F7YRg29heNGqXqJAJW3VjiJhs
QWM3XrUJiyG6jVlUNCYG9cLsnRfGiAlCuljB1cAF4m5jyOquaMhegEwF2ZbC2q1/2lq3Y4hgNFlE
v3txnpiT0nam0zMMnO/7lUzo8/yemfP7puU9Z55zJlWFJKmPP1vpDUiSXliGX5KaMfyS1Izhl6Rm
DL8kNWP4JakZwy9JzRh+SWrG8EtSM6tXegOnsnbt2tq4ceNKb0OSXjL27dv3q6qaWcy1L8rwb9y4
kbm5uZXehiS9ZCT52WKv9VaPJDVj+CWpGcMvSc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmDL8kNWP4
JakZwy9JzRh+SWrG8EtSM4Zfkpox/JLUjOGXpGYMvyQ1Y/glqRnDL0nNGH5JasbwS1Izhl+SmjH8
ktSM4ZekZhYMf5I9SU4k2X+a9ST5YpJDSR5LcvlJ66uSPJLkW5PatCRp6RbzjP8OYNsZ1q8GNg8f
u4BbT1q/ETi4lM1JkiZvwfBX1QPAU2e4ZDvw1Rp5EFiT5BKAJOuBdwG3T2KzkqRzN4l7/OuAI2PH
R4dzAF8APgH8cQKPI0magGV7cTfJu4ETVbVvkdfvSjKXZG5+fn65tiVJ7U0i/MeADWPH64dzbwPe
k+SnwN3AO5J87XRfpKp2V9VsVc3OzMxMYFuSpFOZRPj3AtcN7+7ZCjxdVcer6pNVtb6qNgI7gPur
6toJPJ4k6RysXuiCJHcBVwJrkxwFPgOcB1BVtwH3AdcAh4DfAdcv12YlSeduwfBX1c4F1gu4YYFr
vg98/2w2JklaHv7mriQ1Y/glqRnDL0nNGH5JasbwS1Izhl+SmjH8ktSM4ZekZgy/JDVj+CWpGcMv
Sc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmDL8kNWP4JakZwy9JzRh+SWrG8EtSM4Zfkpox/JLUjOGX
pGYMvyQ1Y/glqRnDL0nNGH5JasbwS1Izhl+SmjH8ktTMguFPsifJiST7T7OeJF9McijJY0kuH85v
SPK9JI8nOZDkxklvXpJ09hbzjP8OYNsZ1q8GNg8fu4Bbh/PPAR+vqi3AVuCGJFuWvlVJ0iQsGP6q
egB46gyXbAe+WiMPAmuSXFJVx6vqh8PX+A1wEFg3iU1LkpZuEvf41wFHxo6PclLgk2wELgMemsDj
SZLOwbK/uJvklcA3gI9W1TNnuG5Xkrkkc/Pz88u9LUlqaxLhPwZsGDteP5wjyXmMov/1qrr3TF+k
qnZX1WxVzc7MzExgW5KkU5lE+PcC1w3v7tkKPF1Vx5ME+DJwsKo+P4HHkSRNwOqFLkhyF3AlsDbJ
UeAzwHkAVXUbcB9wDXAI+B1w/fCpbwP+DvivJI8O5z5VVfdNcgBJ0tlZMPxVtXOB9QJuOMX5/wSy
9K1JkpaDv7krSc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmDL8kNWP4JakZwy9JzRh+SWrG8EtSM4Zf
kpox/JLUjOGXpGYMvyQ1Y/glqRnDL0nNGH5JasbwS1Izhl+SmjH8ktSM4ZekZgy/JDVj+CWpGcMv
Sc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmFgx/kj1JTiTZf5r1JPlikkNJHkty+djatiRPDms3TXLj
kqSlWcwz/juAbWdYvxrYPHzsAm4FSLIKuGVY3wLsTLLlXDYrSTp3C4a/qh4AnjrDJduBr9bIg8Ca
JJcAVwCHqupwVT0L3D1cK0laQasn8DXWAUfGjo8O5051/q8m8Hin9eCX/p4///XB5XwISVo2v1nz
Brb+wz8t++O8aF7cTbIryVySufn5+ZXejiRNrUk84z8GbBg7Xj+cO+8050+pqnYDuwFmZ2drKRt5
Ib5TStJL3SSe8e8Frhve3bMVeLqqjgMPA5uTbEpyPrBjuFaStIIWfMaf5C7gSmBtkqPAZxg9m6eq
bgPuA64BDgG/A64f1p5L8iHg28AqYE9VHViGGSRJZ2HB8FfVzgXWC7jhNGv3MfrGIEl6kXjRvLgr
SXphGH5JasbwS1Izhl+SmjH8ktSM4ZekZgy/JDVj+CWpGcMvSc0YfklqxvBLUjOGX5KaMfyS1Izh
l6RmDL8kNWP4JakZwy9JzRh+SWrG8EtSM4Zfkpox/JLUjOGXpGYMvyQ1Y/glqRnDL0nNGH5Jasbw
S1Izhl+SmjH8ktSM4ZekZgy/JDWzqPAn2ZbkySSHktx0ivWLknwzyWNJfpDk0rG1jyU5kGR/kruS
vHySA0iSzs6C4U+yCrgFuBrYAuxMsuWkyz4FPFpVbwSuA24ePncd8BFgtqouBVYBOya3fUnS2VrM
M/4rgENVdbiqngXuBrafdM0W4H6AqnoC2Jjk4mFtNfCKJKuBC4BfTGTnkqQlWUz41wFHxo6PDufG
/Qh4L0CSK4DXAuur6hjwOeDnwHHg6ar6zrluWpK0dJN6cfezwJokjwIfBh4B/pDkIkY/HWwCXgNc
mOTaU32BJLuSzCWZm5+fn9C2JEknW0z4jwEbxo7XD+f+pKqeqarrq+rNjO7xzwCHgXcCP6mq+ar6
PXAv8NZTPUhV7a6q2aqanZmZWcIokqTFWEz4HwY2J9mU5HxGL87uHb8gyZphDeADwANV9QyjWzxb
k1yQJMBVwMHJbV+SdLZWL3RBVT2X5EPAtxm9K2dPVR1I8sFh/TbgDcCdSQo4ALx/WHsoyT3AD4Hn
GN0C2r0sk0iSFiVVtdJ7eJ7Z2dmam5tb6W1I0ktGkn1VNbuYa/3NXUlqxvBLUjOGX5KaMfyS1Izh
l6RmDL8kNWP4JakZwy9JzRh+SWrG8EtSM4Zfkpox/JLUjOGXpGYMvyQ1Y/glqRnDL0nNGH5Jasbw
S1Izhl+SmjH8ktSM4ZekZgy/JDVj+CWpGcMvSc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmDL8kNbOo
8CfZluTJJIeS3HSK9YuSfDPJY0l+kOTSsbU1Se5J8kSSg0neMskBJElnZ8HwJ1kF3AJcDWwBdibZ
ctJlnwIerao3AtcBN4+t3Qz8W1W9HngTcHASG5ckLc1invFfARyqqsNV9SxwN7D9pGu2APcDVNUT
wMYkFyd5FfB24MvD2rNV9euJ7V6SdNYWE/51wJGx46PDuXE/At4LkOQK4LXAemATMA98JckjSW5P
cuE571qStGSTenH3s8CaJI8CHwYeAf4ArAYuB26tqsuA3wLPe40AIMmuJHNJ5ubn5ye0LUnSyRYT
/mPAhrHj9cO5P6mqZ6rq+qp6M6N7/DPAYUY/HRytqoeGS+9h9I3geapqd1XNVtXszMzMWY4hSVqs
xYT/YWBzkk1Jzgd2AHvHLxjeuXP+cPgB4IHhm8EvgSNJXjesXQU8PqG9S5KWYPVCF1TVc0k+BHwb
WAXsqaoDST44rN8GvAG4M0kBB4D3j32JDwNfH74xHAaun/AMkqSzkKpa6T08z+zsbM3Nza30NiTp
JSPJvqqaXcy1/uauJDVj+CWpGcMvSc0YfklqxvBLUjOGX5KaMfyS1Izhl6RmDL8kNWP4JakZwy9J
zRh+SWrG8EtSM4Zfkpox/JLUjOGXpGYMvyQ1Y/glqRnDL0nNGH5JasbwS1Izhl+SmjH8ktSM4Zek
Zgy/JDWTqlrpPTxPknngZ0v89LXArya4nZeCjjNDz7k7zgw95z7bmV9bVTOLufBFGf5zkWSuqmZX
eh8vpI4zQ8+5O84MPedezpm91SNJzRh+SWpmGsO/e6U3sAI6zgw95+44M/Sce9lmnrp7/JKkM5vG
Z/ySpDOYmvAn2ZbkySSHkty00vtZLkk2JPlekseTHEhy43D+1Un+PcmPh/9etNJ7nbQkq5I8kuRb
w3GHmdckuSfJE0kOJnnLtM+d5GPDv+39Se5K8vJpnDnJniQnkuwfO3faOZN8cujbk0n+5lweeyrC
n2QVcAtwNbAF2Jlky8ruatk8B3y8qrYAW4EbhllvAr5bVZuB7w7H0+ZG4ODYcYeZbwb+rapeD7yJ
0fxTO3eSdcBHgNmquhRYBexgOme+A9h20rlTzjn8P74D+Mvhc740dG9JpiL8wBXAoao6XFXPAncD
21d4T8uiqo5X1Q+HP/+GUQjWMZr3zuGyO4G/XZkdLo8k64F3AbePnZ72mV8FvB34MkBVPVtVv2bK
5wZWA69Ishq4APgFUzhzVT0APHXS6dPNuR24u6r+r6p+Ahxi1L0lmZbwrwOOjB0fHc5NtSQbgcuA
h4CLq+r4sPRL4OIV2tZy+QLwCeCPY+emfeZNwDzwleEW1+1JLmSK566qY8DngJ8Dx4Gnq+o7TPHM
JzndnBNt3LSEv50krwS+AXy0qp4ZX6vRW7Wm5u1aSd4NnKiqfae7ZtpmHqwGLgdurarLgN9y0i2O
aZt7uKe9ndE3vdcAFya5dvyaaZv5dJZzzmkJ/zFgw9jx+uHcVEpyHqPof72q7h1O/2+SS4b1S4AT
K7W/ZfA24D1JfsroNt47knyN6Z4ZRs/qjlbVQ8PxPYy+EUzz3O8EflJV81X1e+Be4K1M98zjTjfn
RBs3LeF/GNicZFOS8xm9CLJ3hfe0LJKE0T3fg1X1+bGlvcD7hj+/D/jXF3pvy6WqPllV66tqI6O/
2/ur6lqmeGaAqvolcCTJ64ZTVwGPM91z/xzYmuSC4d/6VYxex5rmmcedbs69wI4kL0uyCdgM/GDJ
j1JVU/EBXAP8N/A/wKdXej/LOOdfM/rx7zHg0eHjGuAvGL0L4MfAfwCvXum9LtP8VwLfGv489TMD
bwbmhr/vfwEumva5gX8EngD2A/8MvGwaZwbuYvQ6xu8Z/XT3/jPNCXx66NuTwNXn8tj+5q4kNTMt
t3okSYtk+CWpGcMvSc0YfklqxvBLUjOGX5KaMfyS1Izhl6Rm/h+NeKYM7n28qQAAAABJRU5ErkJg
gg==
)</div>

</div>

</div>

</div>

</div>

</div>

</div>