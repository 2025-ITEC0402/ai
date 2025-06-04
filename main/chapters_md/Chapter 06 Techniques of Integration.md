# Chapter 06 Techniques of Integration

# 적분 기법

## 부분적분

부분적분은 곱의 형태로 이루어진 함수의 적분을 계산하는 유용한 기법입니다.

다음과 같은 규칙을 따릅니다.

$$
\int u\,dv = u\,v - \int v\,du
$$

실제로 적분을 계산할 때는 미분했을 때 더 간단해지는 (혹은 최소한 더 복잡해지지 않는) 함수를 `u`로 선택하고, 나머지 부분을 `dv`로 선택합니다. `dv`는 비교적 쉽게 적분할 수 있어야 합니다.

## 삼각함수 적분

삼각함수 `\sin^m x \cos^n x`의 적분은 `m`과 `n`의 홀짝성에 따라 다른 전략을 사용합니다.

* **m이 홀수인 경우**: `\sin^2 x = 1 - \cos^2 x`를 이용하여 하나의 `\sin x`를 제외한 모든 `\sin x`를 `\cos x`로 변환하고, `u = \cos x`로 치환합니다.

* **n이 홀수인 경우**: `\cos^2 x = 1 - \sin^2 x`를 이용하여 하나의 `\cos x`를 제외한 모든 `\cos x`를 `\sin x`로 변환하고, `u = \sin x`로 치환합니다.

* **m과 n이 모두 짝수인 경우**: 반각 공식 `\sin^2 x = (1 - \cos 2x)/2`, `\cos^2 x = (1 + \cos 2x)/2`를 사용하여 적분을 간단히 합니다.

## 특수 치환

특정 형태의 식이 포함된 적분은 다음 치환을 시도해 볼 수 있습니다.

* `\sqrt{a^2 - x^2}`이 포함된 경우: `x = a\sin\theta`

* `\sqrt{a^2 + x^2}`이 포함된 경우: `x = a\tan\theta`

* `\sqrt{x^2 - a^2}`이 포함된 경우: `x = a\sec\theta`

## 부분분수 분해

유리 함수 `P(x)/Q(x)` (단, `\deg P < \deg Q`)의 적분은 부분분수 분해를 통해 계산할 수 있습니다.

* `Q(x)`가 서로 다른 일차식의 곱으로 인수분해되는 경우: 각 일차식에 대응하는 부분분수의 합으로 분해합니다.

* `Q(x)`에 반복되는 일차 인수가 있는 경우: 반복되는 만큼 해당 일차식을 거듭제곱한 형태의 항을 구성합니다.

* `Q(x)`에 분해되지 않는 이차 인수가 있는 경우: 해당 이차식에 대한 `Ax+B` 형태의 분수를 구성합니다.

* `Q(x)`에 반복되는 분해되지 않는 이차 인수가 있는 경우: 반복 횟수만큼 `(Ax+B)/(ax^2+bx+c)^k` 형태의 항을 구성합니다.

## 이상적분

이상적분은 적분 구간이 무한대이거나 피적분함수가 특정 구간에서 정의되지 않는 경우의 적분입니다.

* **(a)** $\displaystyle \int_a^{\infty} f(x)\,dx = \lim_{t\to\infty}\int_a^t f(x)\,dx$

* **(b)** $\displaystyle \int_{-\infty}^b f(x)\,dx = \lim_{t\to-\infty}\int_t^b f(x)\,dx$

* **(c)** $\displaystyle \int_{-\infty}^{\infty} f(x)\,dx = \int_{-\infty}^a f(x)\,dx + \int_a^{\infty} f(x)\,dx$ (a는 임의의 실수)

# 부분 분수 분해와 이상 적분

## 부분 분수 분해

서로 다른 선형 인수로 이루어진 유리 함수 $\dfrac{P(x)}{Q(x)}$는 다음과 같이 분해될 수 있습니다. (적분이 수렴한다고 가정)

$$
\frac{P(x)}{Q(x)} = \frac{A_1}{a_1x + b_1} + \frac{A_2}{a_2x + b_2} + \cdots + \frac{A_k}{a_kx + b_k}
$$

만약 $(a_1x + b_1)$이 $r$번 반복된다면:

$$
\frac{B_1}{a_1x + b_1} + \frac{B_2}{(a_1x + b_1)^2} + \cdots + \frac{B_r}{(a_1x + b_1)^r}
$$

기약 이차 인수 $ax^2 + bx + c$가 있으면:

$$
\frac{Ax + B}{ax^2 + bx + c}
$$

반복된다면 다음과 같이 구성합니다:

$$
\frac{A_1x + B_1}{ax^2 + bx + c} + \frac{A_2x + B_2}{(ax^2 + bx + c)^2} + \cdots + \frac{A_rx + B_r}{(ax^2 + bx + c)^r}
$$

## 이상 적분의 정의

다음과 같은 경우에 이상 적분을 정의합니다.

* **(a)** $a$에서 $f$가 무한 불연속인 경우:
  $$\int_a^b f(x)\,dx = \lim_{t\to a^+}\int_t^b f(x)\,dx$$

* **(b)** $b$에서 $f$가 무한 불연속인 경우:
  $$\int_a^b f(x)\,dx = \lim_{t\to b^-}\int_a^t f(x)\,dx$$

* **(c)** $a<c<b$인 $c$에서 불연속인 경우:
  $$\int_a^b f(x)\,dx = \int_a^c f(x)\,dx + \int_c^b f(x)\,dx$$

## 이상 적분의 비교 판정법

$a<x<b$이고 $f,t$가 연속이며 $f(x)>t(x)>0$일 때:

* **(a)** 만약 $\int_a^\infty f(x)\,dx$가 수렴하면, $\int_a^\infty t(x)\,dx$도 수렴합니다.

* **(b)** 만약 $\int_a^\infty t(x)\,dx$가 발산하면, $\int_a^\infty f(x)\,dx$도 발산합니다.

# 곡선의 길이와 표면적 그리고 확률 밀도 함수

## 곡선의 길이

곡선 $C$를 다각형으로 근사할 때 길이 $L$은:

$$
L = \lim_{n\to\infty} \sum_{i=1}^n |P_{i-1}P_i|
$$

$y = f(x),\,a\le x\le b$일 때:

$$
L = \int_a^b \sqrt{1 + [f'(x)]^2}\,dx
$$

$x = g(y),\,c\le y\le d$일 때:

$$
L = \int_c^d \sqrt{1 + [g'(y)]^2}\,dy
$$

## 표면적

$y = f(x)$, $a\le x\le b$를 x축 회전 시:

$$
S = \int_a^b 2\pi f(x)\sqrt{1 + [f'(x)]^2}\,dx
$$

$x = g(y)$, $c\le y\le d$일 때:

$$
S = \int_c^d 2\pi y\sqrt{1 + [g'(y)]^2}\,dy
$$

y축 회전 시:

$$
S = \int_a^b 2\pi x\sqrt{1 + [f'(x)]^2}\,dx
\quad\text{또는}\quad
S = \int_c^d 2\pi g(y)\sqrt{1 + [g'(y)]^2}\,dy
$$

## 심박출량

**(a)** 심박출량은 단위 시간당 심장이 내보내는 혈액의 양입니다.

**(b)** 염료 희석법: 염료 주입 후 시간에 따른 농도 아래 면적을 이용하여 계산합니다.

$$
\frac{A}{\int_0^T c(t)\,dt}
$$

## 확률 밀도 함수

확률 변수 $X$의 밀도 함수 $f$는 다음을 만족합니다:

* $f(x)\ge0$ 모든 $x$에 대해

* $\displaystyle\int_{-\infty}^{\infty}f(x)\,dx=1$

$a\le X\le b$일 확률:

$$
P(a\le X\le b)=\int_a^b f(x)\,dx
$$

# 공학 수학

## 유체 정역학

벽을 높이 $\Delta x$ 스트립으로 나누고, 각 스트립 깊이 $x$에서 폭 $f(x)$인 직사각형으로 근사합니다. 밀도 $\rho$일 때 힘은:

$$
F=\int_a^b \rho\,x\,f(x)\,dx
$$

## 질량 중심

판이 $a<x<b$에서 $y=f(x)$와 $y=0$ 사이에 있을 때, 질량 중심:

$$
x=\frac{1}{A}\int_a^b x\,f(x)\,dx
\quad,\quad
y=\frac{1}{A}\int_a^b \frac{[f(x)]^2}{2}\,dx
\quad,\quad
A=\int_a^b f(x)\,dx
$$

## 파푸스의 정리

평면 영역 $S$를 한 직선 $l$을 중심으로 회전시킨 입체 부피는 $S$ 면적과 질량 중심이 이동한 거리의 곱입니다.

## 확률 밀도 함수 (중복 섹션 제거 가능)

$f(x)\ge0$, $\displaystyle\int_{-\infty}^{\infty}f(x)\,dx=1$

$\int_0^{60}f(x)\,dx$는 60kg 미만일 확률입니다.

평균: $\displaystyle\mu=\int_{-\infty}^{\infty}x\,f(x)\,dx$

중앙값 $m$: $\displaystyle\int_{-\infty}^m f(x)\,dx=\tfrac12$

## 정규 분포

정규 분포 밀도 함수:

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\tfrac{(x-\mu)^2}{2\sigma^2}}
$$

$\mu$는 평균, $\sigma$는 표준 편차입니다.

## 확률 변수와 확률 분포

확률 변수는 특정 사건 결과에 따라 값이 결정되는 변수입니다.

정규 분포는 종 모양의 분포로, 평균 $\mu$, 표준 편차 $\sigma$를 가집니다.
