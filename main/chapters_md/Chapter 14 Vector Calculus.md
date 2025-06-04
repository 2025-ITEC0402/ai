# Chapter 14 Vector Calculus<br>
<br>
# 7. 그린 정리와 곡면의 넓이<br>
<br>
## 그린 정리<br>
<br>
양의 방향을 가지고 조각별로 부드러운 단순 폐곡선 C와 C로 둘러싸인 영역 D가 평면에 있다고 가정합시다. P와 Q가 D를 포함하는 열린 영역에서 연속적인 편도함수를 갖는다면 다음이 성립합니다.<br>
<br>
$$ \oint_C P\,dx + Q\,dy = \iint_D \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) dA $$<br>
<br>
## 매개변수 곡면의 넓이<br>
<br>
매개변수 u, v에 대한 부드러운 곡면 S가 다음과 같이 주어진 경우,<br>
<br>
$$ \mathbf{r}(u, v) = x(u, v)\,\mathbf{i} + y(u, v)\,\mathbf{j} + z(u, v)\,\mathbf{k} $$<br>
<br>
여기서 (u, v)는 영역 D에 속하고, S는 (u, v)가 D 전체를 움직일 때 한 번만 덮는다고 가정하면, S의 곡면 넓이는 다음과 같습니다.<br>
<br>
$$ A(S) = \iint_D \bigl|\mathbf{r}_u \times \mathbf{r}_v\bigr|\,dA $$<br>
<br>
# 8. 곡선으로 둘러싸인 영역의 넓이<br>
<br>
곡선 C로 둘러싸인 영역의 넓이는 C를 따라 계산되는 선적분으로 표현할 수 있습니다. 다음은 세 가지 표현입니다.<br>
<br>
* $$ A = \oint_C x\,dy $$<br>
* $$ A = -\oint_C y\,dx $$<br>
* $$ A = \tfrac{1}{2}\oint_C \bigl(x\,dy - y\,dx\bigr) $$<br>
<br>
# 9. 벡터장의 curl과 div<br>
<br>
## curl F<br>
<br>
3차원 벡터장 **F** = P **i** + Q **j** + R **k**에 대해 curl **F**는 다음과 같이 정의됩니다.<br>
<br>
$$ \operatorname{curl}\,\mathbf{F}
= \Bigl(\tfrac{\partial R}{\partial y}-\tfrac{\partial Q}{\partial z}\Bigr)\mathbf{i}
+ \Bigl(\tfrac{\partial P}{\partial z}-\tfrac{\partial R}{\partial x}\Bigr)\mathbf{j}
+ \Bigl(\tfrac{\partial Q}{\partial x}-\tfrac{\partial P}{\partial y}\Bigr)\mathbf{k} $$<br>
<br>
## div F<br>
<br>
벡터장 **F**의 divergence(발산)는 다음과 같이 정의됩니다.<br>
<br>
$$ \operatorname{div}\,\mathbf{F}
= \tfrac{\partial P}{\partial x} + \tfrac{\partial Q}{\partial y} + \tfrac{\partial R}{\partial z} $$<br>
<br>
## curl과 div의 물리적 의미<br>
<br>
유체 흐름에서 속도장 **F**가 주어졌을 때, curl **F**는 유체가 회전하는 경향을 나타내며, 그 크기는 회전 속도를 나타냅니다. div **F**는 유체가 한 점에서 바깥으로 흐르는(발산하는) 경향을 나타냅니다.<br>
<br>
# 12. 스칼라 함수의 곡면 적분<br>
<br>
## 정의<br>
<br>
곡면 S 위에서 스칼라 함수 f의 곡면 적분은 다음과 같이 정의됩니다. S를 작은 조각들 Sᵢⱼ로 나누고,<br>
<br>
$$ \iint_S f(x,y,z)\,dS
= \lim_{m,n\to\infty}\sum_{i=1}^m\sum_{j=1}^n f(P_{ij}^*)\,\Delta S_{ij} $$<br>
<br>
여기서 ΔSᵢⱼ는 조각 Sᵢⱼ의 넓이이고, Pᵢⱼ*는 표본점입니다.<br>
<br>
## 매개변수 곡면에서의 계산<br>
<br>
S가 벡터 함수 **r**(u,v)로 주어진 매개변수 곡면인 경우,<br>
<br>
$$ \iint_S f(x,y,z)\,dS
= \iint_D f\bigl(\mathbf{r}(u,v)\bigr)\,\bigl|\mathbf{r}_u\times\mathbf{r}_v\bigr|\,dA $$<br>
<br>
여기서 D는 매개변수 영역입니다.<br>
<br>
## z = g(x, y) 형태의 곡면에서의 계산<br>
<br>
S가 z = g(x,y) 형태로 주어진 경우, 곡면 적분은 g의 편미분을 이용하여 계산할 수 있습니다.<br>
<br>
# 벡터장과 곡면<br>
<br>
## 보존장 (Conservative Field)<br>
<br>
벡터장 **F**가 어떤 스칼라 함수 f의 기울기, 즉 **F** = ∇f로 표현될 수 있으면, **F**를 보존적 벡터장이라고 합니다. 이때 f를 퍼텐셜 함수라고 부릅니다.<br>
<br>
## 매개변수 곡면 (Parametric Surface)<br>
<br>
매개변수 곡면 S는 u, v의 벡터 함수 **r**(u,v)=x(u,v)**i**+y(u,v)**j**+z(u,v)**k 로 표현되는 곡면입니다. u 또는 v가 상수일 때 생성되는 곡선을 격자 곡선이라고 합니다.<br>
<br>
## 방향성 곡면 (Oriented Surface)<br>
<br>
각 점에서 단위 법선 벡터 **n**을 연속적으로 정의할 수 있는 곡면을 방향성 곡면이라 합니다. 곡면의 위쪽 방향은 **n**, 아래쪽 방향은 –**n**을 사용합니다.<br>
<br>
## 질량과 질량 중심<br>
<br>
얇은 판이 곡면 S 형태이고 밀도 ρ(x,y,z)를 가질 때,<br>
<br>
* 질량:  $$ m = \iint_S \rho(x,y,z)\,dS $$<br>
* 질량중심:  
  $$ \bar{x} = \frac{1}{m}\iint_S x\,\rho(x,y,z)\,dS,\quad
     \bar{y} = \frac{1}{m}\iint_S y\,\rho(x,y,z)\,dS,\quad
     \bar{z} = \frac{1}{m}\iint_S z\,\rho(x,y,z)\,dS $$<br>
<br>
## 스토크스 정리 (Stokes’ Theorem)<br>
<br>
조각별로 부드러운 방향성 곡면 S가 경계곡선 C로 둘러싸여 있을 때, 벡터장 **F**의 curl에 대한 면적분과 선적분이 같다는 정리입니다.<br>
<br>
$$ \oint_C \mathbf{F}\cdot d\mathbf{r} = \iint_S (\operatorname{curl}\mathbf{F})\cdot d\mathbf{S} $$<br>
<br>
# 벡터 미적분학<br>
<br>
## 스토크스 정리<br>
<br>
단순하고 닫힌 경계 C를 가진 방향성 곡면 S와 각 성분이 연속인 벡터장 **F**에 대해 다음이 성립합니다.<br>
<br>
$$ \oint_C \mathbf{F}\cdot d\mathbf{r} = \iint_S (\operatorname{curl}\mathbf{F})\cdot d\mathbf{S} $$<br>
<br>
## 발산 정리<br>
<br>
E를 단순 폐체 영역, S를 그 경계면이라 할 때, 연속 편도함수를 갖는 벡터장 **F**에 대해<br>
<br>
$$ \iint_S \mathbf{F}\cdot d\mathbf{S} = \iiint_E (\operatorname{div}\mathbf{F})\,dV $$<br>
<br>
만약 방향이 반대라면 식 앞에 –1을 곱합니다.<br>
<br>
### 면적분 계산 방법<br>
<br>
* 매개변수 곡면 S:  
  $$ \iint_S \mathbf{F}\cdot d\mathbf{S}
     = \iint_D \mathbf{F}\bigl(\mathbf{r}(u,v)\bigr)\cdot(\mathbf{r}_u\times\mathbf{r}_v)\,dA $$<br>
* z = f(x,y) 형태의 S:  
  $$ \iint_S \mathbf{F}\cdot d\mathbf{S}
     = \iint_D \Bigl(-P\,f_x - Q\,f_y + R\Bigr)\,dA $$<br>
<br>
## 적분 정리들의 유사성<br>
<br>
선적분의 기본 정리, 그린 정리, 스토크스 정리, 발산 정리는 모두 내부의 도함수 적분과 경계의 함수 값 차이를 연결합니다.<br>
<br>
## 2차 동차 선형 미분방정식 풀이<br>
<br>
상수 계수 2차 동차 방정식<br>
$$ a y'' + b y' + c y = 0,\quad a\neq0 $$<br>
특성방정식  
$$ a r^2 + b r + c = 0 $$<br>
<br>
### 근에 따른 해<br>
<br>
* 서로 다른 실근 r₁, r₂:  
  $$ y = C_1 e^{r_1 x} + C_2 e^{r_2 x} $$<br>
* 중근 r:  
  $$ y = C_1 e^{r x} + C_2 x e^{r x} $$<br>
* 복소근 α±iβ:  
  $$ y = e^{α x}\bigl(C_1\cos(βx)+C_2\sin(βx)\bigr) $$<br>
<br>
### 비동차 선형 방정식의 특수해<br>
<br>
비동차 방정식 $a y''+b y'+c y = G(x)$의 특수해는 G(x)의 형태에 따라 가정하고 계수 결정합니다.  
* 다항식 G(x): 다항식 가정  
* $e^{k x}$: $A e^{k x}$ 가정  
* $\cos kx$, $\sin kx$: $A\cos kx+B\sin kx$ 가정  
* 이미 동차 해와 같으면 x를 곱해서 수정<br>
<br>
### 일반해<br>
<br>
동차 해와 특수해의 합으로 표현합니다:<br>
$$ y = y_{\text{complementary}} + y_{\text{particular}} $$<br>
<br>
# 2계 미분방정식<br>
<br>
## 초기값 문제와 경계값 문제<br>
<br>
* 초기값 문제: $y(x_0)=y_0$, $y'(x_0)=y_1$ 조건에서 해 찾기<br>
* 경계값 문제: $y(x_0)=y_0$, $y(x_1)=y_1$ 조건에서 해 찾기<br>
<br>
## 비동차 방정식과 상보 방정식<br>
<br>
비동차 방정식 $a y''+b y'+c y=G(x)$의 상보 방정식은 $a y''+b y'+c y=0$이고, 상보 해와 특수해 합으로 일반해를 구합니다.<br>
<br>
## 매개변수 변환법 (Variation of Parameters)<br>
<br>
동차 해 $y^c=C_1y_1+C_2y_2$일 때, 특수해를 $y_p=u_1y_1+u_2y_2$로 가정하고<br>
1. $u_1'y_1+u_2'y_2=0$<br>
2. $u_1'y_1'+u_2'y_2'=G(x)/a$<br>
연립 후 적분하여 $u_1,u_2$ 구함<br>
<br>
## 활용 예시 (본문 생략)<br>
<br>
# 용수철의 운동<br>
<br>
## 미정계수법<br>
<br>
질량 m인 물체가 용수철에 매달린 단순 조화 운동은<br>
$$ m\,\frac{d^2x}{dt^2} + k\,x = 0 $$<br>
여기서 m은 질량, k는 용수철 상수, x는 변위입니다.  
미정계수법은 비동차 방정식의 특수해를 G(x) 형태로 가정하여 계수를 결정하는 방법입니다.<br>
