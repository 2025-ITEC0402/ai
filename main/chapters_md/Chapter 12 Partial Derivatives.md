# Chapter 12 Partial Derivatives<br>
<br>
# 미분 (Differential)<br>
<br>
## 선형화와 선형 근사 (Linearization and Linear Approximation)<br>
<br>
함수 f의 (a, b)에서의 선형화는 z = f(x, y)의 (a, b, f(a, b))에서의 접평면을 그래프로 가지는 선형 함수입니다. 다음과 같이 표현됩니다.<br>
<br>
$$ L(x, y) = f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b) $$<br>
<br>
(a, b)에서의 선형 근사는 함수값 f(x, y)를 (x, y)가 (a, b) 근처에 있을 때 접평면의 값 L(x, y)로 근사하는 것을 의미합니다.<br>
<br>
$$ f(x, y) \approx f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b) $$<br>
<br>
기하학적으로, 선형 근사는 (x, y)가 (a, b) 근처에 있을 때, f(x, y) 값을 (a, b, f(a, b))에서의 그래프에 대한 접평면의 L(x, y) 값으로 근사할 수 있다는 것을 의미합니다.<br>
<br>
## 미분가능성 (Differentiability)<br>
<br>
함수 z = f(x, y)가 (a, b)에서 미분 가능하다는 것은 Δz를 다음과 같은 형태로 표현할 수 있다는 것을 의미합니다.<br>
<br>
$$ \Delta z = f_x(a, b) \Delta x + f_y(a, b) \Delta y + \epsilon_1 \Delta x + \epsilon_2 \Delta y $$<br>
<br>
여기서 ε₁과 ε₂는 (Δx, Δy)→(0,0)일 때 0으로 수렴합니다. 즉, 미분 가능한 함수는 (x, y)가 (a, b) 근처에 있을 때 선형 근사가 좋은 근사가 되는 함수입니다.<br>
<br>
## 음함수 미분 (Implicit Differentiation)<br>
<br>
F(x, y, z) = 0 형태의 방정식으로 x와 y의 함수로서 z가 암시적으로 정의되어 있고, F가 미분 가능하며 F_z ≠ 0일 때, ∂z/∂x와 ∂z/∂y는 다음과 같이 구할 수 있습니다.<br>
<br>
$$ \frac{\partial z}{\partial x} = -\frac{F_x}{F_z}, \quad \frac{\partial z}{\partial y} = -\frac{F_y}{F_z} $$<br>
<br>
## 방향 도함수 (Directional Derivative)<br>
<br>
단위 벡터 **u** = ⟨a, b⟩ 방향으로의 f(x, y)의 (x₀, y₀)에서의 방향 도함수는 다음과 같습니다.<br>
<br>
$$ D_{\mathbf{u}}f(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + ha, y_0 + hb) - f(x_0, y_0)}{h} $$<br>
<br>
이는 **u** 방향으로 (x₀, y₀)에서 f의 변화율로 해석할 수 있습니다. 기하학적으로, 그래프 위의 점 P(x₀, y₀, f(x₀, y₀))에서, P를 지나는 **u** 방향의 수직 평면과 그래프의 교선 C의 접선 기울기가 방향 도함수입니다.<br>
<br>
# 미분 (Differentiation)<br>
<br>
## 접선의 기울기와 미분가능성<br>
<br>
x, y에 대한 함수 f가 점 (a, b) 근처에서 편도함수 fₓ와 fᵧ를 가지고, 이들이 (a, b)에서 연속이면 f는 (a, b)에서 미분 가능합니다.<br>
<br>
## 미분 (Differentials)<br>
<br>
z = f(x, y)라면 dx, dy, dz는 무엇일까요?<br>
<br>
dx와 dy는 독립 변수이며 어떤 값이든 가질 수 있습니다. f가 미분 가능하다면 dz는 다음과 같이 정의됩니다.<br>
<br>
$$ dz = f_x(x, y)\,dx + f_y(x, y)\,dy $$<br>
<br>
## 연쇄 법칙 (Chain Rule)<br>
<br>
z = f(x, y)이고 x, y가 하나의 변수 t의 함수인 경우(x=g(t), y=h(t))라면<br>
<br>
$$ \frac{dz}{dt} = f_x \frac{dx}{dt} + f_y \frac{dy}{dt} $$<br>
<br>
x, y가 두 변수 s, t의 함수인 경우(x=g(s,t), y=h(s,t))라면<br>
<br>
$$ \frac{\partial z}{\partial s} = f_x \frac{\partial x}{\partial s} + f_y \frac{\partial y}{\partial s} $$<br>
<br>
$$ \frac{\partial z}{\partial t} = f_x \frac{\partial x}{\partial t} + f_y \frac{\partial y}{\partial t} $$<br>
<br>
## 기울기 벡터 (Gradient Vector)<br>
<br>
두 변수 함수 f: ∇f(x,y) = ⟨fₓ, fᵧ⟩ = fₓ i + fᵧ j<br>
<br>
세 변수 함수 f: ∇f(x,y,z) = ⟨fₓ, fᵧ, f_z⟩ = fₓ i + fᵧ j + f_z k<br>
<br>
## 방향 도함수와 기울기 벡터<br>
<br>
방향 도함수는 기울기 벡터와 단위벡터의 내적으로 표현됩니다:<br>
<br>
$$ D_u f(x,y) = \nabla f(x,y)\cdot u, \quad D_u f(x,y,z) = \nabla f(x,y,z)\cdot u $$<br>
<br>
## 그래디언트의 기하학적 의미<br>
<br>
그래디언트 벡터는 함수가 가장 빠르게 증가하는 방향을 가리키며, 등고선(등고면)에 수직입니다.<br>
<br>
## 함수의 극값<br>
### 지역/절대 최대·최소·안장점 정의<br>
(a) 지역최댓값: 근처의 모든 점에서 f(x,y) < f(a,b)<br>
(b) 절대최댓값: 정의역 전체에서 f(x,y) < f(a,b)<br>
(c) 지역최솟값: 근처 모든 점에서 f(x,y) > f(a,b)<br>
(d) 절대최솟값: 정의역 전체에서 f(x,y) > f(a,b)<br>
(e) 안장점: 국소최대이자 국소최소인 점<br>
<br>
## 닫힌·유계 집합과 극값 정리<br>
<br>
R²의 닫힌 유계 집합 D에서 연속인 f는 D에서 절대최댓값과 절대최솟값을 갖습니다.<br>
<br>
### 극값 찾기 절차<br>
1. 내부 임계점에서 f값 계산<br>
2. 경계에서 f의 극값 계산<br>
3. 가장 큰 값이 절대최댓값, 가장 작은 값이 절대최솟값<br>
<br>
## 라그랑주 승수법<br>
<br>
제약조건 t(x,y,z)=k 하에서 ∇f = λ ∇t를 만족하는 점들에서 f를 계산하여 극값을 찾습니다.<br>
<br>
## 이중 리만 합과 이중 적분<br>
<br>
R=[a,b]×[c,d]에서의 이중 리만 합:<br>
<br>
$$ \sum_{i=1}^m\sum_{j=1}^n f(x_{ij}^*,y_{ij}^*)\,\Delta A_{ij} $$<br>
<br>
이중 적분 정의:<br>
<br>
$$ \iint_R f(x,y)\,dA = \lim_{m,n\to\infty} \sum_{i,j} f(x_{ij}^*,y_{ij}^*)\,\Delta A_{ij} $$<br>
<br>
– 유형 I 영역 D: ∫_a^b ∫_{t₁(x)}^{t₂(x)} f(x,y)\,dy\,dx<br>
<br>
– 유형 II 영역 D: ∫_c^d ∫_{u₁(y)}^{u₂(y)} f(x,y)\,dx\,dy<br>
<br>
## 이중 적분 성질<br>
<br>
선형성, 비교, 영역 분할, 면적 계산, 부등식 추정 등 다양한 성질이 적용됩니다.<br>
<br>
## 중점법칙과 평균값<br>
<br>
– 중점법칙: 부분영역 중점에서 평가하여 근사<br>
– 평균값: f_ave = (1/A(R)) ∬_R f(x,y) dA<br>
<br>
## 일반 영역 이중 적분<br>
<br>
D를 포함하는 직사각형 R 정의, F(x,y)=f(x,y) (x,y∈D), 0 (else)<br>
– ∬_D f = ∬_R F<br>
<br>
## 주기 함수와 푸리에 급수 개념<br>
<br>
주기 T 함수 f(t): f(t+T)=f(t)<br>
푸리에 급수:<br>
<br>
$$ f(t) = \frac{a_0}{2} + \sum_{n=1}^\infty\bigl[a_n\cos(n\omega t)+b_n\sin(n\omega t)\bigr],\ \omega=\tfrac{2\pi}{T} $$<br>
<br>
## 푸리에 계수 계산<br>
<br>
$$ a_0=\tfrac{2}{T}\int_0^T f(t)\,dt,\quad a_n=\tfrac{2}{T}\int_0^T f(t)\cos(n\omega t)\,dt,\quad b_n=\tfrac{2}{T}\int_0^T f(t)\sin(n\omega t)\,dt $$<br>
<br>
## 푸리에 급수 수렴성<br>
<br>
디리클레 조건(유한 불연속·유한 극값)이면 수렴하며, 불연속점에서는 좌·우극한 평균으로 수렴합니다.<br>
