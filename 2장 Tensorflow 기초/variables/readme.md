# Tensorflow 변수

tf에서 다루는 모든 데이터는 텐서 형식으로 되어있다. 또한 값을 변경가능한 변수와 값을 변경할 수 없는 상수를 구분하여 사용해야 한다.

**하나의 값을 갖는 변수 생성**
하나의 값 만을 저장하는 변수는 tf의 Variable메소드를 사용한다.Variable의 인자들을 살펴보자.

	Variable(initial_value, name=<optional-name>)
	
먼저 initial_value는 해당 변수에 들어가는 초기치를 뜻한다. 변수는 모델에서 학습되어가면서 값이 변경되는데 이때 변수의 초기값을 결정한다. 
name은 변수의 이름을 지칭하여 내가 만든 모델을 시각화했을때 변수가 어떻게 변하는지 추적이 가능하도록 해준다.

**상수 생성**
가끔은 학습률 같이 변하지 않는 상수를 정의해야 할 필요가 있다. 상수는 tf의 constant메소드를 사용한다.

	constant(value, name='const')
	
인자들은 위에 설명한 Variable과 같은 역할을 하지만, 상수의 값은 변경할 수 없다는 점에 주목하자.

**여러개의 값을 담는 텐서 생성**
Variable 과 constant는 하나의 값 만을 저장할 수 있다. 그러나 우리가 가진 데이터는 적게는 수백개, 많게는 수백만개의 데이터를 다루기 때문에 하나의 변수에 하나의 값을 저장하는것은 비효율적이다. 따라서 tf는 하나의 텐서에 여러개의 값을 저장하는 식으로 데이터를 관리한다.
	
	placeholder(dtype, shape=None, name=None)
	
dtype은 텐서에 저장되는 값의 형태를 말한다.  tf에는 int32, float32등 여러개의 데이터 형식을 정의하고 있어서 자신의 데이터에 맞는 값을 설정해 주면 된다. shape은 텐서의 모양을 말한다. 이때 모양을 튜플 형태로 입력하게 되며, None으로 설정하면 해당 방향으로 합친다는 뜻이 된다.
> Tip
>텐서의 모양은 다차원 배열을 생각하면 상상하기 편하다.흔히 이미지 처리에서 자주 보이는 모양은 (5, 64, 64, 3)과 같은 형태를 띄는데, 이는 5개의 64X64 크기의 이미지의 각 픽셀이 RGB 세개의 값을 갖고있는 형태를 뜻한다. 따라서 (None, 64, 64, 3)과 같이 텐서의 모양을 정의하면, 64X64의 RGB이미지를 개수 제한없이 저장한다는 뜻이다.

## 본격적인 사용법
tf의 변수는 단순한 변수처럼 생각하면 안된다. tf의 모든 변수는 하나의 세션에서만 동작하며, 하나의 세션이 끝나면 모든 정보를 잃어버린다. Session클래스를 통해 세션을 열어주고, 모든 연산은 Session의 run메소드를 이용해 처리한다. 

우리가 프로그래밍을 처음 배웠을때, 변수를 초기화 하지 않으면 변수 안에는 쓰레기값이 들어있다고 배웠다. tf의 변수도 마찬가지로 초기화를 해주지 않으면 쓰레기값이 들어가 있어서 변수를 이용할 수가 없다. 우리가 변수를 만들 때 지정해준 초기값은 Define and Run 방식에 따라 형태를 지정한 것일뿐, 실제의 값이 아니다. 변수의 초기화는 run메소드로 global_variables_initializer를 호출하면 된다.
> Tip
>placeholder는 변수와 다르게 run메소드를 통해 텐서와 함께 feed_dict인자에 딕셔너리 형태로 값을 지정해 준다

위의 정보를 바탕으로 의사코드를 통해 어떤 식으로 운용이 되는지 확인해 보자.

	#tensor and variables defined
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#do something with variable
		
		sess.run(placeholer, feed_dict={placeholder:data})
		#do domething with placeholder
		
