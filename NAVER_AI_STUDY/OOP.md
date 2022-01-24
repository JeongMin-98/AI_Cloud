
## python naming rule
- snake_case: 함수와 변수명 사용
- CamelCase: 띄워쓰기 부분에 대문자

# 클래스와 객체

+ 객체: 속성와 행동을 가짐
+ OOP는 객체 개념을 프로그램으로 표현, **속성은 변수, 행동은 함수**로 표현됨
+ 클래스: 객체의 설계도
+ 인스턴스: 실제 구현체


## 클래스
### Attribute 추가하기
+ __init__, self와 함께 추가 
+ __init__은 **객체 초기화 예약 함수**

###클래스의 매직메소드 
+ __는 특수한 예약 함수나 변수 그리고 함수명 변경으로 사용
+ 여러 종류의 매직메소드가 존재

### Private 변수 선언
+ < self.__items > 로 선언 변수명 앞에 __을 붙여 선언

### 기타
+ @property: 숨겨진 변수를 반환하게 해줌, 함수를 변수처럼 호출가능

### decorater
1. first-class objects
    + 일등함수 또는 일급객체
    + 변수나 데이터 구조에 할당이 가능한 객체
    + 파라미터로 전달이 가능 + 리턴 값으로 사용
    + map(f, ex) => 함수가 파라미터처럼 사용됨
2. inner function
    + inner function을 return 값으로 반환
    + ex) closure
3. 
### method 구현하기
+ method 추가는 기존 함수와 같으나, 반드시 self를 추가해야만 class 함수로 인정됨
+ **self는 instance 자체이다**

oob.py 보기
구현 가능한 oop 만들기 -  메모장 만들기

- 메모를 정리하는 프로그램
- 사용자는 메모에 뭔가를 _적을 수 있다_. 
- 메모에는 **content**가 있고 내용을 _제거_ 할 수 있다. 
- 두개의 _메모를 하나로 합칠 수 있다_. 

memo
+ method : write_content, remove_all
+ variable: content

memo book
+ method : add_note, remove_note, get_number_of_pages
+ variable: title, page_number, notes

<hr>

## 객체 지향 언어 특징

### Inheritance (상속)
+ 부모 클래스로 부터 속성과 method를 물려받은 자식 클래스를 생성하는 것
+ super(): 자기 자신의 부모클래스의 속성들을 물러오는것 (부모 객체 사용)

### Polymorphism (다형성)
+ 같은 이름의 메소드의 내부 로직을 다르게 작성
+ Dynamic Typing 특성으로 인해 파이썬에서는 같은 부모클래스의 상속에서 주로 발생함
+ 상속을 받을 때 다른 역할을 하도록 함수를 만들 수 있다. 

### visibility (가시성)
+ 캡슐화, 정보은닉
+ 간섭/정보공유의 최소화
+ 인터페이스만 알면 서로의 클래스의 속성을 이용할 수 있도록 함





