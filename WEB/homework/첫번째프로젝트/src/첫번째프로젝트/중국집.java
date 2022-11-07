package 첫번째프로젝트;

public class 중국집{
	
	static void 메뉴선택() {
		System.out.println("여기는 중국집입니다.");
		System.out.println("메뉴를 선택해주세요.");
	}
	
	public static void main(String[] args) {
		
		메뉴 m1 = new 메뉴(); //인스턴스 변수 
		메뉴 m2 = new 메뉴(); //똑같은 메뉴 클래스에 대한 메뉴 객체는 모두 다르다.
		메뉴 m3 = new 메뉴();
		메뉴 m4 = new 메뉴();
		메뉴선택();
		m1.setName("자장면");
		m1.주문();
		System.out.println("주문하신 " + m1.getName() + " 입니다." + "\n");
		
		메뉴선택();
		m2.setName("짬뽕");
		m2.주문();
		System.out.println("주문하신 " + m2.getName() + " 입니다."+ "\n");
		
		메뉴선택();
		m3.setName("탕수육");
		m3.주문();
		System.out.println("주문하신 " + m3.getName() + " 입니다."+ "\n");
		m4.setName("잡채밥");
		m4.주문();

		
	}
}

