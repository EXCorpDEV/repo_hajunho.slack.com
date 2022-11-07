package 첫번째프로젝트;

import java.util.ArrayList;

public class 자동차 {
	
	//상태(member variable, attributes, properties) 
	//행동(function, method, action, ...)

	private Integer 속도 = 0;
	private String 자동차이름 = "기본자동차";
	public String 외관 = "멋짐";
	public String 바퀴휠모양 = "다이아몬드";
	public String 핸들 = "스포츠카핸들";
	public String 최신형스피커 = "보스껄로 빵빵함";
	

	public void 앞으로간다() {
		this.set속도(this.속도를_바꾼다() + 10);

	}

	public void 뒤로간다() {
		this.set속도(this.속도를_바꾼다() - 10);
	}

	public void 좌회전() {

	}

	public void 우회전() {

	}

	public Integer 속도를_바꾼다() {
		return 속도;
	}

	public void set속도(Integer 속도) {
		
		System.out.println("if문 전의 속도" + 속도);
		
		if (속도 > 200) {
			
			this.속도 = 0;
			System.out.println("야 바보들아 니네들 속도 200까지인데 그것보다 큰값 넣었어!!");
		} else if(속도 < 0) {
			this.속도 = 0;
		}
		else {
			System.out.println("this속도 지정전 파라미터 속도" + 속도);
			this.속도 = 속도;
		}
	}

	public String get자동차이름() {
		return 자동차이름;
	}

	public void set자동차이름(String 자동차이름) {
		this.자동차이름 = 자동차이름;
	}
	
	public static void main(String[] args) {
		
		ArrayList al = new ArrayList<자동차>();
		
		
		String 찾고싶은자동차 = "ㅈ덜지ㅓㄹㅈㄹ";
		
		자동차[] car = new 자동차[4];
		
//		car[0] = new 스마트();
//		car[1] = new 장갑차();
//		car[2] = new 제네시스();
//		car[3] = new 지프차();
		al.add(new 스마트());
		al.add(new 지프차());
		al.add(new 장갑차());
		al.add(new 제네시스());
//		al.add(new 제네시스());
//		al.add(new 장갑차());
//		
		
//		for(int i=0; i<car.length; i++) {
//			if(car[i].get자동차이름() == 찾고싶은자동차 ) {
//				System.out.println("자동차를 찾았다." + car[i].get자동차이름());
//			}
//		}
		
//Casting 방법 (String)
//		(Integeger)
//		(Double)
		
		int index = 9999;
		
		for(int i=0; i<al.size();i++) {
			if( ((자동차)(al.get(i))).get자동차이름() == "제네시스" ) {
				index = i;
				al.remove(index);
			}
		}
		
		for(Object a : al ) {
			System.out.println(((자동차)a).get자동차이름());
			
		}
		
	}

}
