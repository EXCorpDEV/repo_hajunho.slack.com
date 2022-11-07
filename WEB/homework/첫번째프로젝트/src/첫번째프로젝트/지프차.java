package 첫번째프로젝트;

public class 지프차 extends 자동차 {
	
	public 지프차() {
		this.set자동차이름("지프차");
	}
	
	public String 옵션 = "보조바퀴, 뒤에 달린 ";

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		지프차 jeep = new 지프차();
		System.out.println(jeep.바퀴휠모양);
		System.out.println(jeep.외관);
//		System.out.println(jeep.핸들);
		System.out.println(jeep.옵션);
	}

}
