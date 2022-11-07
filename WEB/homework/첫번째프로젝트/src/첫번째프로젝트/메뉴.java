package 첫번째프로젝트;

public class 메뉴 {
	
	private String name;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	
	public void 주문() {
		if(this.name == "자장면")
			System.out.println("자장면이요");
		else {
			if(this.name == "짬뽕")
				System.out.println("짬뽕이요");
			else {
				if(this.name == "탕수육")
					System.out.println("탕수육이요");
				else
					System.out.println("주문하신 음식이 없습니다.");
			}
		}
	}
}
