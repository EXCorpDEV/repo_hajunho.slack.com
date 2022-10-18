package com.study.board.entity;

import lombok.Data;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity //db에 있는 테이블을 의미한다
@Data // 이것을 넣으면 원하는 값만 추출할 수 있다.
public class Board {
    @Id //프라이머리키를 의미하는것이다.
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String title;
    private String content;
    private String filename;
    private String filepath;
}
