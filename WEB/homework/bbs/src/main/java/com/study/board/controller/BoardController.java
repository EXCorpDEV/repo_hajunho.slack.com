package com.study.board.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class BoardController {

    @GetMapping("/") //여기다가 이동 경로를 써주면 된다
    @ResponseBody
    public String main(){
        return "Hello World";
    }
}
