package com.study.board.controller;

import com.study.board.entity.Board;
import com.study.board.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class BoardController {
    @Autowired
    private BoardService boardService;

    @GetMapping("/board/write") //local:8090/board/write
    public String boardWrite(){
        return"boardwrite";
    }
    @PostMapping("/board/writepro")
    public String boardWritePro(Board board){

        boardService.write(board);

        return "";
    }
}
