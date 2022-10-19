package com.study.board.controller;

import com.study.board.entity.Board;
import com.study.board.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Controller
public class BoardController {
    @Autowired
    private BoardService boardService;

    @GetMapping("/board/write") //local:8090/board/write
    public String boardWrite(){
        return"boardwrite"; //자신이 설정하고싶은 html을 써준다.
    }
    @PostMapping("/board/writepro") //local:8090/board/writepro
    public String boardWritePro(Board board, Model model, MultipartFile file) throws Exception{
        System.out.println("writepro");
        boardService.write(board, file);
        model.addAttribute("message","글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl","/board/list");
        return "message";
    }

//    @RequestMapping(method = RequestMethod.POST, path="/board/testpost")
    @PostMapping("/board/testpost")
    public String testPost(Model model) {
        System.out.println("testPost");
        return "boardList";
    }

    @GetMapping("/board/list")
    public String boardList(Model model){
        System.out.println("/board/list/");
        model.addAttribute("list",boardService.boardList());
        return "boardList";
    }
    @GetMapping("/board/view") // localhost:8070/board/view?id=1
    public String boardView(Model model, Integer id) {
        model.addAttribute("board",boardService.boardView(id));
        return "boardview";
    }

    @GetMapping("/board/delete")
    public String boardDelete(Integer id){
        boardService.boardDelete(id);
        return "redirect:/board/list";
    }

    @GetMapping("/board/modify/{id}")
    public String boardModify(@PathVariable("id") Integer id,Model model){
        System.out.println("/board/modifi/");
        model.addAttribute("board",boardService.boardView(id));
        return "boardmodify";
    }
    @PostMapping("/board/update/{id}")
    public String boardUpdate(@PathVariable("id") Integer id, Board board, MultipartFile file) throws Exception{
        System.out.println("/board/update/");
        Board boardTemp = boardService.boardView(id);
        boardTemp.setTitle(board.getTitle()); //board에 타이틀을 가져와라
        boardTemp.setContent(board.getContent());
        System.out.println(boardTemp.getTitle());
        System.out.println(boardTemp.getContent());
        boardService.write(boardTemp, file);
        return "redirect:/board/list";
    }
}
