package com.study.board.controller;

import com.study.board.entity.Board;
import com.study.board.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;


@Controller
public class BoardController {
    @Autowired
    private BoardService boardService;

    @GetMapping("/board/write") //local:8070/board/write
    public String boardWrite(){
        return"boardwrite"; //자신이 설정하고싶은 html을 써준다.
    }
    @PostMapping("/board/writepro") //local:8090/board/writepro
    public String boardWritePro(Board board){
        boardService.write(board);
        return "redirect:/board/list";
    }

    @PostMapping("/board/testpost")
    public String testPost(Model model) {
        System.out.println("testPost");
        return "boardList";
    }
    @GetMapping("/board/list")
    public String boardList(Model model,
                            @PageableDefault(page = 0, size = 10, sort = "id", direction = Sort.Direction.DESC) Pageable pageable,
                            String searchKeyword) {

        Page<Board> list = null;

        if (searchKeyword == null){
            list = boardService.boardList(pageable);
        }else{
            list = boardService.boardSearchList(searchKeyword,pageable);
        }

        int nowPage = list.getPageable().getPageNumber() + 1; //페이지가 0에서 부터 시작하기 때문에 1을 추가 해줘야한다.
        int startPage = Math.max(nowPage - 4,1); //1 - 4 면 마이너스니깐 그때 1을 선택하게 만들어준다.
        int endPage = Math.min(nowPage + 5,list.getTotalPages());

        model.addAttribute("list", list);
        model.addAttribute("nowPage",nowPage);
        model.addAttribute("startPage",startPage);
        model.addAttribute("endPage",endPage);
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
    public String boardUpdate(@PathVariable("id") Integer id, Board board){
        System.out.println("/board/update/");
        Board boardTemp = boardService.boardView(id);
        boardTemp.setTitle(board.getTitle()); //board에 타이틀을 가져와라
        boardTemp.setContent(board.getContent());
        System.out.println(boardTemp.getTitle());
        System.out.println(boardTemp.getContent());
        boardService.write(boardTemp);
        return "redirect:/board/list";
    }
}
