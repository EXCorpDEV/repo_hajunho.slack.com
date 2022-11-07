package com.study.board.controller;

import com.study.board.entity.Board;
import com.study.board.service.BoardService;
import org.springframework.beans.factory.annotation.Autowired;
<<<<<<< .merge_file_RFbyrG
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.ResponseBody;
=======
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

>>>>>>> .merge_file_XcDi42

@Controller
public class BoardController {
    @Autowired
    private BoardService boardService;

<<<<<<< .merge_file_RFbyrG
    @GetMapping("/board/write") //local:8090/board/write
=======
    @GetMapping("/board/write") //local:8070/board/write
>>>>>>> .merge_file_XcDi42
    public String boardWrite(){
        return"boardwrite"; //자신이 설정하고싶은 html을 써준다.
    }
    @PostMapping("/board/writepro") //local:8090/board/writepro
<<<<<<< .merge_file_RFbyrG
    public String boardWritePro(Board board,Model model){
        boardService.write(board);

        model.addAttribute("message","글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl","/board/list");
        return "message";
    }

    @GetMapping("/board/list")
    public String boardList(Model model){
        model.addAttribute("list",boardService.boardList());
        return "boardList";
    }
    @GetMapping("/board/view") // localhost:8070/board/view?id=1
    public String boardView(Model model, Integer id){
=======
    public String boardWritePro(Board board, Model model){
//        System.out.println(board.getTitle()); -> board테이블 @Data를 넣었기 때문에 특정값을 얻을 수 있다.
        boardService.write(board);
        model.addAttribute("message","글 작성이 완료되었습니다.");
        model.addAttribute("searchUrl","/board/list");

        return "message";
    }

    @PostMapping("/board/testpost")
    public String testPost(Model model) {
        System.out.println("testPost");
        return "boardList";
    }
    @GetMapping("/board/list")
    public String boardList(Model model,
                            @PageableDefault(page = 0, size = 5, sort = "id", direction = Sort.Direction.DESC) Pageable pageable,
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
>>>>>>> .merge_file_XcDi42
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
<<<<<<< .merge_file_RFbyrG
=======
        System.out.println("/board/modifi/");
>>>>>>> .merge_file_XcDi42
        model.addAttribute("board",boardService.boardView(id));
        return "boardmodify";
    }
    @PostMapping("/board/update/{id}")
<<<<<<< .merge_file_RFbyrG
    public String boardUpdate(@PathVariable("id") Integer id, Board board){

        Board boardTemp = boardService.boardView(id);
        boardTemp.setTitle(board.getTitle());
        boardTemp.setContent(board.getContent());

        boardService.write(boardTemp);

        return "redirect:/board/list";

=======
    public String boardUpdate(@PathVariable("id") Integer id, Board board,Model model){
        System.out.println("/board/update/");
//        boardView에 있는걸 boardTemp에 넘겨라
        Board boardTemp = boardService.boardView(id);
//        boardTemp에서 수정한것을 가져와라
        boardTemp.setTitle(board.getTitle()); //board에 타이틀을 가져와라
        boardTemp.setContent(board.getContent());

        System.out.println(boardTemp.getTitle());
        System.out.println(boardTemp.getContent());
//        저장을 하는것이다.
        boardService.write(boardTemp);

        model.addAttribute("message","글이 수정되었습니다.");
        model.addAttribute("searchUrl","/board/list");
        return "message";
>>>>>>> .merge_file_XcDi42
    }
}
